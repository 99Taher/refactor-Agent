import os
import re
import shutil
import requests
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

MAX_FILES = 10
MAX_WORKERS = 5
GROQ_TIMEOUT = 30
CLONE_DEPTH = 500  # → Essaie 0 ou 1000 si toujours 0 fichiers malgré refs OK

class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch: str = "auto-refactor"

def clone_repo(repo_url: str, branch: str, base_ref: str) -> str:
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")

    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)

    logger.info(f"Clonage de '{branch}' --depth={CLONE_DEPTH} --no-single-branch...")
    subprocess.run([
        "git", "clone",
        "--depth", str(CLONE_DEPTH),
        "--no-single-branch",
        "-b", branch,
        clone_url, repo_name
    ], check=True)

    os.chdir(repo_name)

    logger.info("Configuration remote pour fetcher toutes les branches...")
    subprocess.run(["git", "remote", "set-branches", "origin", "*"], check=True, capture_output=True)

    # Fetch large pour peupler les refs
    logger.info("Fetch large shallow pour initialiser les remote refs...")
    subprocess.run(["git", "fetch", "origin", "--depth=" + str(CLONE_DEPTH)], capture_output=True, text=True)

    # Fetch explicite avec refspec → force création refs/remotes/origin/{base_ref}
    logger.info(f"Fetch forcé avec refspec pour {base_ref}...")
    fetch_result = subprocess.run([
        "git", "fetch", "origin",
        f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
        f"--depth={CLONE_DEPTH}"
    ], capture_output=True, text=True)

    if fetch_result.returncode != 0:
        logger.warning(f"Fetch refspec a échoué : {fetch_result.stderr.strip()}")
        # Fallback : update-ref manuel depuis FETCH_HEAD
        subprocess.run(["git", "update-ref", f"refs/remotes/origin/{base_ref}", "FETCH_HEAD"], check=False)

    # Deepen pour capturer le merge-base si divergence ancienne
    logger.info("Deepen du clone...")
    subprocess.run(["git", "fetch", "--deepen=400"], check=False, capture_output=True)

    # Debug précoce
    logger.info("DEBUG CLONE - git branch -r :")
    logger.info(subprocess.getoutput("git branch -r") or "Aucune branche remote")

    logger.info(f"DEBUG CLONE - show-ref | grep {base_ref} :")
    logger.info(subprocess.getoutput(f"git show-ref | grep {base_ref}") or "Aucune ref trouvée")

    os.chdir("..")
    return repo_name

def get_changed_files(repo_path: str, base_ref: str):
    os.chdir(repo_path)
    try:
        base_local = f"base-{base_ref}"

        logger.info(f"Création branche locale {base_local} depuis FETCH_HEAD ou origin...")
        create_result = subprocess.run([
            "git", "branch", base_local, f"refs/remotes/origin/{base_ref}"
        ], capture_output=True, text=True)

        if create_result.returncode != 0:
            # Fallback sur FETCH_HEAD
            logger.warning("Branche depuis origin échoue → fallback FETCH_HEAD")
            subprocess.run(["git", "branch", base_local, "FETCH_HEAD"], capture_output=True)

        # Debug essentiel
        logger.info("=== DEBUG DIFF ===")
        logger.info("Toutes branches (git branch -a) :")
        logger.info(subprocess.getoutput("git branch -a") or "Aucune branche")

        logger.info(f"Log {base_local} (3 derniers) :")
        logger.info(subprocess.getoutput(f"git log -n 3 --oneline {base_local} || echo 'Aucun log'"))

        logger.info("Log HEAD (3 derniers) :")
        logger.info(subprocess.getoutput("git log -n 3 --oneline HEAD"))

        # Diff sur branches locales (plus fiable en shallow)
        files = []
        for mode, cmd in [
            ("three-dot local", f"git diff --name-only {base_local}...HEAD"),
            ("two-dot local", f"git diff --name-only {base_local}..HEAD"),
            ("three-dot origin", f"git diff --name-only origin/{base_ref}...HEAD"),
        ]:
            logger.info(f"Essai {mode}: {cmd}")
            try:
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
                candidates = [f.strip() for f in output.splitlines() if f.strip()]
                logger.info(f"{mode} → {len(candidates)} fichiers")
                if candidates:
                    logger.info(f"Exemples : {candidates[:8]}")
                    files = candidates
                    break
            except subprocess.CalledProcessError as e:
                err = e.output.decode(errors='ignore').strip()
                logger.warning(f"{mode} échoué : {err}")

        kt_files = [f for f in files if f.endswith(".kt") and os.path.exists(f)]
        logger.info(f"Fichiers .kt à refactoriser : {len(kt_files)} → {kt_files[:5] or 'aucun'}")

        return kt_files

    except Exception as e:
        logger.exception("Erreur lors du calcul des changements")
        return []
    finally:
        os.chdir("..")

# Les fonctions refactor_file et commit_and_push restent inchangées (bonnes)

def refactor_file(repo_path: str, filepath: str) -> str:
    full_path = os.path.join(repo_path, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()

        if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
            return f"{filepath} - Pas de logs détectés"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

        prompt = (  # ton prompt est OK, inchangé
            "You are a Kotlin Refactoring Expert.\n"
            "Your mission: Clean up and deduplicate logging in this Android code.\n\n"
            # ... reste du prompt ...
            "Return ONLY raw source code. NO markdown markers, NO explanations."
        )

        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are a Kotlin expert. Output only raw source code."},
                {"role": "user", "content": f"{prompt}\n\nCODE:\n{code}"}
            ],
            "temperature": 0
        }

        response = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        response.raise_for_status()

        new_code = response.json()["choices"][0]["message"]["content"].strip()
        new_code = re.sub(r"^```kotlin\s*|^```\s*", "", new_code, flags=re.MULTILINE)
        new_code = re.sub(r"\s*```$", "", new_code)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        return f"{filepath} → refactored"

    except requests.exceptions.Timeout:
        return f"{filepath} → Groq timeout"
    except Exception as e:
        logger.error(f"Erreur refactor {filepath}: {e}")
        return f"{filepath} → Erreur: {str(e)}"

def commit_and_push(repo_path: str):
    os.chdir(repo_path)
    try:
        subprocess.run(["git", "add", "."], check=True)
        # --allow-empty pour debug si aucun changement réel mais on veut tester push
        commit = subprocess.run(["git", "commit", "-m", "🤖 Auto refactor: centralized logging cleanup", "--allow-empty"], capture_output=True, text=True)
        if commit.returncode != 0:
            logger.warning(f"Commit skipped (peut-être vide) : {commit.stderr.strip()}")
        else:
            subprocess.run(["git", "push"], check=True)
            logger.info("Commit & push OK")
    except Exception as e:
        logger.warning(f"Commit/push problème : {e}")
    finally:
        os.chdir("..")

@app.get("/")
async def root():
    return {"message": "Refactor Agent API Is Active"}

@app.post("/refactor")
def run_refactor(request: RefactorRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="Missing env vars")

    logger.info(f"Refactor démarré → repo: {request.repo_url} | base: {request.base_ref} | head: {request.branch}")

    try:
        repo_path = clone_repo(request.repo_url, request.branch, request.base_ref)
        files = get_changed_files(repo_path, request.base_ref)

        if not files:
            logger.info("Aucun .kt modifié détecté")
            return {"status": "ok", "processed": 0, "message": "No changes"}

        files = files[:MAX_FILES] if len(files) > MAX_FILES else files

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(refactor_file, repo_path, f): f for f in files}
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                logger.info(res)

        if any("refactored" in r for r in results):
            commit_and_push(repo_path)
        else:
            logger.info("Pas de vrai changement → skip commit")

        return {"status": "success", "processed_files": results}

    except subprocess.CalledProcessError as git_err:
        logger.exception("Erreur Git")
        raise HTTPException(500, detail=f"Git error: {git_err.stderr or str(git_err)}")
    except Exception as e:
        logger.exception("Erreur inattendue")
        raise HTTPException(500, detail=str(e))
