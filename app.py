import os
import re
import shutil
import requests
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

MAX_FILES = 10
MAX_WORKERS = 5
GROQ_TIMEOUT = 30
CLONE_DEPTH = 500   # → mets à 0 temporairement si toujours 0 fichiers malgré refs OK

class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch: str = "auto-refactor"


def clone_repo(repo_url: str, branch: str, base_ref: str) -> str:
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")

    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)

    logger.info(f"Clonage {branch!r} depth={CLONE_DEPTH} --no-single-branch")
    subprocess.run([
        "git", "clone",
        "--depth", str(CLONE_DEPTH),
        "--no-single-branch",
        "-b", branch,
        clone_url, repo_name
    ], check=True)

    os.chdir(repo_name)

    logger.info("Configuration remote fetch → toutes branches")
    subprocess.run(["git", "remote", "set-branches", "origin", "*"], check=True, capture_output=True)

    logger.info("Fetch large shallow initial")
    subprocess.run(["git", "fetch", "origin", "--depth=" + str(CLONE_DEPTH)], capture_output=True)

    logger.info(f"Fetch forcé refspec → refs/remotes/origin/{base_ref}")
    fetch_cmd = [
        "git", "fetch", "origin",
        f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
        f"--depth={CLONE_DEPTH}"
    ]
    fetch_res = subprocess.run(fetch_cmd, capture_output=True, text=True)
    if fetch_res.returncode != 0:
        logger.warning(f"Fetch refspec échoué : {fetch_res.stderr.strip() or 'code ' + str(fetch_res.returncode)}")
        # Fallback
        subprocess.run(["git", "update-ref", f"refs/remotes/origin/{base_ref}", "FETCH_HEAD"], check=False)

    logger.info("Deepen du clone (pour merge-base)")
    subprocess.run(["git", "fetch", "--deepen=400"], check=False, capture_output=True)

    # Debug important
    logger.info("DEBUG CLONE - git branch -r :")
    logger.info(subprocess.getoutput("git branch -r") or "<vide>")

    logger.info(f"DEBUG CLONE - show-ref | grep {base_ref}")
    logger.info(subprocess.getoutput(f"git show-ref | grep {base_ref}") or "<aucune ref>")

    os.chdir("..")
    return repo_name


def get_changed_files(repo_path: str, base_ref: str):
    os.chdir(repo_path)
    try:
        base_local = f"base-{base_ref.replace('/', '-')}"  # évite / dans nom branche

        logger.info(f"Création branche locale {base_local}")
        # Essai depuis remote tracking
        res = subprocess.run(
            ["git", "branch", base_local, f"refs/remotes/origin/{base_ref}"],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            logger.warning("Branche depuis origin échoue → fallback FETCH_HEAD")
            subprocess.run(["git", "branch", base_local, "FETCH_HEAD"], capture_output=True)

        logger.info("=== DEBUG DIFF ===")
        logger.info("Branches (git branch -a) :")
        logger.info(subprocess.getoutput("git branch -a") or "<vide>")

        logger.info(f"Log {base_local} :")
        logger.info(subprocess.getoutput(f"git log -n 3 --oneline {base_local} || echo 'pas accessible'"))

        logger.info("Log HEAD :")
        logger.info(subprocess.getoutput("git log -n 3 --oneline HEAD"))

        files = []
        attempts = [
            ("three-dot local", f"git diff --name-only {base_local}...HEAD"),
            ("two-dot local",   f"git diff --name-only {base_local}..HEAD"),
            ("three-dot origin", f"git diff --name-only origin/{base_ref}...HEAD"),
            ("two-dot origin",   f"git diff --name-only origin/{base_ref}..HEAD"),
        ]

        for name, cmd in attempts:
            logger.info(f"→ {name}: {cmd}")
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=60)
                candidates = [line.strip() for line in out.decode().splitlines() if line.strip()]
                logger.info(f"{name} → {len(candidates)} fichiers trouvés")
                if candidates:
                    logger.info(f"exemples : {candidates[:6]}")
                    files = candidates
                    break
            except subprocess.TimeoutExpired:
                logger.error(f"{name} → timeout")
            except subprocess.CalledProcessError as e:
                err = e.output.decode(errors='ignore').strip()
                logger.warning(f"{name} → {err}")

        kt_files = [f for f in files if f.endswith(".kt") and os.path.exists(f)]
        logger.info(f".kt trouvés : {len(kt_files)} → {kt_files[:5] or 'aucun'}")

        return kt_files

    except Exception as e:
        logger.exception("Erreur calcul changements")
        return []
    finally:
        os.chdir("..")


def refactor_file(repo_path: str, filepath: str) -> str:
    full_path = os.path.join(repo_path, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()

        if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
            return f"{filepath} - pas de logs"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

        prompt = (
            "You are a Kotlin Refactoring Expert.\n"
            "Your mission: Clean up and deduplicate logging in this Android code.\n\n"
            "1. CONVERSION RULES:\n"
            " - Log.d/i/w/e OR Logr.d/i/w/e -> AppLogger.d/i/w/e(tag, msg)\n"
            " - AppSTLogger.appendLogST(STLevelLog.DEBUG, tag, msg) -> AppLogger.d(tag, msg)\n"
            " - AppSTLogger.appendLogST(STLevelLog.INFO, tag, msg) -> AppLogger.i(tag, msg)\n"
            " - AppSTLogger.appendLogST(STLevelLog.WARN, tag, msg) -> AppLogger.w(tag, msg)\n"
            " - AppSTLogger.appendLogST(STLevelLog.ERROR, tag, msg) -> AppLogger.e(tag, msg)\n\n"
            "2. DEDUPLICATION RULE (CRITICAL):\n"
            " - Merge consecutive lines of AppLogger with EXACT SAME tag and message into ONE.\n"
            " - Example: Multiple AppLogger.e(MODULE, 'text') calls become just one.\n\n"
            "3. IMPORTS:\n"
            " - ADD: 'import com.honeywell.domain.managers.loggerApp.AppLogger'.\n"
            " - REMOVE: android.util.Log, Logr, STLevelLog, and AppSTLogger imports.\n\n"
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

        resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        resp.raise_for_status()

        new_code = resp.json()["choices"][0]["message"]["content"].strip()
        new_code = re.sub(r'^```kotlin\s*', '', new_code, flags=re.M)
        new_code = re.sub(r'^\s*```$', '', new_code, flags=re.M)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        return f"{filepath} → refactored"

    except requests.exceptions.Timeout:
        return f"{filepath} → timeout Groq"
    except Exception as e:
        logger.error(f"Erreur refactor {filepath}: {e}")
        return f"{filepath} → erreur"


def commit_and_push(repo_path: str):
    os.chdir(repo_path)
    try:
        subprocess.run(["git", "add", "."], check=True)
        commit_res = subprocess.run(
            ["git", "commit", "-m", "🤖 Auto refactor logs", "--allow-empty"],
            capture_output=True, text=True
        )
        if commit_res.returncode == 0:
            subprocess.run(["git", "push"], check=True)
            logger.info("Commit & push OK")
        else:
            logger.info("Commit vide ou déjà fait → skip push")
    except Exception as e:
        logger.warning(f"Problème commit/push : {e}")
    finally:
        os.chdir("..")


@app.get("/")
async def root():
    return {"message": "Refactor Agent API Active"}


@app.post("/refactor")
def run_refactor(
    request: RefactorRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")

    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(500, "Variables d'environnement manquantes")

    logger.info(f"→ {request.repo_url} | base={request.base_ref} | branch={request.branch}")

    try:
        repo_path = clone_repo(request.repo_url, request.branch, request.base_ref)
        files = get_changed_files(repo_path, request.base_ref)

        if not files:
            return {"status": "ok", "message": "Aucun .kt modifié détecté"}

        files = files[:MAX_FILES] if len(files) > MAX_FILES else files

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(refactor_file, repo_path, f): f for f in files}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(result)

        if any("refactored" in r for r in results):
            commit_and_push(repo_path)

        return {"status": "success", "processed_files": results}

    except subprocess.CalledProcessError as e:
        logger.exception("Erreur git")
        raise HTTPException(500, f"Git error: {e.stderr or str(e)}")
    except Exception as e:
        logger.exception("Erreur générale")
        raise HTTPException(500, str(e))
