import os
import re
import shutil
import requests
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

# Paramètres de performance
MAX_FILES = 10
MAX_WORKERS = 5
GROQ_TIMEOUT = 30
CLONE_DEPTH = 500  # You can increase to 1000 or set to 0 if still issues

class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch: str = "auto-refactor"

def clone_repo(repo_url: str, branch: str, base_ref: str) -> str:
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")

    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)

    logger.info(f"Clonage shallow de '{branch}' depth={CLONE_DEPTH} --no-single-branch...")
    subprocess.run([
        "git", "clone",
        "--depth", str(CLONE_DEPTH),
        "--no-single-branch",
        "-b", branch,
        clone_url, repo_name
    ], check=True)

    os.chdir(repo_name)

    # Widen to all branches (you already have this)
    logger.info("Set remote branches to fetch all (*)...")
    subprocess.run(["git", "remote", "set-branches", "origin", "*"], check=True, capture_output=True)

    # Broad shallow fetch of all branch tips first (helps populate remotes/*)
    logger.info("Broad fetch --depth to populate remote refs...")
    broad_fetch = subprocess.run(["git", "fetch", "origin", "--depth=" + str(CLONE_DEPTH)], 
                                 capture_output=True, text=True)
    if broad_fetch.returncode != 0:
        logger.warning(f"Broad fetch warning: {broad_fetch.stderr.strip()}")

    # Explicit fetch for the base branch
    logger.info(f"Explicit fetch origin {base_ref}...")
    explicit_fetch = subprocess.run([
        "git", "fetch", "origin", base_ref, "--depth=" + str(CLONE_DEPTH)
    ], capture_output=True, text=True)
    if explicit_fetch.returncode != 0:
        logger.warning(f"Explicit fetch failed: {explicit_fetch.stderr.strip()}")

    # Fallback: force create remote-tracking ref from FETCH_HEAD if needed
    logger.info("Fallback: create remote-tracking ref if missing...")
    subprocess.run([
        "git", "update-ref", f"refs/remotes/origin/{base_ref}", "FETCH_HEAD"
    ], check=False, capture_output=True)

    # Deepen history a bit more (helps if merge-base is older than depth)
    logger.info("Deepening clone...")
    subprocess.run(["git", "fetch", "--deepen=300"], check=False, capture_output=True)

    # === DEBUG OUTPUT ===
    logger.info("=== GIT DEBUG ===")
    logger.info("Remote branches (git branch -r):")
    logger.info(subprocess.getoutput("git branch -r"))

    logger.info(f"Refs for {base_ref} (git show-ref | grep {base_ref}):")
    logger.info(subprocess.getoutput(f"git show-ref | grep {base_ref} || echo 'No refs found'"))

    logger.info("Last 3 commits on HEAD:")
    logger.info(subprocess.getoutput("git log -n 3 --oneline --decorate"))

    logger.info("=== END DEBUG ===")

    os.chdir("..")
    return repo_name

def get_changed_files(repo_path: str, base_ref: str):
    """Retourne les fichiers .kt modifiés entre base_ref et HEAD (changements du PR)."""
    os.chdir(repo_path)
    try:
        # === DEBUG: État git ===
        logger.info("Remote branches disponibles:")
        logger.info(subprocess.getoutput("git branch -r"))

        logger.info("Refs contenant la base branch:")
        logger.info(subprocess.getoutput(f"git show-ref | grep {base_ref} || echo 'Aucune ref trouvée'"))

        logger.info("Log HEAD (5 derniers commits):")
        logger.info(subprocess.getoutput("git log -n 5 --oneline --decorate --graph"))

        # Fetch HEAD aussi au cas où
        subprocess.run(["git", "fetch", "origin", "HEAD", f"--depth={CLONE_DEPTH}"], check=False)

        # Préférence : three-dot (changements depuis le point de divergence = ce que GitHub montre dans PR)
        cmd_three = f"git diff --name-only origin/{base_ref}...HEAD"
        logger.info(f"Tentative three-dot: {cmd_three}")
        try:
            output = subprocess.check_output(cmd_three, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
            files = [f.strip() for f in output.splitlines() if f.strip()]
            logger.info(f"Three-dot → {len(files)} fichiers trouvés")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Three-dot échoué: {e.output.decode(errors='ignore')}")
            files = []

        # Fallback : two-dot direct
        if not files:
            cmd_two = f"git diff --name-only origin/{base_ref}..HEAD"
            logger.info(f"Fallback two-dot: {cmd_two}")
            try:
                output = subprocess.check_output(cmd_two, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
                files = [f.strip() for f in output.splitlines() if f.strip()]
                logger.info(f"Two-dot → {len(files)} fichiers trouvés")
            except subprocess.CalledProcessError as e:
                logger.error(f"Two-dot aussi échoué: {e.output.decode(errors='ignore')}")
                files = []

        if files:
            logger.info(f"Fichiers changés (premiers 10): {files[:10]}")

        # Filtrer .kt existants
        kt_files = [f for f in files if f.endswith(".kt") and os.path.exists(f)]
        logger.info(f"Fichiers .kt trouvés : {len(kt_files)} → {kt_files[:5] or 'aucun'}...")

        return kt_files

    except Exception as e:
        logger.exception("Erreur inattendue lors du calcul des changements")
        return []
    finally:
        os.chdir("..")

def refactor_file(repo_path: str, filepath: str) -> str:
    """Appelle Groq pour refactoriser un fichier (log cleanup)."""
    full_path = os.path.join(repo_path, filepath)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()

        if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
            return f"{filepath} - Pas de logs détectés"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

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

        response = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        response.raise_for_status()

        new_code = response.json()["choices"][0]["message"]["content"].strip()
        new_code = re.sub(r"^```kotlin\s*|^```\s*", "", new_code)
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
        subprocess.run(["git", "commit", "-m", "🤖 Auto refactor: centralized logging cleanup"], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Commit & push réussi")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Commit/push a échoué (peut-être aucun changement): {e}")
    finally:
        os.chdir("..")

@app.get("/")
async def root():
    return {"message": "Refactor Agent API Is Active"}

@app.post("/refactor")
def run_refactor(
    request: RefactorRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="Missing environment variables")

    logger.info(f"Début refactor → repo: {request.repo_url} | base: {request.base_ref} | branch: {request.branch}")

    try:
        repo_path = clone_repo(request.repo_url, request.branch, request.base_ref)
        files = get_changed_files(repo_path, request.base_ref)

        if not files:
            logger.info("Aucun fichier .kt modifié trouvé → fin")
            return {"status": "success", "processed_files": [], "message": "No .kt changes detected"}

        if len(files) > MAX_FILES:
            logger.warning(f"Trop de fichiers ({len(files)}), limité à {MAX_FILES}")
            files = files[:MAX_FILES]

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(refactor_file, repo_path, f): f for f in files}
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
                logger.info(result)

        if any("refactored" in r for r in results):
            commit_and_push(repo_path)
        else:
            logger.info("Aucun changement réel après refactor → pas de commit")

        return {"status": "success", "processed_files": results}

    except subprocess.CalledProcessError as e:
        logger.exception("Erreur git")
        raise HTTPException(500, detail=f"Git error: {e.stderr or str(e)}")
    except Exception as e:
        logger.exception("Erreur globale")
        raise HTTPException(500, detail=f"Server error: {str(e)}")


