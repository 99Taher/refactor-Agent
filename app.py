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

# Nombre maximum de fichiers à traiter (pour éviter les timeouts)
MAX_FILES = 10
# Nombre de workers parallèles pour Groq
MAX_WORKERS = 5
# Timeout pour chaque appel Groq (en secondes)
GROQ_TIMEOUT = 30

class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch: str = "auto-refactor"


def clone_repo(repo_url, branch):
    """Clone le dépôt de manière superficielle (un seul commit, une seule branche)."""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_url = repo_url.replace(
        "https://",
        f"https://{GITHUB_TOKEN}@"
    )

    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)

    subprocess.run([
        "git", "clone", "-b", branch,
        "--depth", "1",
        "--single-branch",
        clone_url
    ], check=True)
    return repo_name


def get_changed_files(repo_path, base_ref):
    """Retourne la liste des fichiers .kt modifiés entre base_ref et HEAD."""
    os.chdir(repo_path)
    cmd = f"git diff --name-only origin/{base_ref}...HEAD"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    files = [
        f for f in output.splitlines()
        if f.endswith(".kt") and os.path.exists(f)
    ]
    os.chdir("..")
    return files


def refactor_file(repo_path, filepath):
    """Appelle Groq pour refactoriser un fichier, avec timeout et gestion d'erreur."""
    full_path = os.path.join(repo_path, filepath)

    with open(full_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Si le fichier ne contient aucun des patterns de log, on ignore
    if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
        return f"{filepath} - No logs found"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    # Correction : indentation à 4 espaces (au lieu de 5)
    prompt = (
        "You are a Kotlin Refactoring Expert.\n"
        "Your mission: Clean up and deduplicate logging in this Android code.\n\n"
        "1. CONVERSION RULES:\n"
        "   - Log.d/i/w/e OR Logr.d/i/w/e -> AppLogger.d/i/w/e(tag, msg)\n"
        "   - AppSTLogger.appendLogST(STLevelLog.DEBUG, tag, msg) -> AppLogger.d(tag, msg)\n"
        "   - AppSTLogger.appendLogST(STLevelLog.INFO, tag, msg)  -> AppLogger.i(tag, msg)\n"
        "   - AppSTLogger.appendLogST(STLevelLog.WARN, tag, msg)  -> AppLogger.w(tag, msg)\n"
        "   - AppSTLogger.appendLogST(STLevelLog.ERROR, tag, msg) -> AppLogger.e(tag, msg)\n\n"
        "2. DEDUPLICATION RULE (CRITICAL):\n"
        "   - Merge consecutive lines of AppLogger with EXACT SAME tag and message into ONE.\n"
        "   - Example: Multiple AppLogger.e(MODULE, 'text') calls become just one.\n\n"
        "3. IMPORTS:\n"
        "   - ADD: 'import com.honeywell.domain.managers.loggerApp.AppLogger'.\n"
        "   - REMOVE: android.util.Log, Logr, STLevelLog, and AppSTLogger imports.\n\n"
        "Return ONLY raw source code. NO markdown markers, NO explanations."
    )

    payload = {
        "model": "mixtral-8x7b-32768",  # ou "llama-3.3-70b-versatile" si vous préférez
        "messages": [
            {"role": "system", "content": "You are a Kotlin expert. Output only raw source code."},
            {"role": "user", "content": f"{prompt}\n\nCODE:\n{code}"}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        response.raise_for_status()
        new_code = response.json()["choices"][0]["message"]["content"].strip()
        new_code = re.sub(r"^```kotlin\s*|^```\s*", "", new_code)
        new_code = re.sub(r"\s*```$", "", new_code)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        return f"{filepath} refactored"
    except requests.exceptions.Timeout:
        logger.error(f"Timeout sur Groq pour {filepath}")
        return f"{filepath} - Groq timeout"
    except Exception as e:
        logger.error(f"Erreur sur {filepath}: {e}")
        return f"{filepath} - Error: {str(e)}"


def commit_and_push(repo_path):
    """Commit et push les modifications."""
    os.chdir(repo_path)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Auto refactor logs"], check=True)
    subprocess.run(["git", "push"], check=True)
    os.chdir("..")


@app.get("/")
async def root():
    return {
        "message": "Refactor Agent API Is Active",
        "usage": "POST /refactor avec les headers x-api-key et Content-Type",
        "docs": "/docs"
    }


@app.post("/refactor")
def run_refactor(
    request: RefactorRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="Missing environment variables")

    logger.info(f"Début du traitement pour {request.repo_url}, branche {request.branch}")

    # 1. Clone du dépôt
    repo_path = clone_repo(request.repo_url, request.branch)

    # 2. Récupération des fichiers modifiés
    files = get_changed_files(repo_path, request.base_ref)
    logger.info(f"Fichiers modifiés trouvés : {len(files)}")

    # 3. Limitation du nombre de fichiers (optionnel)
    if len(files) > MAX_FILES:
        logger.warning(f"Trop de fichiers ({len(files)}), traitement limité aux {MAX_FILES} premiers")
        files = files[:MAX_FILES]

    # 4. Traitement parallèle avec ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(refactor_file, repo_path, f): f for f in files}
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            logger.info(result)

    # 5. Commit et push
    commit_and_push(repo_path)

    logger.info("Traitement terminé avec succès")
    return {"processed_files": results}
