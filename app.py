"""
VERSION v6 FINALE - CORRIGE TOUS LES PROBLEMES
- Fix git identity (commit marchent)
- Fix rate limits (0 erreur 429)
- MAX_WORKERS = 1 (séquentiel)
- Délai 2s entre fichiers
- Retry automatique
"""

import os
import re
import shutil
import sys
import time
import requests
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

MAX_FILES = 10
MAX_WORKERS = 1  # ⭐ 1 fichier à la fois (évite rate limits)
GROQ_TIMEOUT = 30
CLONE_DEPTH = 500
MAX_FILE_SIZE = 50000
REQUEST_DELAY = 2  # ⭐ 2s entre chaque fichier
MAX_RETRIES = 3

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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
    sys.stdout.flush()
    subprocess.run([
        "git", "clone",
        "--depth", str(CLONE_DEPTH),
        "--no-single-branch",
        "-b", branch,
        clone_url, repo_name
    ], check=True)

    os.chdir(repo_name)

    logger.info("Configuration remote fetch → toutes branches")
    sys.stdout.flush()
    subprocess.run(["git", "remote", "set-branches", "origin", "*"], check=True, capture_output=True)

    logger.info("Fetch large shallow initial")
    sys.stdout.flush()
    subprocess.run(["git", "fetch", "origin", "--depth=" + str(CLONE_DEPTH)], capture_output=True)

    logger.info(f"Fetch forcé refspec → refs/remotes/origin/{base_ref}")
    sys.stdout.flush()
    fetch_cmd = [
        "git", "fetch", "origin",
        f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
        f"--depth={CLONE_DEPTH}"
    ]
    fetch_res = subprocess.run(fetch_cmd, capture_output=True, text=True)
    if fetch_res.returncode != 0:
        logger.warning(f"Fetch refspec échoué : {fetch_res.stderr.strip() or 'code ' + str(fetch_res.returncode)}")
        sys.stdout.flush()
        subprocess.run(["git", "update-ref", f"refs/remotes/origin/{base_ref}", "FETCH_HEAD"], check=False)

    logger.info("Deepen désactivé (depth=500 suffit)")
    sys.stdout.flush()

    logger.info("DEBUG CLONE - git branch -r :")
    sys.stdout.flush()
    logger.info(subprocess.getoutput("git branch -r") or "<vide>")

    logger.info(f"DEBUG CLONE - show-ref | grep {base_ref}")
    sys.stdout.flush()
    logger.info(subprocess.getoutput(f"git show-ref | grep {base_ref}") or "<aucune ref>")

    os.chdir("..")
    return repo_name


def get_changed_files(repo_path: str, base_ref: str):
    os.chdir(repo_path)
    try:
        base_local = f"base-{base_ref.replace('/', '-')}"

        logger.info(f"Création branche locale {base_local}")
        sys.stdout.flush()
        res = subprocess.run(
            ["git", "branch", base_local, f"refs/remotes/origin/{base_ref}"],
            capture_output=True, text=True
        )
        if res.returncode != 0:
            logger.warning("Branche depuis origin échoue → fallback FETCH_HEAD")
            sys.stdout.flush()
            subprocess.run(["git", "branch", base_local, "FETCH_HEAD"], capture_output=True)

        logger.info("=== DEBUG DIFF ===")
        sys.stdout.flush()
        logger.info("Branches (git branch -a) :")
        logger.info(subprocess.getoutput("git branch -a") or "<vide>")

        logger.info(f"Log {base_local} :")
        sys.stdout.flush()
        logger.info(subprocess.getoutput(f"git log -n 3 --oneline {base_local} || echo 'pas accessible'"))

        logger.info("Log HEAD :")
        sys.stdout.flush()
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
            sys.stdout.flush()
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=60)
                candidates = [line.strip() for line in out.decode().splitlines() if line.strip()]
                logger.info(f"{name} → {len(candidates)} fichiers trouvés")
                sys.stdout.flush()
                if candidates:
                    logger.info(f"exemples : {candidates[:6]}")
                    files = candidates
                    break
            except subprocess.TimeoutExpired:
                logger.error(f"{name} → timeout")
                sys.stdout.flush()
            except subprocess.CalledProcessError as e:
                err = e.output.decode(errors='ignore').strip()
                logger.warning(f"{name} → {err}")
                sys.stdout.flush()

        kt_files = [f for f in files if f.endswith(".kt") and os.path.exists(f)]
        logger.info(f".kt trouvés : {len(kt_files)} → {kt_files[:5] or 'aucun'}")
        sys.stdout.flush()

        return kt_files

    except Exception as e:
        logger.exception("Erreur calcul changements")
        sys.stdout.flush()
        return []
    finally:
        os.chdir("..")


def refactor_file(repo_path: str, filepath: str) -> str:
    """Refactorise un fichier avec retry automatique sur rate limit."""
    full_path = os.path.join(repo_path, filepath)
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()

        if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
            return f"{filepath} - pas de logs"

        file_size = len(code)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"⚠️  {filepath} trop gros ({file_size:,} chars > {MAX_FILE_SIZE:,}), ignoré")
            sys.stdout.flush()
            return f"{filepath} → trop gros ({file_size:,} chars)"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}", 
            "Content-Type": "application/json"
        }

        prompt = (
    "You are a Kotlin Refactoring Expert.\n"
    "Your mission: Clean up and deduplicate logging in this Android code.\n\n"

    "🚨 CRITICAL CONSTRAINT:\n"
    " - DO NOT modify business logic.\n"
    " - DO NOT change control flow (if, when, for, while, try/catch).\n"
    " - DO NOT modify variable names.\n"
    " - DO NOT modify function signatures.\n"
    " - DO NOT add or remove methods.\n"
    " - DO NOT change return values.\n"
    " - DO NOT optimize code.\n"
    " - ONLY refactor logging statements.\n\n"

    "1. CONVERSION RULES:\n"
    " - Log.d/i/w/e OR Logr.d/i/w/e -> AppLogger.d/i/w/e(tag, msg)\n"
    " - AppSTLogger.appendLogST(STLevelLog.DEBUG, tag, msg) -> AppLogger.d(tag, msg)\n"
    " - AppSTLogger.appendLogST(STLevelLog.INFO, tag, msg) -> AppLogger.i(tag, msg)\n"
    " - AppSTLogger.appendLogST(STLevelLog.WARN, tag, msg) -> AppLogger.w(tag, msg)\n"
    " - AppSTLogger.appendLogST(STLevelLog.ERROR, tag, msg) -> AppLogger.e(tag, msg)\n\n"

    "2. DEDUPLICATION RULE (CRITICAL):\n"
    " - Merge consecutive lines of AppLogger with EXACT SAME tag and message into ONE.\n"
    " - Only merge if they are strictly identical and consecutive.\n\n"

    "3. IMPORTS:\n"
    " - ADD: 'import com.honeywell.domain.managers.loggerApp.AppLogger'.\n"
    " - REMOVE: android.util.Log, Logr, STLevelLog, and AppSTLogger imports.\n"
    " - Do not modify any other imports.\n\n"

    "Return ONLY raw source code.\n"
    "NO markdown.\n"
    "NO explanations.\n"
    "NO comments added.\n"
    "Preserve formatting and indentation."
)

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a Kotlin expert. Output only raw source code."},
                {"role": "user", "content": f"{prompt}\n\nCODE:\n{code}"}
            ],
            "temperature": 0
        }

        logger.info(f"🤖 Refactoring {filepath} ({file_size:,} chars) avec {GROQ_MODEL}...")
        sys.stdout.flush()
        
        # ⭐ RETRY LOGIC pour gérer les 429
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
                
                # Si rate limit (429), attendre et réessayer
                if resp.status_code == 429:
                    if attempt < MAX_RETRIES:
                        wait_time = 10
                        logger.warning(f"⏳ Rate limit (429) - tentative {attempt}/{MAX_RETRIES}, attente {wait_time}s...")
                        sys.stdout.flush()
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ Rate limit persistant après {MAX_RETRIES} tentatives")
                        sys.stdout.flush()
                        return f"{filepath} → rate limit (max retries)"
                
                # Autres erreurs
                if resp.status_code != 200:
                    logger.error(f"❌ Groq API Error {resp.status_code} pour {filepath}")
                    try:
                        error_data = resp.json()
                        error_msg = error_data.get('error', {}).get('message', resp.text[:200])
                        logger.error(f"   Message: {error_msg}")
                    except:
                        logger.error(f"   Response: {resp.text[:500]}")
                    sys.stdout.flush()
                    return f"{filepath} → Groq error {resp.status_code}"

                # Succès !
                new_code = resp.json()["choices"][0]["message"]["content"].strip()
                
                new_code = re.sub(r'^```kotlin\s*', '', new_code, flags=re.M)
                new_code = re.sub(r'^\s*```$', '', new_code, flags=re.M)

                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(new_code)

                logger.info(f"✅ {filepath} refactorisé")
                sys.stdout.flush()
                
                # ⭐ DÉLAI entre requêtes (évite rate limits)
                time.sleep(REQUEST_DELAY)
                
                return f"{filepath} → refactored"
                
            except requests.exceptions.Timeout:
                logger.error(f"⏱️  Timeout Groq pour {filepath}")
                sys.stdout.flush()
                return f"{filepath} → timeout Groq"

    except Exception as e:
        logger.error(f"❌ Erreur refactor {filepath}: {e}")
        sys.stdout.flush()
        return f"{filepath} → erreur"


def commit_and_push(repo_path: str):
    """Commit et push avec configuration git identity."""
    os.chdir(repo_path)
    try:
        # ⭐ CONFIGURATION GIT IDENTITY (FIX PRINCIPAL)
        logger.info("🔧 Configuration Git identity: Refactor Agent Bot <bot@refactor-agent.local>")
        sys.stdout.flush()
        subprocess.run(["git", "config", "user.name", "Refactor Agent Bot"], check=True)
        subprocess.run(["git", "config", "user.email", "bot@refactor-agent.local"], check=True)
        
        # Debug: voir le statut git avant commit
        logger.info("🔍 Git status avant commit:")
        sys.stdout.flush()
        status_output = subprocess.getoutput("git status --short")
        logger.info(status_output if status_output else "  <aucun changement détecté>")
        sys.stdout.flush()
        
        subprocess.run(["git", "add", "."], check=True)
        
        # Debug: voir le statut après git add
        logger.info("🔍 Git status après git add:")
        sys.stdout.flush()
        staged_output = subprocess.getoutput("git diff --cached --name-only")
        logger.info(staged_output if staged_output else "  <rien à commiter>")
        sys.stdout.flush()
        
        commit_res = subprocess.run(
            ["git", "commit", "-m", "🤖 Auto refactor logs"],
            capture_output=True, text=True
        )
        if commit_res.returncode == 0:
            logger.info("✅ Commit créé, push en cours...")
            sys.stdout.flush()
            subprocess.run(["git", "push"], check=True)
            logger.info("✅ Push réussi!")
            sys.stdout.flush()
        else:
            logger.warning("⚠️  Commit vide ou déjà fait → skip push")
            logger.warning(f"   Git commit output: {commit_res.stdout.strip() or commit_res.stderr.strip()}")
            sys.stdout.flush()
    except Exception as e:
        logger.warning(f"⚠️  Problème commit/push : {e}")
        sys.stdout.flush()
    finally:
        os.chdir("..")


@app.get("/")
async def root():
    return {
        "message": "Refactor Agent API Active",
        "version": "6.0-final-corrected",
        "config": {
            "groq_model": GROQ_MODEL,
            "max_file_size": f"{MAX_FILE_SIZE:,} chars",
            "max_workers": MAX_WORKERS,
            "request_delay": f"{REQUEST_DELAY}s",
            "max_retries": MAX_RETRIES
        }
    }


@app.get("/health")
async def health():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not API_SECRET:
        missing.append("API_SECRET")
    
    if missing:
        return {"status": "unhealthy", "missing_env": missing}
    
    return {
        "status": "healthy",
        "groq_model": GROQ_MODEL,
        "optimized_for": "free_tier"
    }


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
    sys.stdout.flush()

    try:
        repo_path = clone_repo(request.repo_url, request.branch, request.base_ref)
        files = get_changed_files(repo_path, request.base_ref)

        if not files:
            return {"status": "ok", "message": "Aucun .kt modifié détecté"}

        files = files[:MAX_FILES] if len(files) > MAX_FILES else files
        
        logger.info(f"📝 Traitement séquentiel de {len(files)} fichiers (plan gratuit optimisé)")
        sys.stdout.flush()

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(refactor_file, repo_path, f): f for f in files}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(result)
                sys.stdout.flush()

        if any("refactored" in r for r in results):
            commit_and_push(repo_path)

        return {"status": "success", "processed_files": results}

    except subprocess.CalledProcessError as e:
        logger.exception("Erreur git")
        sys.stdout.flush()
        raise HTTPException(500, f"Git error: {e.stderr or str(e)}")
    except Exception as e:
        logger.exception("Erreur générale")
        sys.stdout.flush()
        raise HTTPException(500, str(e))


