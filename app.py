import os
import re
import shutil
import sys
import requests
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel

# Logging avec flush immédiat pour Render
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force unbuffered output pour voir les logs en temps réel
sys.stdout.reconfigure(line_buffering=True)

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

MAX_FILES = 10
MAX_WORKERS = 5
GROQ_TIMEOUT = 30
CLONE_DEPTH = 500

class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch: str = "auto-refactor"


def clone_repo(repo_url: str, branch: str, base_ref: str) -> str:
    """Clone le repo avec la branche spécifiée."""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")

    if os.path.exists(repo_name):
        logger.info(f"Nettoyage ancien clone {repo_name}")
        shutil.rmtree(repo_name)

    logger.info(f"🔄 Clonage {branch!r} depth={CLONE_DEPTH}")
    sys.stdout.flush()
    
    try:
        subprocess.run([
            "git", "clone",
            "--depth", str(CLONE_DEPTH),
            "--no-single-branch",
            "-b", branch,
            clone_url, repo_name
        ], check=True, capture_output=True, text=True, timeout=120)
        logger.info(f"✅ Clone réussi")
    except subprocess.TimeoutExpired:
        raise Exception(f"Timeout lors du clone (>120s)")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Erreur clone: {e.stderr}")

    os.chdir(repo_name)

    logger.info("📡 Configuration remote fetch")
    sys.stdout.flush()
    subprocess.run(["git", "remote", "set-branches", "origin", "*"], check=True, capture_output=True)

    logger.info(f"📡 Fetch {base_ref}")
    sys.stdout.flush()
    
    # Fetch plus agressif pour s'assurer d'avoir base_ref
    fetch_cmd = [
        "git", "fetch", "origin",
        f"+refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
        f"--depth={CLONE_DEPTH}"
    ]
    fetch_res = subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=60)
    
    if fetch_res.returncode != 0:
        logger.warning(f"⚠️  Fetch refspec a échoué, tentative de récupération...")
        sys.stdout.flush()
        # Fallback: fetch all refs
        subprocess.run(["git", "fetch", "origin", "--depth=" + str(CLONE_DEPTH)], 
                      capture_output=True, timeout=60)

    # Deepen pour avoir plus d'historique
    logger.info("📚 Deepen historique")
    sys.stdout.flush()
    subprocess.run(["git", "fetch", "--deepen=400"], check=False, capture_output=True, timeout=60)

    # Debug - afficher les refs disponibles
    logger.info("🔍 DEBUG: Branches distantes disponibles")
    sys.stdout.flush()
    branches_output = subprocess.getoutput("git branch -r")
    logger.info(branches_output if branches_output else "  <aucune branche distante>")
    
    logger.info(f"🔍 DEBUG: Recherche de {base_ref}")
    sys.stdout.flush()
    ref_output = subprocess.getoutput(f"git show-ref | grep {base_ref}")
    logger.info(ref_output if ref_output else f"  <{base_ref} non trouvé>")
    sys.stdout.flush()

    os.chdir("..")
    return repo_name


def get_changed_files(repo_path: str, base_ref: str):
    """Détecte les fichiers .kt modifiés entre base_ref et HEAD."""
    os.chdir(repo_path)
    
    try:
        logger.info(f"🔍 Recherche fichiers modifiés vs {base_ref}")
        sys.stdout.flush()
        
        # Normaliser le nom de la branche locale
        base_local = f"base-{base_ref.replace('/', '-')}"
        
        # Créer branche locale pointant sur origin/base_ref
        logger.info(f"📌 Création branche locale {base_local}")
        sys.stdout.flush()
        
        res = subprocess.run(
            ["git", "branch", base_local, f"refs/remotes/origin/{base_ref}"],
            capture_output=True, text=True
        )
        
        if res.returncode != 0:
            logger.warning(f"⚠️  Impossible de créer {base_local} depuis remote")
            logger.warning(f"   Erreur: {res.stderr.strip()}")
            sys.stdout.flush()
            
            # Fallback: essayer avec FETCH_HEAD
            logger.info("   Fallback → FETCH_HEAD")
            subprocess.run(["git", "branch", base_local, "FETCH_HEAD"], 
                         capture_output=True, check=False)

        # Debug: vérifier les branches
        logger.info("📋 Branches locales:")
        sys.stdout.flush()
        branches = subprocess.getoutput("git branch -a")
        logger.info(branches if branches else "  <aucune>")
        
        # Vérifier que les deux refs existent
        logger.info(f"📋 Commits récents {base_local}:")
        sys.stdout.flush()
        base_log = subprocess.getoutput(f"git log -n 3 --oneline {base_local} 2>&1")
        logger.info(base_log)
        
        logger.info("📋 Commits récents HEAD:")
        sys.stdout.flush()
        head_log = subprocess.getoutput("git log -n 3 --oneline HEAD")
        logger.info(head_log)

        # Tentatives de diff avec différentes stratégies
        files = []
        attempts = [
            ("three-dot local", f"git diff --name-only {base_local}...HEAD"),
            ("two-dot local",   f"git diff --name-only {base_local}..HEAD"),
            ("direct local",    f"git diff --name-only {base_local} HEAD"),
            ("three-dot origin", f"git diff --name-only origin/{base_ref}...HEAD"),
            ("two-dot origin",   f"git diff --name-only origin/{base_ref}..HEAD"),
        ]

        for name, cmd in attempts:
            logger.info(f"🔍 Tentative: {name}")
            logger.info(f"   Commande: {cmd}")
            sys.stdout.flush()
            
            try:
                out = subprocess.check_output(
                    cmd, 
                    shell=True, 
                    stderr=subprocess.STDOUT, 
                    timeout=30,
                    text=True
                )
                candidates = [line.strip() for line in out.splitlines() if line.strip()]
                logger.info(f"   ✅ {len(candidates)} fichiers trouvés")
                
                if candidates:
                    logger.info(f"   Exemples: {candidates[:6]}")
                    files = candidates
                    break
                else:
                    logger.info(f"   ⚠️  0 fichiers (pas d'erreur mais liste vide)")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"   ❌ Timeout (>30s)")
            except subprocess.CalledProcessError as e:
                err = e.output.strip() if e.output else str(e)
                logger.warning(f"   ❌ Erreur: {err}")
            
            sys.stdout.flush()

        # Filtrer les .kt qui existent
        kt_files = [f for f in files if f.endswith(".kt") and os.path.exists(f)]
        logger.info(f"📄 Fichiers .kt trouvés: {len(kt_files)}")
        
        if kt_files:
            logger.info(f"   Liste: {kt_files[:10]}")
        else:
            logger.warning("   ⚠️  Aucun fichier .kt modifié détecté")
        
        sys.stdout.flush()
        return kt_files

    except Exception as e:
        logger.exception("❌ Erreur lors du calcul des changements")
        sys.stdout.flush()
        return []
    finally:
        os.chdir("..")


def refactor_file(repo_path: str, filepath: str) -> str:
    """Refactorise un fichier Kotlin via Groq."""
    full_path = os.path.join(repo_path, filepath)
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Vérifier si le fichier contient des logs à refactoriser
        if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
            return f"{filepath} - pas de logs à refactoriser"

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

        logger.info(f"🤖 Refactoring {filepath} via Groq...")
        sys.stdout.flush()
        
        resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        resp.raise_for_status()

        new_code = resp.json()["choices"][0]["message"]["content"].strip()
        
        # Nettoyer les marqueurs markdown si présents
        new_code = re.sub(r'^```kotlin\s*', '', new_code, flags=re.M)
        new_code = re.sub(r'^\s*```$', '', new_code, flags=re.M)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        logger.info(f"✅ {filepath} refactorisé")
        sys.stdout.flush()
        return f"{filepath} → refactored"

    except requests.exceptions.Timeout:
        logger.error(f"⏱️  Timeout Groq pour {filepath}")
        sys.stdout.flush()
        return f"{filepath} → timeout Groq"
    except Exception as e:
        logger.error(f"❌ Erreur refactor {filepath}: {e}")
        sys.stdout.flush()
        return f"{filepath} → erreur: {str(e)}"


def commit_and_push(repo_path: str):
    """Commit et push les changements."""
    os.chdir(repo_path)
    
    try:
        logger.info("💾 Git add...")
        sys.stdout.flush()
        subprocess.run(["git", "add", "."], check=True, timeout=30)
        
        logger.info("📝 Git commit...")
        sys.stdout.flush()
        commit_res = subprocess.run(
            ["git", "commit", "-m", "🤖 Auto refactor logs", "--allow-empty"],
            capture_output=True, text=True, timeout=30
        )
        
        if commit_res.returncode == 0:
            logger.info("🚀 Git push...")
            sys.stdout.flush()
            push_res = subprocess.run(
                ["git", "push"], 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            logger.info("✅ Commit & push réussis")
            sys.stdout.flush()
        else:
            logger.info("ℹ️  Rien à commiter")
            sys.stdout.flush()
            
    except subprocess.TimeoutExpired as e:
        logger.error(f"⏱️  Timeout git: {e}")
        sys.stdout.flush()
    except Exception as e:
        logger.warning(f"⚠️  Problème commit/push: {e}")
        sys.stdout.flush()
    finally:
        os.chdir("..")


def process_refactor(request: RefactorRequest):
    """Traitement principal du refactoring."""
    logger.info("="*60)
    logger.info(f"🚀 DÉMARRAGE REFACTOR")
    logger.info(f"   Repo: {request.repo_url}")
    logger.info(f"   Base: {request.base_ref}")
    logger.info(f"   Branch: {request.branch}")
    logger.info("="*60)
    sys.stdout.flush()

    try:
        # 1. Clone
        repo_path = clone_repo(request.repo_url, request.branch, request.base_ref)
        
        # 2. Détection fichiers
        files = get_changed_files(repo_path, request.base_ref)

        if not files:
            logger.warning("⚠️  Aucun fichier .kt modifié détecté")
            sys.stdout.flush()
            return {"status": "ok", "message": "Aucun .kt modifié détecté", "processed_files": []}

        # Limiter au max
        if len(files) > MAX_FILES:
            logger.info(f"⚠️  Limitation à {MAX_FILES} fichiers (total: {len(files)})")
            files = files[:MAX_FILES]
        
        logger.info(f"📝 Fichiers à traiter: {len(files)}")
        sys.stdout.flush()

        # 3. Refactoring parallèle
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(refactor_file, repo_path, f): f for f in files}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                logger.info(f"  → {result}")
                sys.stdout.flush()

        # 4. Commit & Push si au moins un fichier refactorisé
        if any("refactored" in r for r in results):
            commit_and_push(repo_path)
        else:
            logger.info("ℹ️  Aucun fichier n'a été modifié, pas de commit")
            sys.stdout.flush()

        logger.info("="*60)
        logger.info("✅ REFACTOR TERMINÉ")
        logger.info("="*60)
        sys.stdout.flush()
        
        return {"status": "success", "processed_files": results}

    except subprocess.CalledProcessError as e:
        error_msg = f"Git error: {e.stderr or str(e)}"
        logger.exception(f"❌ {error_msg}")
        sys.stdout.flush()
        raise HTTPException(500, error_msg)
    except Exception as e:
        logger.exception(f"❌ Erreur générale: {e}")
        sys.stdout.flush()
        raise HTTPException(500, str(e))


@app.get("/")
async def root():
    return {
        "message": "Refactor Agent API Active",
        "version": "2.0",
        "endpoints": ["/refactor"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not API_SECRET:
        missing.append("API_SECRET")
    
    if missing:
        return {
            "status": "unhealthy",
            "missing_env": missing
        }
    
    return {"status": "healthy"}


@app.post("/refactor")
def run_refactor(
    request: RefactorRequest,
    x_api_key: str = Header(None)
):
    """Endpoint principal de refactoring."""
    # Vérification auth
    if x_api_key != API_SECRET:
        logger.warning(f"⚠️  Tentative d'accès non autorisée")
        sys.stdout.flush()
        raise HTTPException(403, "Unauthorized")

    # Vérification variables d'environnement
    if not GROQ_API_KEY or not GITHUB_TOKEN:
        logger.error("❌ Variables d'environnement manquantes")
        sys.stdout.flush()
        raise HTTPException(500, "Variables d'environnement manquantes (GROQ_API_KEY ou GITHUB_TOKEN)")

    # Traitement
    return process_refactor(request)
