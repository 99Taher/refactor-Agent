import os
import re
import sys
import time
import shutil
import logging
import requests
import subprocess
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# ================= CONFIG =================

MAX_FILES        = 10
MAX_WORKERS      = 1
GROQ_TIMEOUT     = 60
CLONE_DEPTH      = 200
MAX_FILE_SIZE    = 8000   # seuil pour activer le chunking
REQUEST_DELAY    = 3
MAX_RETRIES      = 5
CHUNK_SIZE       = 8000   # llama-3.1-8b-instant = ~8192 tokens output max
MAX_TOKENS_OUT   = 8000
RATE_LIMIT_BASE_WAIT  = 15  # secondes, doublé à chaque tentative
MAX_INTER_CHUNK_DELAY = 15  # ⭐ plafond des pauses entre chunks (secondes)

# Un chunk de moins de N chars est considéré trivial (ex: `}`) — on saute les checks
MIN_CHUNK_SIZE_FOR_CHECKS = 100

# Nombre max de lignes non-log perdues par chunk (absolue)
NON_LOG_LINE_LOSS_MAX = 2

GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET   = os.getenv("API_SECRET")

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
app = FastAPI()


class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch:   str = "auto-refactor"


# ================= UTIL =================

def run_git(cmd: List[str], cwd=None, check=True):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Exception(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def split_code(code: str) -> List[str]:
    """
    Découpe le code en chunks de CHUNK_SIZE caractères max.
    Coupe sur \\nfun  si possible, sinon sur une ligne vide.
    """
    chunks = []
    start = 0
    while start < len(code):
        end = min(start + CHUNK_SIZE, len(code))
        if end < len(code):
            split = code.rfind("\nfun ", start, end)
            if split != -1 and split > start:
                end = split
            else:
                split = code.rfind("\n\n", start, end)
                if split != -1 and split > start:
                    end = split
        chunks.append(code[start:end])
        start = end
    return chunks


# ================= GIT =================

def clone_repo(repo_url: str, branch: str) -> Path:
    if not repo_url.startswith("https://"):
        raise HTTPException(400, "repo_url must be https")

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = Path(repo_name)

    if repo_path.exists():
        shutil.rmtree(repo_path)

    clone_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")

    logger.info("Cloning repository...")
    run_git([
        "git", "clone",
        "--depth", str(CLONE_DEPTH),
        "--no-single-branch",
        "-b", branch,
        clone_url,
        repo_name
    ])

    return repo_path


def get_changed_files(repo_path: Path, base_ref: str) -> List[str]:
    logger.info(f"Detecting changed files (base={base_ref})...")

    try:
        run_git([
            "git", "fetch", "origin",
            f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
            f"--depth={CLONE_DEPTH}"
        ], cwd=repo_path)
        logger.info(f"✅ Branch origin/{base_ref} fetched")
    except Exception as e:
        logger.warning(f"⚠️ Fetch direct échoué: {e}")
        try:
            run_git(["git", "fetch", "--all", f"--depth={CLONE_DEPTH}"], cwd=repo_path)
        except Exception:
            pass

    branches_raw = run_git(["git", "branch", "-r"], cwd=repo_path, check=False)
    branches = [b.strip() for b in branches_raw.splitlines()]
    origin_branch = f"origin/{base_ref}"

    logger.info(f"Branches disponibles: {branches}")

    if origin_branch in branches:
        base_ref_for_diff = origin_branch
        logger.info(f"✅ Diff avec {origin_branch}")
    else:
        logger.warning(f"⚠️ Branch {origin_branch} non trouvée")
        if "origin/HEAD" in branches:
            base_ref_for_diff = "origin/HEAD"
            logger.info("Fallback: diff avec origin/HEAD")
        else:
            base_ref_for_diff = "HEAD^"
            logger.info("Fallback: diff avec HEAD^")

    try:
        diff = run_git(
            ["git", "diff", "--name-only", f"{base_ref_for_diff}...HEAD", "--"],
            cwd=repo_path
        )
        if not diff:
            logger.info("Aucun fichier modifié trouvé avec three-dot, essai two-dot...")
            diff = run_git(
                ["git", "diff", "--name-only", f"{base_ref_for_diff}..HEAD", "--"],
                cwd=repo_path
            )

        files = [f for f in diff.splitlines() if f.strip().endswith(".kt")]
        logger.info(f"📝 {len(files)} fichiers .kt trouvés: {files[:5]}")
        return files[:MAX_FILES]

    except Exception as e:
        logger.error(f"Erreur git diff: {e}")
        return []


# ================= GROQ =================

LOG_PATTERNS = re.compile(
    r"(Log\.[diwev]\(|Logr\.[diwev]\(|AppSTLogger\.appendLogST\(|AppLogger\.[diwev]\()"
)

IMPORT_PATTERNS = re.compile(
    r"^import\s+(android\.util\.Log|com\.honeywell.*[Ll]og|com\.st.*[Ll]og|.*Logr|.*STLevelLog|.*AppLogger|.*AppSTLogger)"
)

TRUNCATION_PATTERNS = re.compile(
    r"(^\s*//\s*\.\.\.\s*$"
    r"|^\s*/\*\s*\.\.\.\s*\*/\s*$"
    r"|^\s*\.\.\.\s*$"
    r"|^\s*//\s*rest of (the )?code"
    r"|^\s*//\s*\[truncated\]"
    r")",
    re.IGNORECASE | re.MULTILINE
)


def is_log_related(line: str) -> bool:
    stripped = line.strip()
    return bool(LOG_PATTERNS.search(stripped)) or bool(IMPORT_PATTERNS.match(stripped))


def detect_truncation(code: str) -> bool:
    return bool(TRUNCATION_PATTERNS.search(code))


def check_chunk_non_log_loss(original: str, refactored: str, chunk_index: int) -> tuple[bool, str]:
    """
    Vérifie uniquement les lignes NON-LOG.
    Les logs peuvent diminuer par déduplication — c'est normal.
    """
    orig_non_log = [l.strip() for l in original.splitlines()   if l.strip() and not is_log_related(l)]
    new_non_log  = [l.strip() for l in refactored.splitlines() if l.strip() and not is_log_related(l)]

    orig_log = sum(1 for l in original.splitlines()   if l.strip() and is_log_related(l))
    new_log  = sum(1 for l in refactored.splitlines() if l.strip() and is_log_related(l))
    orig_total = len([l for l in original.splitlines()   if l.strip()])
    new_total  = len([l for l in refactored.splitlines() if l.strip()])

    logger.info(
        f"  📊 Chunk {chunk_index}: total {orig_total}→{new_total} | "
        f"non-log {len(orig_non_log)}→{len(new_non_log)} | "
        f"log {orig_log}→{new_log} (déduplication: {orig_log - new_log} supprimés)"
    )

    orig_set = set(orig_non_log)
    new_set  = set(new_non_log)
    removed  = orig_set - new_set
    added    = new_set  - orig_set

    if len(removed) <= NON_LOG_LINE_LOSS_MAX and len(added) <= NON_LOG_LINE_LOSS_MAX:
        if removed or added:
            for line in list(removed)[:3]:
                logger.warning(f"  ⚠️  Non-log supprimée: '{line[:80]}'")
            for line in list(added)[:3]:
                logger.warning(f"  ⚠️  Non-log ajoutée:   '{line[:80]}'")
        logger.info(f"  ✅ Chunk {chunk_index}: code métier OK")
        return True, "ok"

    msg = (
        f"Chunk {chunk_index}: {len(removed)} lignes non-log supprimées, "
        f"{len(added)} ajoutées (max={NON_LOG_LINE_LOSS_MAX})"
    )
    logger.error(f"  ❌ {msg}")
    for line in list(removed)[:3]:
        logger.error(f"    - '{line[:80]}'")
    for line in list(added)[:3]:
        logger.error(f"    + '{line[:80]}'")
    return False, msg


def build_chunk_prompt(is_retry: bool = False) -> str:
    retry_prefix = (
        "⚠️ YOUR PREVIOUS RESPONSE WAS REJECTED: you removed non-log lines.\n"
        "THIS TIME: copy every single non-log line into your output unchanged. "
        "Only modify lines that are Log.x / Logr.x / AppSTLogger calls.\n\n"
    ) if is_retry else ""

    return (
        retry_prefix +
        "You are a Kotlin log-refactoring tool.\n"
        "Input: a FRAGMENT of a Kotlin file.\n"
        "Output: the SAME fragment with ONLY logging calls refactored.\n\n"

        "ABSOLUTE RULES:\n"
        " - Output EVERY line of the input. Zero omissions.\n"
        " - NEVER write '// ...', '...', or any placeholder.\n"
        " - DO NOT modify any business logic, control flow, variable names,\n"
        "   function signatures, return values, or non-log code.\n"
        " - ONLY change: Log.x / Logr.x / AppSTLogger.appendLogST calls and their imports.\n\n"

        "CONVERSIONS:\n"
        " - Log.d/i/w/e(tag, msg)                         -> AppLogger.d/i/w/e(tag, msg)\n"
        " - Logr.d/i/w/e(tag, msg)                        -> AppLogger.d/i/w/e(tag, msg)\n"
        " - AppSTLogger.appendLogST(STLevelLog.DEBUG, ..)  -> AppLogger.d(tag, msg)\n"
        " - AppSTLogger.appendLogST(STLevelLog.INFO, ..)   -> AppLogger.i(tag, msg)\n"
        " - AppSTLogger.appendLogST(STLevelLog.WARN, ..)   -> AppLogger.w(tag, msg)\n"
        " - AppSTLogger.appendLogST(STLevelLog.ERROR, ..)  -> AppLogger.e(tag, msg)\n\n"

        "DEDUPLICATION: merge consecutive AppLogger calls with identical tag+message into one.\n\n"

        "IMPORTS (first fragment only):\n"
        " - ADD:    import com.honeywell.domain.managers.loggerApp.AppLogger\n"
        " - REMOVE: android.util.Log, Logr, STLevelLog, AppSTLogger imports\n\n"

        "FORMAT: raw Kotlin only. No markdown. No backticks. No explanations.\n"
        "Preserve all indentation and blank lines.\n"
    )


def call_groq(code: str, chunk_index: Optional[int] = None, is_retry: bool = False) -> str:
    url   = "https://api.groq.com/openai/v1/chat/completions"
    label = f"chunk {chunk_index}" if chunk_index is not None else "fichier"

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "messages": [
            {
                "role": "system",
                "content": "You are a code transformer. Output raw Kotlin only. Never truncate. Return every line."
            },
            {
                "role": "user",
                "content": build_chunk_prompt(is_retry=is_retry) + "\n\nFRAGMENT:\n" + code
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        except requests.exceptions.Timeout:
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            logger.warning(f"⏱️ Timeout sur {label} (tentative {attempt+1}/{MAX_RETRIES}). Attente {wait}s...")
            time.sleep(wait)
            continue
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erreur réseau Groq: {e}")

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", RATE_LIMIT_BASE_WAIT * (2 ** attempt)))
            logger.warning(
                f"🚦 Rate limit sur {label} (tentative {attempt+1}/{MAX_RETRIES}). "
                f"Attente {retry_after}s..."
            )
            time.sleep(retry_after)
            continue

        if resp.status_code != 200:
            raise Exception(f"Groq error {resp.status_code}: {resp.text[:200]}")

        data   = resp.json()
        choice = data["choices"][0]

        finish_reason = choice.get("finish_reason", "")
        if finish_reason == "length":
            logger.warning(f"  ⚠️ {label}: finish_reason=length — réponse coupée par max_tokens")

        content = choice["message"]["content"]
        content = re.sub(r"```.*?\n", "", content)
        content = content.replace("```", "")
        return content.strip()

    raise Exception(f"Max retries atteint ({MAX_RETRIES}) sur {label} — rate limit persistant")


# ================= VALIDATION FINALE =================

def validate_refactoring(original: str, refactored: str, filepath: str) -> tuple[bool, str]:
    orig_lines   = original.splitlines()
    new_lines    = refactored.splitlines()

    orig_non_log = [l.strip() for l in orig_lines if l.strip() and not is_log_related(l)]
    new_non_log  = [l.strip() for l in new_lines  if l.strip() and not is_log_related(l)]

    if orig_non_log == new_non_log:
        logger.info("✅ Validation finale OK — code non-log identique")
        return True, "ok"

    orig_set = set(orig_non_log)
    new_set  = set(new_non_log)
    added    = new_set  - orig_set
    removed  = orig_set - new_set

    # Tolérance globale : max 2 lignes non-log différentes sur tout le fichier
    if len(added) <= 2 and len(removed) <= 2:
        for line in list(added)[:2]:
            logger.warning(f"  Ligne ajoutée:    '{line[:80]}'")
        for line in list(removed)[:2]:
            logger.warning(f"  Ligne supprimée:  '{line[:80]}'")
        logger.info(
            f"✅ Validation finale OK — différences mineures tolérées "
            f"({len(added)} ajout(s), {len(removed)} suppression(s))"
        )
        return True, "ok"

    msg = f"❌ REJETÉ: {len(added)} lignes non-log ajoutées, {len(removed)} supprimées"
    logger.error(f"{filepath}: {msg}")
    for line in list(added)[:3]:
        logger.error(f"  + '{line[:80]}'")
    for line in list(removed)[:3]:
        logger.error(f"  - '{line[:80]}'")
    return False, msg


# ================= REFACTOR =================

def refactor_chunk_with_retry(chunk: str, chunk_index: int, total: int) -> str:
    """
    Envoie un chunk à Groq.
    ⭐ Les chunks triviaux (< MIN_CHUNK_SIZE_FOR_CHECKS chars) passent sans validation
       pour éviter les faux positifs sur des `}` ou fins de fichier.
    """
    result = call_groq(chunk, chunk_index=chunk_index)

    # ⭐ Chunk trivial : pas de validation (faux positifs garantis)
    if len(chunk.strip()) < MIN_CHUNK_SIZE_FOR_CHECKS:
        logger.info(f"  ✅ Chunk {chunk_index}/{total}: trivial ({len(chunk)} chars) — pas de validation")
        return result

    # Check 1 : marqueurs de troncature explicites
    if detect_truncation(result):
        logger.warning(f"  ⚠️ Chunk {chunk_index}/{total}: troncature détectée — retry...")
        time.sleep(REQUEST_DELAY)
        result = call_groq(chunk, chunk_index=chunk_index, is_retry=True)
        if detect_truncation(result):
            raise Exception(f"Chunk {chunk_index}/{total}: troncature persistante après retry")

    # Check 2 : perte de lignes NON-LOG uniquement
    ok, reason = check_chunk_non_log_loss(chunk, result, chunk_index)
    if not ok:
        logger.warning(f"  ⚠️ Chunk {chunk_index}/{total}: perte code métier — retry...")
        time.sleep(REQUEST_DELAY)
        result = call_groq(chunk, chunk_index=chunk_index, is_retry=True)
        ok, reason = check_chunk_non_log_loss(chunk, result, chunk_index)
        if not ok:
            raise Exception(reason)

    return result


def refactor_file(repo_path: Path, filepath: str) -> str:
    full_path = repo_path / filepath

    try:
        original_code = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"{filepath} - erreur lecture: {e}"

    if not any(x in original_code for x in ["Log.", "AppSTLogger", "Logr."]):
        return f"{filepath} - skipped (pas de logs)"

    logger.info(f"🤖 Refactoring {filepath} ({len(original_code):,} chars)...")

    try:
        if len(original_code) > MAX_FILE_SIZE:
            chunks = split_code(original_code)
            total  = len(chunks)
            logger.info(f"📦 Grand fichier: divisé en {total} chunks de ~{CHUNK_SIZE} chars")

            refactored_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"  Chunk {i+1}/{total} ({len(chunk):,} chars)...")
                try:
                    result = refactor_chunk_with_retry(chunk, i + 1, total)
                    refactored_chunks.append(result)
                except Exception as chunk_err:
                    logger.error(
                        f"❌ Échec chunk {i+1}/{total}: {chunk_err}. "
                        f"{len(refactored_chunks)}/{total} chunks traités — fichier ignoré."
                    )
                    return f"{filepath} - error (chunk {i+1}/{total}): {str(chunk_err)[:120]}"

                # ⭐ Délai progressif plafonné à MAX_INTER_CHUNK_DELAY
                if i < total - 1:
                    delay = min(REQUEST_DELAY * (i + 1), MAX_INTER_CHUNK_DELAY)
                    logger.info(f"  Pause {delay}s avant chunk suivant...")
                    time.sleep(delay)

            new_code = "\n".join(refactored_chunks)

        else:
            result = call_groq(original_code)

            if detect_truncation(result):
                logger.warning("⚠️ Troncature sur fichier entier — retry...")
                time.sleep(REQUEST_DELAY)
                result = call_groq(original_code, is_retry=True)

            new_code = result
            time.sleep(REQUEST_DELAY)

        # Validation finale sur le fichier complet
        valid, reason = validate_refactoring(original_code, new_code, filepath)
        if not valid:
            logger.error(f"🚫 Refactoring rejeté pour {filepath}: {reason}")
            return f"{filepath} - rejeté (LLM a modifié du code non-log)"

        full_path.write_text(new_code, encoding="utf-8")
        logger.info(f"✅ {filepath} refactorisé")
        return f"{filepath} - refactored"

    except Exception as e:
        logger.error(f"❌ Erreur refactor {filepath}: {e}")
        return f"{filepath} - error: {str(e)[:100]}"


# ================= COMMIT =================

def commit_and_push(repo_path: Path):
    run_git(["git", "config", "user.name", "Refactor Bot"], cwd=repo_path)
    run_git(["git", "config", "user.email", "bot@refactor.local"], cwd=repo_path)
    run_git(["git", "add", "."], cwd=repo_path)

    status = run_git(["git", "status", "--porcelain"], cwd=repo_path)
    if not status:
        logger.info("Nothing to commit.")
        return

    run_git(["git", "commit", "-m", "🤖 Auto refactor logs"], cwd=repo_path)
    run_git(["git", "push"], cwd=repo_path)
    logger.info("✅ Push successful.")


# ================= API =================

@app.post("/refactor")
def run_refactor(request: RefactorRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")

    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(500, "Missing env vars")

    repo_path = clone_repo(request.repo_url, request.branch)
    files = get_changed_files(repo_path, request.base_ref)

    if not files:
        return {"status": "ok", "message": "No .kt changes", "processed_files": []}

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(refactor_file, repo_path, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(result)

    if any("refactored" in r for r in results):
        commit_and_push(repo_path)

    return {
        "status": "success",
        "processed_files": results
    }


@app.get("/")
def root():
    return {
        "message": "Refactor Agent API Active",
        "version": "12.0-stable",
        "groq_model": GROQ_MODEL
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "groq_model": GROQ_MODEL,
        "chunk_size": CHUNK_SIZE,
        "max_tokens_out": MAX_TOKENS_OUT,
        "max_workers": MAX_WORKERS,
        "request_delay": REQUEST_DELAY,
        "max_retries": MAX_RETRIES,
        "rate_limit_base_wait": RATE_LIMIT_BASE_WAIT,
        "max_inter_chunk_delay": MAX_INTER_CHUNK_DELAY,
        "non_log_line_loss_max": NON_LOG_LINE_LOSS_MAX,
        "min_chunk_size_for_checks": MIN_CHUNK_SIZE_FOR_CHECKS
    }
