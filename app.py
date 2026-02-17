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
MAX_FILES = 10
MAX_WORKERS = 1
GROQ_TIMEOUT = 180
CLONE_DEPTH = 500
REQUEST_DELAY = 5                    # ⬆️ plus de calme
MAX_RETRIES = 5
MAX_INTER_CHUNK_DELAY = 12           # ⬆️
CHUNK_SIZE = 30000                   # ⬇️ + overlap
MAX_FILE_SIZE = 35000                # force chunking tôt
MAX_TOKENS_OUT = 32000
RATE_LIMIT_BASE_WAIT = 35            # ⬆️ très important pour Groq
MIN_CHUNK_SIZE_FOR_CHECKS = 100
NON_LOG_LINE_LOSS_MAX = 6            # ⬆️ 2 → 6 (tolérance + overlap)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

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
    branch: str = "auto-refactor"


# ================= UTIL =================
def run_git(cmd: List[str], cwd=None, check=True):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise Exception(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def split_code(code: str) -> List[str]:
    """Split avec overlap pour éviter les pertes aux frontières de chunks"""
    chunks = []
    start = 0
    overlap = 2500  # ~50 lignes Kotlin
    while start < len(code):
        end = min(start + CHUNK_SIZE, len(code))
        if end < len(code):
            split = code.rfind("\nfun ", start, end)
            if split == -1 or split <= start:
                split = code.rfind("\n\n", start, end)
            if split != -1 and split > start:
                end = split
        chunk = code[start:end]
        chunks.append(chunk)
        start = max(start + CHUNK_SIZE - overlap, end) if end < len(code) else end
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
    run_git(["git", "clone", "--depth", str(CLONE_DEPTH), "--no-single-branch", "-b", branch, clone_url, repo_name])
    return repo_path


def _try_diff(repo_path: Path, base: str, dot: str) -> Optional[str]:
    try:
        return run_git(["git", "diff", "--name-only", f"{base}{dot}HEAD", "--"], cwd=repo_path)
    except Exception as e:
        logger.warning(f" git diff {dot} échoué: {e}")
        return None


def get_changed_files(repo_path: Path, base_ref: str) -> List[str]:
    logger.info(f"Detecting changed files (base={base_ref})...")
    try:
        run_git(["git", "fetch", "origin", f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}", f"--depth={CLONE_DEPTH}"], cwd=repo_path)
    except Exception:
        pass

    branches_raw = run_git(["git", "branch", "-r"], cwd=repo_path, check=False)
    branches = [b.strip() for b in branches_raw.splitlines()]
    origin_branch = f"origin/{base_ref}"
    base_ref_for_diff = origin_branch if origin_branch in branches else ("origin/HEAD" if "origin/HEAD" in branches else "HEAD^")

    diff = _try_diff(repo_path, base_ref_for_diff, "...") or _try_diff(repo_path, base_ref_for_diff, "..")
    if not diff:
        logger.info("Aucun fichier .kt modifié trouvé")
        return []
    files = [f for f in diff.splitlines() if f.strip().endswith(".kt")]
    logger.info(f"📝 {len(files)} fichiers .kt trouvés")
    return files[:MAX_FILES]


# ================= GROQ =================
LOG_PATTERNS = re.compile(r"(Log\.[diwev]\(|Logr\.[diwev]\(|AppSTLogger\.appendLogST\(|AppLogger\.[diwev]\()")
IMPORT_PATTERNS = re.compile(r"^import\s+(android\.util\.Log|com\.honeywell.*[Ll]og|com\.st.*[Ll]og|.*Logr|.*STLevelLog|.*AppLogger|.*AppSTLogger)")
TRUNCATION_PATTERNS = re.compile(r"(^\s*//\s*\.\.\.\s*$|^\s*/\*\s*\.\.\.\s*\*/\s*$|^\s*\.\.\.\s*$|^\s*//\s*rest of (the )?code|^\s*//\s*\[truncated\])", re.IGNORECASE | re.MULTILINE)


def is_log_related(line: str) -> bool:
    stripped = line.strip()
    return bool(LOG_PATTERNS.search(stripped)) or bool(IMPORT_PATTERNS.match(stripped))


def detect_truncation(code: str) -> bool:
    return bool(TRUNCATION_PATTERNS.search(code))


def check_chunk_non_log_loss(original: str, refactored: str, chunk_index: int) -> tuple[bool, str]:
    orig_non_log = [l.strip() for l in original.splitlines() if l.strip() and not is_log_related(l)]
    new_non_log = [l.strip() for l in refactored.splitlines() if l.strip() and not is_log_related(l)]
    lost = len(orig_non_log) - len(new_non_log)
    logger.info(f" 📊 Chunk {chunk_index}: non-log {len(orig_non_log)} → {len(new_non_log)} (perte: {lost})")
    if lost <= NON_LOG_LINE_LOSS_MAX:
        return True, "ok"
    return False, f"perte trop importante: {lost} lignes non-log"


def build_prompt(is_retry: bool = False) -> str:
    retry_prefix = (
        "⚠️ PREVIOUS RESPONSE REJECTED – you removed business code!\n"
        "THIS TIME: KEEP EVERY SINGLE NON-LOG LINE 100% UNCHANGED. "
        "Do not remove, modify or shorten anything that is not a logging call.\n\n"
    ) if is_retry else ""

    return (
        retry_prefix +
        "You are a Kotlin Refactoring Expert.\n"
        "Mission: Clean & deduplicate logging ONLY.\n\n"
        "🚨 STRICT RULES:\n"
        "- Return THE COMPLETE CODE – every single line, no truncation, no '...'\n"
        "- NEVER change business logic, variables, comments, empty lines, control flow.\n"
        "- KEEP EVERY NON-LOG LINE EXACTLY AS IS.\n\n"
        "1. CONVERSION:\n"
        "   Log.d/i/w/e → AppLogger.d/i/w/e(tag, msg)\n"
        "   AppSTLogger... → AppLogger.d/i/w/e selon niveau\n\n"
        "2. DEDUPLICATION (CRITICAL):\n"
        "   Merge consecutive or nearby AppLogger with SAME tag AND SAME message into ONE.\n\n"
        "3. IMPORTS:\n"
        "   ADD: import com.honeywell.domain.managers.loggerApp.AppLogger\n"
        "   REMOVE: all old Log / Logr / AppSTLogger / STLevelLog imports\n\n"
        "Return ONLY raw Kotlin code. NO explanations."
    )


def call_groq(code: str, chunk_index: Optional[int] = None, is_retry: bool = False) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    label = f"chunk {chunk_index}" if chunk_index is not None else "fichier"
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "messages": [
            {"role": "system", "content": "Output raw Kotlin code only. Never truncate."},
            {"role": "user", "content": build_prompt(is_retry) + "\n\nCODE:\n" + code}
        ],
        "temperature": 0
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", RATE_LIMIT_BASE_WAIT * (2 ** attempt)))
                logger.warning(f"🚦 Rate limit {label} → attente {wait}s (tentative {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise Exception(f"Groq {resp.status_code}: {resp.text[:150]}")
            content = resp.json()["choices"][0]["message"]["content"]
            return re.sub(r"```.*?\n", "", content).replace("```", "").strip()
        except requests.exceptions.Timeout:
            wait = RATE_LIMIT_BASE_WAIT * (1.5 ** attempt)
            logger.warning(f"⏱️ Timeout {label} → attente {wait}s")
            time.sleep(wait)
    raise Exception(f"Max retries sur {label}")


def validate_refactoring(original: str, refactored: str, filepath: str) -> tuple[bool, str]:
    orig_non_log = [l.strip() for l in original.splitlines() if l.strip() and not is_log_related(l)]
    new_non_log = [l.strip() for l in refactored.splitlines() if l.strip() and not is_log_related(l)]
    if abs(len(orig_non_log) - len(new_non_log)) <= NON_LOG_LINE_LOSS_MAX + 2:
        return True, "ok"
    return False, "trop de modifications code métier"


# ================= REFACTOR =================
def refactor_chunk_with_retry(chunk: str, chunk_index: int, total: int) -> str:
    logger.info(f"   → Envoi chunk {chunk_index}/{total} ({len(chunk):,} chars)...")
    result = call_groq(chunk, chunk_index=chunk_index)
    if len(chunk.strip()) < MIN_CHUNK_SIZE_FOR_CHECKS:
        return result

    ok, reason = check_chunk_non_log_loss(chunk, result, chunk_index)
    if detect_truncation(result) or not ok:
        logger.warning(f"   ⚠️ Retry chunk {chunk_index} ({reason})")
        time.sleep(REQUEST_DELAY + 3)
        result = call_groq(chunk, chunk_index=chunk_index, is_retry=True)
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
            logger.info(f"📦 Découpé en {len(chunks)} chunks (~{CHUNK_SIZE} chars + overlap)")
            refactored_chunks = []
            for i, chunk in enumerate(chunks):
                result = refactor_chunk_with_retry(chunk, i + 1, len(chunks))
                refactored_chunks.append(result)
                if i < len(chunks) - 1:
                    time.sleep(min(REQUEST_DELAY * (i + 2), MAX_INTER_CHUNK_DELAY))
            new_code = "\n".join(refactored_chunks)
        else:
            new_code = call_groq(original_code)
            if detect_truncation(new_code):
                new_code = call_groq(original_code, is_retry=True)

        valid, reason = validate_refactoring(original_code, new_code, filepath)
        if not valid:
            return f"{filepath} - rejeté"

        full_path.write_text(new_code, encoding="utf-8")
        logger.info(f"✅ {filepath} terminé (100% LLM)")
        return f"{filepath} - refactored"

    except Exception as e:
        logger.error(f"❌ Erreur {filepath}: {e}")
        return f"{filepath} - error: {str(e)[:100]}"


# ================= COMMIT & API =================
def commit_and_push(repo_path: Path):
    run_git(["git", "config", "user.name", "Refactor Bot"], cwd=repo_path)
    run_git(["git", "config", "user.email", "bot@refactor.local"], cwd=repo_path)
    run_git(["git", "add", "."], cwd=repo_path)
    if run_git(["git", "status", "--porcelain"], cwd=repo_path):
        run_git(["git", "commit", "-m", "🤖 Auto refactor logs"], cwd=repo_path)
        run_git(["git", "push"], cwd=repo_path)
        logger.info("✅ Push OK")


@app.post("/refactor")
def run_refactor(request: RefactorRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")
    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(500, "Missing env vars")

    repo_path = clone_repo(request.repo_url, request.branch)
    files = get_changed_files(repo_path, request.base_ref)

    if not files:
        return {"status": "ok", "message": "No .kt changes"}

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(refactor_file, repo_path, f): f for f in files}
        for future in as_completed(futures):
            results.append(future.result())

    if any("refactored" in r for r in results):
        commit_and_push(repo_path)

    return {"status": "success", "processed_files": results}


@app.get("/")
def root():
    return {"message": "Refactor Agent v16.3", "status": "ready"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "deduplication": "100% LLM + overlap",
        "chunk_size": CHUNK_SIZE,
        "max_file_size": MAX_FILE_SIZE,
        "non_log_tolerance": NON_LOG_LINE_LOSS_MAX,
        "rate_limit_wait": RATE_LIMIT_BASE_WAIT
    }
