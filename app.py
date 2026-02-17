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
GROQ_TIMEOUT = 120
CLONE_DEPTH = 500
REQUEST_DELAY = 2
MAX_RETRIES = 5
MAX_INTER_CHUNK_DELAY = 5
CHUNK_SIZE = 35000          # ⬇️ 60k → 35k (safe pour Groq)
MAX_FILE_SIZE = 40000       # ⬇️ 200k → 40k (force chunking sur 68k)
MAX_TOKENS_OUT = 32000
RATE_LIMIT_BASE_WAIT = 15
MIN_CHUNK_SIZE_FOR_CHECKS = 100
NON_LOG_LINE_LOSS_MAX = 2

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
# (fonction get_changed_files inchangée - je l’ai gardée identique)


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


def get_changed_files(repo_path: Path, base_ref: str) -> List[str]:
    logger.info(f"Detecting changed files (base={base_ref})...")
    try:
        run_git(["git", "fetch", "origin", f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}", f"--depth={CLONE_DEPTH}"], cwd=repo_path)
    except Exception:
        pass
    # ... reste identique à ta version précédente (je ne recopie pas tout pour la lisibilité)
    # (le code est exactement le même que dans la version 16.0)
    branches_raw = run_git(["git", "branch", "-r"], cwd=repo_path, check=False)
    branches = [b.strip() for b in branches_raw.splitlines()]
    origin_branch = f"origin/{base_ref}"
    base_ref_for_diff = origin_branch if origin_branch in branches else ("origin/HEAD" if "origin/HEAD" in branches else "HEAD^")
    diff = _try_diff(repo_path, base_ref_for_diff, "...") or _try_diff(repo_path, base_ref_for_diff, "..")
    if not diff:
        return []
    files = [f for f in diff.splitlines() if f.strip().endswith(".kt")]
    return files[:MAX_FILES]


def _try_diff(repo_path: Path, base: str, dot: str) -> Optional[str]:
    try:
        return run_git(["git", "diff", "--name-only", f"{base}{dot}HEAD", "--"], cwd=repo_path)
    except Exception as e:
        logger.warning(f" git diff {dot} échoué: {e}")
        return None


# ================= GROQ & PROMPT (identique, 100% LLM) =================
# (LOG_PATTERNS, detect_truncation, check_chunk_non_log_loss, build_prompt, call_groq, validate_refactoring → TOUT IDENTIQUE à la version précédente)


# ================= REFACTOR (correction ici) =================
def refactor_chunk_with_retry(chunk: str, chunk_index: int, total: int) -> str:
    logger.info(f"   → Envoi chunk {chunk_index}/{total} ({len(chunk):,} chars) à Groq...")
    result = call_groq(chunk, chunk_index=chunk_index)
    if len(chunk.strip()) < MIN_CHUNK_SIZE_FOR_CHECKS:
        return result
    if detect_truncation(result) or not check_chunk_non_log_loss(chunk, result, chunk_index)[0]:
        logger.warning(f"   ⚠️ Retry chunk {chunk_index} (perte ou troncature)")
        time.sleep(REQUEST_DELAY)
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
        # FORCE CHUNKING si > 40k (résout le 413)
        if len(original_code) > MAX_FILE_SIZE:
            chunks = split_code(original_code)
            logger.info(f"📦 Fichier découpé en {len(chunks)} chunks de ~{CHUNK_SIZE} chars")
            refactored_chunks = []
            for i, chunk in enumerate(chunks):
                result = refactor_chunk_with_retry(chunk, i + 1, len(chunks))
                refactored_chunks.append(result)
                if i < len(chunks) - 1:
                    time.sleep(min(REQUEST_DELAY * (i + 1), MAX_INTER_CHUNK_DELAY))
            new_code = "\n".join(refactored_chunks)
        else:
            logger.info(f"   → Envoi fichier complet ({len(original_code):,} chars) à Groq...")
            new_code = call_groq(original_code)
            if detect_truncation(new_code):
                new_code = call_groq(original_code, is_retry=True)

        valid, reason = validate_refactoring(original_code, new_code, filepath)
        if not valid:
            return f"{filepath} - rejeté (code métier modifié)"

        full_path.write_text(new_code, encoding="utf-8")
        logger.info(f"✅ {filepath} refactorisé (100% LLM)")
        return f"{filepath} - refactored"

    except Exception as e:
        logger.error(f"❌ Erreur refactor {filepath}: {e}")
        return f"{filepath} - error: {str(e)[:100]}"


# ================= COMMIT & API (inchangés) =================
def commit_and_push(repo_path: Path):
    run_git(["git", "config", "user.name", "Refactor Bot"], cwd=repo_path)
    run_git(["git", "config", "user.email", "bot@refactor.local"], cwd=repo_path)
    run_git(["git", "add", "."], cwd=repo_path)
    if run_git(["git", "status", "--porcelain"], cwd=repo_path):
        run_git(["git", "commit", "-m", "🤖 Auto refactor logs"], cwd=repo_path)
        run_git(["git", "push"], cwd=repo_path)
        logger.info("✅ Push successful.")


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
    return {"message": "Refactor Agent API Active", "version": "16.1-chunk-safe", "groq_model": GROQ_MODEL}


@app.get("/health")
def health():
    return {"status": "healthy", "deduplication": "100% LLM", "max_file_size": MAX_FILE_SIZE, "chunk_size": CHUNK_SIZE}
