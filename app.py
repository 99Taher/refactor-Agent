import os
import re
import sys
import time
import shutil
import logging
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import threading


# ================= CONFIG =================

MAX_WORKERS = 1
GROQ_TIMEOUT = 120
CLONE_DEPTH = 500
REQUEST_DELAY = 10
MAX_RETRIES = 8
CHUNK_SIZE = 12000
MAX_TOKENS_OUT = 8000
RATE_LIMIT_BASE_WAIT = 15

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

# Log call prefixes — simple string checks, no regex
LOG_PREFIXES = (
    "Log.d(", "Log.i(", "Log.w(", "Log.e(", "Log.v(",
    "Logr.d(", "Logr.i(", "Logr.w(", "Logr.e(", "Logr.v(",
    "AppSTLogger.appendLogST(",
    "AppLogger.",  # include already-converted lines for deduplication
)

IMPORT_PREFIXES_TO_REMOVE = (
    "import android.util.Log",
    "import android.util.Logr",
    "import com.streamwide.smartms.lib.core.api.logger.AppSTLogger",
    "import com.streamwide.smartms.lib.core.api.logger.STLevelLog",
    "import com.streamwide.smartms.lib.core.api.AppSTLogger",
    "import com.streamwide.smartms.lib.core.STLevelLog",
)
APPLOGGER_IMPORT = "import com.honeywell.domain.managers.loggerApp.AppLogger"

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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


def clean_llm_output(content: str) -> str:
    if not content:
        return ""
    content = re.sub(r"```kotlin", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```", "", content)
    return content.strip()


# ================= GIT =================

def clone_repo(repo_url: str, branch: str) -> Path:
    if not repo_url.startswith("https://"):
        raise HTTPException(400, "repo_url must be https")
    if not GITHUB_TOKEN:
        raise HTTPException(500, "Missing GITHUB_TOKEN")

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
        run_git(["git", "fetch", "origin", base_ref], cwd=repo_path)
    except Exception:
        pass

    diff = run_git(
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        cwd=repo_path,
        check=False,
    )

    if not diff:
        return []

    files = [f for f in diff.splitlines() if f.endswith(".kt")]
    logger.info(f"{len(files)} Kotlin files found")
    return files


# ================= LINE DETECTION =================

def is_log_line(line: str) -> bool:
    """Detect log call lines using simple startswith — no regex."""
    return line.strip().startswith(LOG_PREFIXES)


def needs_refactoring(code: str) -> bool:
    for prefix in LOG_PREFIXES:
        if prefix in code:
            return True
    # Detect duplicate consecutive AppLogger lines
    lines = code.splitlines()
    for i in range(len(lines) - 1):
        s = lines[i].strip()
        if s.startswith("AppLogger.") and s == lines[i + 1].strip():
            return True
    return False


# ================= GROQ =================

def call_groq_with_prompt(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a Kotlin log conversion assistant. Follow instructions exactly. Return only what is asked."
            },
            {"role": "user", "content": prompt}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)

            if response.status_code == 429:
                wait = min(RATE_LIMIT_BASE_WAIT * (2 ** attempt), 90)
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise Exception(response.text)

            data = response.json()
            if "choices" not in data:
                raise Exception("Invalid Groq response")

            return clean_llm_output(data["choices"][0]["message"]["content"])

        except requests.exceptions.Timeout:
            wait = min(RATE_LIMIT_BASE_WAIT * (2 ** attempt), 90)
            logger.warning(f"Timeout. Retrying in {wait}s")
            time.sleep(wait)

    raise Exception("Max retries reached for Groq")


# ================= LLM PROMPTS =================

def build_log_conversion_prompt(snippet: str) -> str:
    """
    Prompt for converting ONLY log lines.
    Each line is prefixed with LINE_N: so we can reinject at exact positions.
    """
    return "\n".join([
        "You are given a list of Kotlin log call lines extracted from a source file.",
        "Each line is prefixed with LINE_N: where N is its original line number.",
        "",
        "CONVERT each line following these rules:",
        "  Log.d(TAG, msg)  ->  AppLogger.d(TAG, msg)",
        "  Log.i(TAG, msg)  ->  AppLogger.i(TAG, msg)",
        "  Log.w(TAG, msg)  ->  AppLogger.w(TAG, msg)",
        "  Log.e(TAG, msg)  ->  AppLogger.e(TAG, msg)",
        "  Logr.d/i/w/e(TAG, msg) -> AppLogger.d/i/w/e(TAG, msg)",
        "  AppSTLogger.appendLogST(STLevelLog.DEBUG, t, m) -> AppLogger.d(t, m)",
        "  AppSTLogger.appendLogST(STLevelLog.INFO,  t, m) -> AppLogger.i(t, m)",
        "  AppSTLogger.appendLogST(STLevelLog.WARN,  t, m) -> AppLogger.w(t, m)",
        "  AppSTLogger.appendLogST(STLevelLog.ERROR, t, m) -> AppLogger.e(t, m)",
        "  If line is already AppLogger.x(...) -> keep it exactly as-is.",
        "",
        "DEDUPLICATION:",
        "  If two or more consecutive lines are identical AppLogger calls -> keep only ONE.",
        "",
        "RULES:",
        "  - Keep the LINE_N: prefix on every output line.",
        "  - Keep the EXACT indentation of each line.",
        "  - Return ONLY the converted lines with their LINE_N: prefixes.",
        "  - No explanation, no markdown, no extra text.",
        "",
        "Lines to convert:",
        snippet,
    ])


def build_import_prompt(snippet: str) -> str:
    """Prompt for handling only the import lines."""
    return "\n".join([
        "You are given Kotlin import lines extracted from a source file.",
        "Each line is prefixed with LINE_N: where N is its original line number.",
        "",
        "RULES:",
        "  - If the line imports android.util.Log -> output: LINE_N: DELETE",
        "  - If the line imports Logr, STLevelLog, or AppSTLogger -> output: LINE_N: DELETE",
        f"  - If the line is already: {APPLOGGER_IMPORT} -> keep it as-is",
        f"  - If AppLogger import is NOT present in the list, add: LINE_NEW: {APPLOGGER_IMPORT}",
        "  - All other import lines -> keep them EXACTLY as-is.",
        "",
        "Return ONLY the import lines with their LINE_N: prefixes. No explanation.",
        "",
        "Import lines:",
        snippet,
    ])


# ================= EXTRACT / CONVERT / INJECT =================

def extract_log_lines(lines: List[str]) -> Dict[int, str]:
    """Extract log call lines with their indices."""
    return {i: line for i, line in enumerate(lines) if is_log_line(line)}


def extract_import_lines(lines: List[str]) -> Tuple[Dict[int, str], int]:
    """Extract import lines to handle, and find last import index."""
    import_lines = {}
    last_import_idx = -1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import "):
            last_import_idx = i
        if stripped.startswith(IMPORT_PREFIXES_TO_REMOVE) or stripped == APPLOGGER_IMPORT:
            import_lines[i] = line

    return import_lines, last_import_idx


def split_log_lines_into_batches(log_lines: Dict[int, str]) -> List[List[Tuple[int, str]]]:
    """Split log lines into batches respecting CHUNK_SIZE."""
    items = sorted(log_lines.items())
    batches = []
    current_batch = []
    current_size = 0

    for idx, line in items:
        size = len(line) + 10  # +10 for LINE_N: prefix
        if current_size + size > CHUNK_SIZE and current_batch:
            batches.append(current_batch)
            current_batch = [(idx, line)]
            current_size = size
        else:
            current_batch.append((idx, line))
            current_size += size

    if current_batch:
        batches.append(current_batch)

    return batches


def call_llm_for_log_lines(batch: List[Tuple[int, str]], filepath: str, batch_num: int, total: int) -> Dict[int, str]:
    """Send only log lines to LLM and parse LINE_N: response."""
    snippet = "\n".join(f"LINE_{idx}: {line}" for idx, line in batch)
    prompt = build_log_conversion_prompt(snippet)

    logger.info(f"{filepath} - log batch {batch_num}/{total} ({len(batch)} lines)")
    result = call_groq_with_prompt(prompt)

    converted = {}
    for rline in result.splitlines():
        if rline.startswith("LINE_") and ": " in rline:
            sep = rline.index(": ")
            try:
                idx = int(rline[5:sep])
                converted[idx] = rline[sep + 2:]
            except ValueError:
                pass

    return converted


def call_llm_for_imports(import_lines: Dict[int, str], filepath: str) -> Tuple[Dict[int, str], List[str]]:
    """Send only import lines to LLM and parse response."""
    snippet = "\n".join(f"LINE_{idx}: {line}" for idx, line in sorted(import_lines.items()))
    prompt = build_import_prompt(snippet)

    logger.info(f"{filepath} - processing {len(import_lines)} import lines")
    result = call_groq_with_prompt(prompt)

    changes = {}
    new_imports = []

    for rline in result.splitlines():
        if rline.startswith("LINE_NEW:"):
            new_imports.append(rline[9:].strip())
        elif rline.startswith("LINE_") and ": " in rline:
            sep = rline.index(": ")
            try:
                idx = int(rline[5:sep])
                val = rline[sep + 2:].strip()
                changes[idx] = None if val == "DELETE" else val
            except ValueError:
                pass

    return changes, new_imports


def rebuild_file(
    lines: List[str],
    converted_logs: Dict[int, str],
    import_changes: Dict[int, str],
    new_imports: List[str],
    last_import_idx: int,
) -> List[str]:
    """
    Rebuild file lines:
    - Replace log lines with LLM-converted versions
    - Apply import deletions/additions
    - Copy ALL other lines exactly as-is
    """
    result = []
    applogger_injected = False

    for i, line in enumerate(lines):

        # Handle import changes (delete or replace)
        if i in import_changes:
            if import_changes[i] is None:
                continue  # DELETE
            result.append(import_changes[i])
            # Inject new imports right after last import
            if i == last_import_idx and new_imports and not applogger_injected:
                for imp in new_imports:
                    result.append(imp)
                applogger_injected = True
            continue

        # Inject new imports after last import if not already done
        if i == last_import_idx and new_imports and not applogger_injected:
            result.append(line)
            for imp in new_imports:
                result.append(imp)
            applogger_injected = True
            continue

        # Replace log lines with LLM output
        if i in converted_logs:
            result.append(converted_logs[i])
            continue

        # Everything else: copy EXACTLY as-is
        result.append(line)

    return result


# ================= REFACTOR FILE =================

def refactor_file(repo_path: Path, filepath: str) -> str:
    full_path = repo_path / filepath

    try:
        original_code = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"{filepath} - read error: {e}"

    if not needs_refactoring(original_code):
        return f"{filepath} - skipped (no pattern found)"

    logger.info(f"Refactoring {filepath} ({len(original_code)} chars)")

    try:
        lines = original_code.splitlines()

        # Step 1: Extract log lines and import lines
        log_lines = extract_log_lines(lines)
        import_lines, last_import_idx = extract_import_lines(lines)

        logger.info(f"{filepath} - {len(log_lines)} log lines | {len(import_lines)} import lines")

        # Step 2: Convert log lines via LLM (batched by CHUNK_SIZE)
        converted_logs = {}
        if log_lines:
            batches = split_log_lines_into_batches(log_lines)
            logger.info(f"{filepath} - {len(batches)} LLM batch(es) for log lines")
            for i, batch in enumerate(batches):
                result = call_llm_for_log_lines(batch, filepath, i + 1, len(batches))
                converted_logs.update(result)
                if i < len(batches) - 1:
                    time.sleep(REQUEST_DELAY)

        # Step 3: Convert imports via LLM
        import_changes = {}
        new_imports = []
        if import_lines:
            import_changes, new_imports = call_llm_for_imports(import_lines, filepath)

        # Step 4: Rebuild file — non-log lines copied EXACTLY as-is
        new_lines = rebuild_file(lines, converted_logs, import_changes, new_imports, last_import_idx)
        new_code = "\n".join(new_lines)

        if new_code.strip() == original_code.strip():
            return f"{filepath} - skipped (LLM made no changes)"

        full_path.write_text(new_code, encoding="utf-8")
        logger.info(f"{filepath} - written ({len(log_lines)} log lines converted, {len(batches) if log_lines else 0} batch(es))")
        return f"{filepath} - refactored ({len(log_lines)} log lines)"

    except Exception as e:
        return f"{filepath} - error: {str(e)[:100]}"


# ================= COMMIT =================

def commit_and_push(repo_path: Path):
    run_git(["git", "config", "user.name", "Refactor Bot"], cwd=repo_path)
    run_git(["git", "config", "user.email", "bot@refactor.local"], cwd=repo_path)
    run_git(["git", "add", "."], cwd=repo_path)

    if run_git(["git", "status", "--porcelain"], cwd=repo_path):
        run_git(["git", "commit", "-m", "🤖 Auto refactor logs"], cwd=repo_path)
        run_git(["git", "push", "--set-upstream", "origin", "HEAD"], cwd=repo_path)
        logger.info("Push successful")
    else:
        logger.info("Nothing to commit")


# ================= ASYNC JOBS =================

_jobs: dict = {}
_jobs_lock = threading.Lock()


def run_refactor_job(job_id: str, request: RefactorRequest):
    try:
        with _jobs_lock:
            _jobs[job_id] = {"status": "running", "files": []}

        repo_path = clone_repo(request.repo_url, request.branch)
        files = get_changed_files(repo_path, request.base_ref)

        if not files:
            with _jobs_lock:
                _jobs[job_id] = {"status": "done", "files": [], "message": "No .kt changes"}
            return

        results = []
        for f in files:
            result = refactor_file(repo_path, f)
            results.append(result)
            with _jobs_lock:
                _jobs[job_id]["files"] = list(results)
            logger.info(f"[{job_id}] {result}")

        if any("refactored" in r for r in results):
            commit_and_push(repo_path)

        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "files": results}

    except Exception as e:
        logger.error(f"[{job_id}] Job failed: {e}")
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


# ================= API =================

@app.post("/refactor")
def run_refactor(request: RefactorRequest, background_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")
    if not GROQ_API_KEY:
        raise HTTPException(500, "Missing GROQ_API_KEY")

    job_id = f"job_{int(time.time())}"
    background_tasks.add_task(run_refactor_job, job_id, request)

    return {"status": "started", "job_id": job_id, "message": "Refactoring running in background"}


@app.get("/refactor/status/{job_id}")
def get_status(job_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")

    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    return {"job_id": job_id, **job}


@app.get("/")
def root():
    return {
        "message": "Refactor Agent API Active",
        "version": "22.0-stable",
        "model": GROQ_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "chunk_size": CHUNK_SIZE,
    }
