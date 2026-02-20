import os
import re
import sys
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
from litellm import Router
import litellm
import threading


# ================= CONFIG =================

CLONE_DEPTH    = 500
REQUEST_DELAY  = 10
CHUNK_SIZE     = 30000   # ~30k chars ≈ ~9,000–10,000 tokens input
MAX_TOKENS_OUT = 12000

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET   = os.getenv("API_SECRET")

# Quick-scan prefixes to decide whether a file needs refactoring at all
LOG_PREFIXES = (
    "Log.d(", "Log.i(", "Log.w(", "Log.e(", "Log.v(",
    "Logr.d(", "Logr.i(", "Logr.w(", "Logr.e(", "Logr.v(",
    "AppSTLogger.appendLogST(",
)

APPLOGGER_IMPORT = "import com.honeywell.domain.managers.loggerApp.AppLogger"

# ==========================================
# ================= LLM ROUTER =============
# ==========================================
#
# Render environment variables to set:
# ┌─────────────────────────┬──────────────────────────────────────────────────┐
# │ Variable                │ Value                                            │
# ├─────────────────────────┼──────────────────────────────────────────────────┤
# │ GROQ_API_KEY            │ gsk_...          (get at console.groq.com)       │
# │ GROQ_MODEL              │ groq/qwen/qwen3-32b          (optional)          │
# │ OPENROUTER_API_KEY      │ sk-or-...        (get at openrouter.ai)          │
# │ OPENROUTER_MODEL        │ openrouter/deepseek/deepseek-r1:free  (optional) │
# │ HUGGINGFACE_API_KEY     │ hf_...           (get at huggingface.co)         │
# │ HUGGINGFACE_MODEL       │ huggingface/Qwen/Qwen2.5-72B-Instruct (optional) │
# │ GITHUB_TOKEN            │ ghp_...                                          │
# │ API_SECRET              │ your_secret                                      │
# └─────────────────────────┴──────────────────────────────────────────────────┘
#
# The Router switches providers automatically when:
#   - Rate limit hit (429)  → cooldown provider 60s, try next immediately
#   - Timeout               → retry 3x then switch to next provider
#   - API down (5xx)        → switch to next provider immediately
#   - Auth error (401)      → skip that provider entirely for the whole job


def _build_model_list() -> list:
    """
    Builds provider list from env vars.
    Only includes providers whose API key is actually set.
    Order = priority: first entry is tried first.
    """
    models = []

    # ── 1. Groq (primary — fastest, 14,400 req/day free) ─────────
    if os.getenv("GROQ_API_KEY"):
        models.append({
            "model_name": "llm",
            "litellm_params": {
                "model":   os.getenv("GROQ_MODEL", "groq/qwen/qwen3-32b"),
                "api_key": os.getenv("GROQ_API_KEY"),
            }
        })

    # ── 2. OpenRouter (fallback 1 — 300+ models, free tier) ──────
    if os.getenv("OPENROUTER_API_KEY"):
        models.append({
            "model_name": "llm",
            "litellm_params": {
                "model":   os.getenv("OPENROUTER_MODEL", "openrouter/deepseek/deepseek-r1:free"),
                "api_key": os.getenv("OPENROUTER_API_KEY"),
            }
        })

    # ── 3. Hugging Face (fallback 2 — free inference API) ────────
    if os.getenv("HUGGINGFACE_API_KEY"):
        models.append({
            "model_name": "llm",
            "litellm_params": {
                "model":   os.getenv("HUGGINGFACE_MODEL", "huggingface/Qwen/Qwen2.5-72B-Instruct"),
                "api_key": os.getenv("HUGGINGFACE_API_KEY"),
            }
        })

    return models


_model_list = _build_model_list()

if not _model_list:
    raise RuntimeError(
        "No LLM provider configured. "
        "Set at least one of: GROQ_API_KEY, OPENROUTER_API_KEY, HUGGINGFACE_API_KEY."
    )

# Router handles retries, fallbacks, rate limits and timeouts automatically
router = Router(
    model_list=_model_list,
    num_retries=3,          # retry same provider N times before switching
    retry_after=5,          # seconds between retries on same provider
    timeout=60,             # switch provider after this many seconds
    cooldown_time=60,       # cooldown a provider after rate limit (seconds)
    routing_strategy="simple-shuffle",
    set_verbose=False,
)

# Silence LiteLLM internal logs
litellm.suppress_debug_info = True

# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_active_providers = [m["litellm_params"]["model"] for m in _model_list]
logger.info(f"LLM Router initialized with {len(_model_list)} provider(s): {_active_providers}")

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


def clean_llm_output(content: str) -> str:
    if not content:
        return ""
    # Strip <think>...</think> blocks produced by reasoning models (e.g. Qwen3)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
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
        repo_name,
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


# ================= QUICK SCAN =================

def needs_refactoring(code: str) -> bool:
    """Fast pre-check — avoids sending files with nothing to do to the LLM."""
    for prefix in LOG_PREFIXES:
        if prefix in code:
            return True
    lines = code.splitlines()
    for i in range(len(lines) - 1):
        s = lines[i].strip()
        if s.startswith("AppLogger.") and s == lines[i + 1].strip():
            return True
    return False


# ================= CHUNKING =================

def split_into_chunks(lines: List[str]) -> List[List[str]]:
    """
    Split file lines into chunks that each stay under CHUNK_SIZE chars.
    Chunks always split on line boundaries so the LLM receives valid code blocks.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # +1 for the newline
        if current_size + line_size > CHUNK_SIZE and current:
            chunks.append(current)
            current = [line]
            current_size = line_size
        else:
            current.append(line)
            current_size += line_size

    if current:
        chunks.append(current)

    return chunks


# ================= LLM CALL =================

def call_llm(prompt: str) -> str:
    """
    Single entry point for all LLM calls.
    The Router automatically handles retries, provider switching,
    rate limit cooldowns and timeouts — no manual retry loop needed.
    """
    response = router.completion(
        model="llm",            # alias — Router picks the actual provider
        temperature=0,
        max_tokens=MAX_TOKENS_OUT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Kotlin log-refactoring assistant. "
                    "Follow instructions exactly. Return only what is asked."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return clean_llm_output(response.choices[0].message.content)


# ================= LLM PROMPT =================

def is_valid_kotlin_output(llm_output: str, original: str) -> bool:
    """
    Returns False if the LLM returned a prose explanation instead of Kotlin code.
    """
    if not llm_output:
        return False
    if len(llm_output) < len(original) * 0.20:
        kotlin_tokens = ("fun ", "val ", "var ", "class ", "import ", "package ", "object ", "return ", "{", "}")
        if not any(t in llm_output for t in kotlin_tokens):
            return False
    return True


def build_refactor_prompt(chunk: str, is_first_chunk: bool, has_applogger_import: bool) -> str:
    import_instruction = ""
    if is_first_chunk:
        if has_applogger_import:
            import_instruction = (
                f"  - The import `{APPLOGGER_IMPORT}` is already present — keep it.\n"
            )
        else:
            import_instruction = (
                f"  - Add `{APPLOGGER_IMPORT}` next to the other import statements.\n"
            )
        import_instruction += (
            "  - REMOVE any of these imports if present:\n"
            "      import android.util.Log\n"
            "      import android.util.Logr\n"
            "      import com.streamwide.smartms.lib.core.api.logger.AppSTLogger\n"
            "      import com.streamwide.smartms.lib.core.api.logger.STLevelLog\n"
            "      import com.streamwide.smartms.lib.core.api.AppSTLogger\n"
            "      import com.streamwide.smartms.lib.core.STLevelLog\n"
        )

    return "\n".join([
        "Refactor the Kotlin code below following ONLY these rules:",
        "",
        "LOG CONVERSION:",
        "  - Log.d(TAG, msg)   ->  AppLogger.d(TAG, msg)",
        "  - Log.i(TAG, msg)   ->  AppLogger.i(TAG, msg)",
        "  - Log.w(TAG, msg)   ->  AppLogger.w(TAG, msg)",
        "  - Log.e(TAG, msg)   ->  AppLogger.e(TAG, msg)",
        "  - Logr.d/i/w/e(TAG, msg) -> AppLogger.d/i/w/e(TAG, msg)",
        "  - AppSTLogger.appendLogST(STLevelLog.DEBUG, t, m) -> AppLogger.d(t, m)",
        "  - AppSTLogger.appendLogST(STLevelLog.INFO,  t, m) -> AppLogger.i(t, m)",
        "  - AppSTLogger.appendLogST(STLevelLog.WARN,  t, m) -> AppLogger.w(t, m)",
        "  - AppSTLogger.appendLogST(STLevelLog.ERROR, t, m) -> AppLogger.e(t, m)",
        "  - Lines already using AppLogger.x(...) -> keep them exactly as-is.",
        "",
        "DEDUPLICATION:",
        "  - If two or more consecutive lines are identical AppLogger calls -> keep only ONE.",
        "",
        "IMPORTS (apply only if import statements appear in this block):",
        import_instruction,
        "STRICT RULES:",
        "  - Preserve the EXACT indentation and formatting of every line.",
        "  - Do NOT touch any line that is not a log call or a relevant import.",
        "  - Do NOT add explanations, comments, or markdown — return raw Kotlin code only.",
        "  - If there is NOTHING to convert or fix, return the code EXACTLY as received, character for character.",
        "  - NEVER write sentences like 'No changes needed' or 'The code already uses AppLogger'.",
        "  - Your response MUST always be valid Kotlin code, nothing else.",
        "",
        "CODE:",
        chunk,
    ])


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
        has_applogger_import = APPLOGGER_IMPORT in original_code

        # ── Small file: send whole file as one chunk ──────────────────────────
        if len(original_code) <= CHUNK_SIZE:
            logger.info(f"{filepath} - small file, single LLM call")
            prompt = build_refactor_prompt(
                chunk=original_code,
                is_first_chunk=True,
                has_applogger_import=has_applogger_import,
            )
            new_code = call_llm(prompt)
            n_chunks = 1

            if not is_valid_kotlin_output(new_code, original_code):
                logger.warning(f"{filepath} - LLM returned prose instead of code, keeping original")
                return f"{filepath} - skipped (LLM returned explanation instead of code)"

        # ── Large file: split into chunks, refactor each, reassemble ──────────
        else:
            chunks = split_into_chunks(lines)
            n_chunks = len(chunks)
            logger.info(f"{filepath} - large file, {n_chunks} chunks")

            refactored_parts: List[str] = []
            for i, chunk_lines in enumerate(chunks):
                chunk_text = "\n".join(chunk_lines)
                prompt = build_refactor_prompt(
                    chunk=chunk_text,
                    is_first_chunk=(i == 0),
                    has_applogger_import=has_applogger_import,
                )
                logger.info(f"{filepath} - chunk {i + 1}/{n_chunks} ({len(chunk_text)} chars)")
                result = call_llm(prompt)

                if not is_valid_kotlin_output(result, chunk_text):
                    logger.warning(f"{filepath} - chunk {i + 1} returned prose, keeping original chunk")
                    result = chunk_text

                refactored_parts.append(result)

                if i < n_chunks - 1:
                    time.sleep(REQUEST_DELAY)

            new_code = "\n".join(refactored_parts)

        # ── Nothing changed ───────────────────────────────────────────────────
        if new_code.strip() == original_code.strip():
            return f"{filepath} - skipped (LLM made no changes)"

        full_path.write_text(new_code, encoding="utf-8")
        logger.info(f"{filepath} - written ({n_chunks} chunk(s))")
        return f"{filepath} - refactored ({n_chunks} chunk(s))"

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
def run_refactor(
    request: RefactorRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(None),
):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")
    if not _model_list:
        raise HTTPException(500, "No LLM provider configured")

    job_id = f"job_{int(time.time())}"
    background_tasks.add_task(run_refactor_job, job_id, request)

    return {
        "status":    "started",
        "job_id":    job_id,
        "message":   "Refactoring running in background",
        "providers": _active_providers,
    }


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
        "message":   "Refactor Agent API Active",
        "version":   "24.0-litellm",
        "providers": _active_providers,
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status":     "healthy",
        "providers":  _active_providers,
        "chunk_size": CHUNK_SIZE,
    }
