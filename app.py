import os
import re
import sys
import time
import shutil
import logging
import requests
import subprocess
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel


# ================= CONFIG =================

MAX_WORKERS = 1
GROQ_TIMEOUT = 120
CLONE_DEPTH = 500
REQUEST_DELAY = 2
MAX_RETRIES = 5
CHUNK_SIZE = 80000  # llama-3.3-70b-versatile: 128k context, safe chunk size
MAX_TOKENS_OUT = 32000
RATE_LIMIT_BASE_WAIT = 15

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")

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

    return files  # no limit


# ================= PROMPT =================

def build_refactor_prompt(code: str) -> str:
    prompt = "\n".join([
        "You are a Kotlin Refactoring Expert.",
        "Your mission: Clean up and deduplicate logging in this Android code.",
        "",
        "1. CONVERSION RULES:",
        "   - Log.d/i/w/e OR Logr.d/i/w/e -> AppLogger.d/i/w/e(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.DEBUG, tag, msg) -> AppLogger.d(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.INFO, tag, msg)  -> AppLogger.i(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.WARN, tag, msg)  -> AppLogger.w(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.ERROR, tag, msg) -> AppLogger.e(tag, msg)",
        "",
        "2. DEDUPLICATION RULE (CRITICAL):",
        "   - Merge consecutive lines of AppLogger with EXACT SAME tag and message into ONE.",
        "   - Example: Multiple AppLogger.e(MODULE, 'text') calls become just one.",
        "",
        "3. IMPORTS:",
        "   - ADD: 'import com.honeywell.domain.managers.loggerApp.AppLogger'.",
        "   - REMOVE ONLY: android.util.Log, Logr, STLevelLog, and AppSTLogger imports.",
        "   - Do NOT remove or modify ANY other import that is not log-related.",
        "",
        "4. STRICT PRESERVATION RULES (CRITICAL - DO NOT VIOLATE):",
        "   - Do NOT change ANY line of code that is not a log call or a log-related import.",
        "   - Do NOT reformat, reorder, rename, or restructure anything.",
        "   - Do NOT modify comments, blank lines, indentation, or spacing.",
        "   - Do NOT touch business logic, function signatures, class structure, or variables.",
        "   - The only allowed changes are: log call conversions, deduplication, and imports.",
        "   - If a line is not a log call, copy it EXACTLY as-is, character by character.",
        "",
        "Return ONLY raw source code. NO markdown markers, NO explanations.",
    ])
    return f"{prompt}\n\nKotlin file to refactor:\n{code}"


def build_chunk_prompt(chunk_code: str) -> str:
    prompt = "\n".join([
        "You are a Kotlin Refactoring Expert.",
        "This is a continuation chunk of a larger Kotlin file. Apply ONLY the following:",
        "",
        "1. CONVERSION RULES:",
        "   - Log.d/i/w/e OR Logr.d/i/w/e -> AppLogger.d/i/w/e(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.DEBUG, tag, msg) -> AppLogger.d(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.INFO, tag, msg)  -> AppLogger.i(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.WARN, tag, msg)  -> AppLogger.w(tag, msg)",
        "   - AppSTLogger.appendLogST(STLevelLog.ERROR, tag, msg) -> AppLogger.e(tag, msg)",
        "",
        "2. DEDUPLICATION RULE (CRITICAL):",
        "   - Merge consecutive lines of AppLogger with EXACT SAME tag and message into ONE.",
        "",
        "3. STRICT PRESERVATION RULES (CRITICAL):",
        "   - Do NOT add or remove ANY imports (already handled in first chunk).",
        "   - Do NOT change ANY line that is not a log call.",
        "   - Do NOT reformat, reorder, rename, or restructure anything.",
        "   - Do NOT modify comments, blank lines, indentation, or spacing.",
        "   - Copy every non-log line EXACTLY as-is, character by character.",
        "",
        "Return ONLY raw source code. NO markdown markers, NO explanations.",
    ])
    return f"{prompt}\n\nKotlin chunk to refactor:\n{chunk_code}"


# ================= GROQ =================

def call_groq_with_prompt(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a Kotlin refactoring assistant. Return raw Kotlin code only, no markdown, no explanation."},
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
                wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
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
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            logger.warning(f"Timeout. Retrying in {wait}s")
            time.sleep(wait)

    raise Exception("Max retries reached for Groq")


# ================= REFACTOR =================

def needs_refactoring(code: str) -> bool:
    log_patterns = [
        "Log.d(", "Log.e(", "Log.w(", "Log.i(", "Log.v(",
        "Logr.d(", "Logr.e(", "Logr.w(", "Logr.i(", "Logr.v(",
        "AppSTLogger",
    ]
    if any(p in code for p in log_patterns):
        return True

    # Detect duplicate consecutive AppLogger lines
    lines = [l.strip() for l in code.splitlines()]
    for i in range(len(lines) - 1):
        if lines[i].startswith("AppLogger.") and lines[i] == lines[i + 1]:
            return True

    return False


def split_into_chunks(lines: List[str], chunk_size: int = CHUNK_SIZE) -> List[List[str]]:
    chunks = []
    current_chunk: List[str] = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1
        if current_size + line_size > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def refactor_file(repo_path: Path, filepath: str) -> str:
    full_path = repo_path / filepath

    try:
        original_code = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"{filepath} - read error: {e}"

    if not needs_refactoring(original_code):
        return f"{filepath} - skipped (no pattern found)"

    file_size = len(original_code)
    logger.info(f"Refactoring {filepath} ({file_size} chars)")

    try:
        # Small file: single call
        if file_size <= CHUNK_SIZE:
            new_code = call_groq_with_prompt(build_refactor_prompt(original_code))

            if not new_code or len(new_code) < 50:
                return f"{filepath} - rejected (empty result)"
            if new_code.strip() == original_code.strip():
                return f"{filepath} - skipped (LLM made no changes)"

            full_path.write_text(new_code, encoding="utf-8")
            logger.info(f"{filepath} - written successfully")
            return f"{filepath} - refactored"

        # Large file: chunk processing
        lines = original_code.splitlines()
        chunks = split_into_chunks(lines, CHUNK_SIZE)
        logger.info(f"{filepath} - large file split into {len(chunks)} chunks")

        refactored_chunks = []
        for i, chunk_lines in enumerate(chunks):
            chunk_code = "\n".join(chunk_lines)
            is_first = (i == 0)

            logger.info(f"{filepath} - chunk {i+1}/{len(chunks)} ({len(chunk_code)} chars)")

            prompt = build_refactor_prompt(chunk_code) if is_first else build_chunk_prompt(chunk_code)
            chunk_result = call_groq_with_prompt(prompt)

            if not chunk_result or len(chunk_result) < 10:
                logger.error(f"{filepath} - chunk {i+1} returned empty, aborting")
                return f"{filepath} - error: chunk {i+1} empty result"

            refactored_chunks.append(chunk_result)

            if i < len(chunks) - 1:
                time.sleep(REQUEST_DELAY)

        new_code = "\n".join(refactored_chunks)

        if new_code.strip() == original_code.strip():
            return f"{filepath} - skipped (LLM made no changes)"

        full_path.write_text(new_code, encoding="utf-8")
        logger.info(f"{filepath} - written successfully ({len(chunks)} chunks)")
        return f"{filepath} - refactored ({len(chunks)} chunks)"

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


# ================= API =================

@app.post("/refactor")
def run_refactor(request: RefactorRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")
    if not GROQ_API_KEY:
        raise HTTPException(500, "Missing GROQ_API_KEY")

    repo_path = clone_repo(request.repo_url, request.branch)
    files = get_changed_files(repo_path, request.base_ref)

    if not files:
        return {"status": "ok", "message": "No .kt changes"}

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(refactor_file, repo_path, f) for f in files]
        for future in as_completed(futures):
            results.append(future.result())

    if any("refactored" in r for r in results):
        commit_and_push(repo_path)

    return {"status": "success", "processed_files": results}


@app.get("/")
def root():
    return {
        "message": "Refactor Agent API Active",
        "version": "19.0-stable",
        "model": GROQ_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "chunk_size": CHUNK_SIZE,
    }
