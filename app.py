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

MAX_FILES = 10
MAX_WORKERS = 1
GROQ_TIMEOUT = 30
CLONE_DEPTH = 200
MAX_FILE_SIZE = 50000
REQUEST_DELAY = 2
MAX_RETRIES = 3
CHUNK_SIZE = 25000

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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

def run_git(cmd: List[str], cwd=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(result.stderr.strip())
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

    clone_url = repo_url.replace(
        "https://",
        f"https://{GITHUB_TOKEN}@"
    )

    logger.info("Cloning repository...")
    run_git([
        "git", "clone",
        "--depth", str(CLONE_DEPTH),
        "-b", branch,
        clone_url,
        repo_name
    ])

    return repo_path


def get_changed_files(repo_path: Path, base_ref: str):
    logger.info("Detecting changed files...")
    run_git(["git", "fetch", "origin", base_ref], cwd=repo_path)

    diff = run_git(
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        cwd=repo_path
    )

    files = [
        f for f in diff.splitlines()
        if f.endswith(".kt")
    ]

    return files[:MAX_FILES]


# ================= GROQ =================

def call_groq(code: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

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
            {"role": "system", "content": "Output raw code only."},
            {"role": "user", "content": prompt + "\n\nCODE:\n" + code}
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)

        if resp.status_code == 429:
            logger.warning("Rate limit. Waiting...")
            time.sleep(10)
            continue

        if resp.status_code != 200:
            raise Exception(resp.text)

        content = resp.json()["choices"][0]["message"]["content"]

        content = re.sub(r"```.*?\n", "", content)
        content = content.replace("```", "")

        return content.strip()

    raise Exception("Max retries reached")


# ================= REFACTOR =================

def refactor_file(repo_path: Path, filepath: str):
    full_path = repo_path / filepath

    code = full_path.read_text(encoding="utf-8")

    if not any(x in code for x in ["Log.", "AppSTLogger", "Logr."]):
        return f"{filepath} - skipped"

    logger.info(f"Refactoring {filepath}...")

    if len(code) > MAX_FILE_SIZE:
        chunks = split_code(code)
        new_code = ""
        for chunk in chunks:
            new_code += call_groq(chunk)
            time.sleep(REQUEST_DELAY)
    else:
        new_code = call_groq(code)

    full_path.write_text(new_code, encoding="utf-8")

    time.sleep(REQUEST_DELAY)
    return f"{filepath} - refactored"


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

    logger.info("Push successful.")


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
        return {"status": "ok", "message": "No .kt changes"}

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(refactor_file, repo_path, f): f
            for f in files
        }

        for future in as_completed(futures):
            results.append(future.result())

    if any("refactored" in r for r in results):
        commit_and_push(repo_path)

    return {
        "status": "success",
        "files": results
    }


@app.get("/health")
def health():
    return {
        "groq_model": GROQ_MODEL,
        "max_workers": MAX_WORKERS,
        "request_delay": REQUEST_DELAY
    }



