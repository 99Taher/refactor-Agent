import os
import re
import sys
import time
import shutil
import logging
import requests
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
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
CHUNK_SIZE = 35000
MAX_FILE_SIZE = 40000
MAX_TOKENS_OUT = 32000
RATE_LIMIT_BASE_WAIT = 15
MIN_CHUNK_SIZE_FOR_CHECKS = 100

NON_LOG_DIFF_MAX_PER_CHUNK = 3
NON_LOG_DIFF_MAX_FINAL = 5

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
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
    """Nettoyage sécurisé de la réponse LLM"""
    if not content:
        return ""

    # Remove markdown fences
    content = re.sub(r"```kotlin", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```", "", content)

    # Remove accidental leading explanations
    content = content.strip()

    return content


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

    return files[:MAX_FILES]


# ================= GROQ =================

def call_groq(code: str, is_retry=False) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Return raw Kotlin code only."},
            {"role": "user", "content": code}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=GROQ_TIMEOUT,
            )

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

            content = data["choices"][0]["message"]["content"]
            return clean_llm_output(content)

        except requests.exceptions.Timeout:
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            logger.warning(f"Timeout. Retrying in {wait}s")
            time.sleep(wait)

    raise Exception("Max retries reached for Groq")


# ================= REFACTOR =================

def refactor_file(repo_path: Path, filepath: str) -> str:
    full_path = repo_path / filepath

    try:
        original_code = full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"{filepath} - read error: {e}"

    if "Log." not in original_code and "AppSTLogger" not in original_code:
        return f"{filepath} - skipped"

    logger.info(f"Refactoring {filepath}")

    try:
        new_code = call_groq(original_code)

        if not new_code or len(new_code) < 50:
            return f"{filepath} - rejected (empty result)"

        full_path.write_text(new_code, encoding="utf-8")

        return f"{filepath} - refactored"

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
        "version": "17.0-stable",
        "model": GROQ_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "max_file_size": MAX_FILE_SIZE,
        "chunk_size": CHUNK_SIZE,
    }
