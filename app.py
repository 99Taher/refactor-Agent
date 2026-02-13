import os
import re
import shutil
import time
import requests
import subprocess
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
API_SECRET = os.getenv("API_SECRET")


class RefactorRequest(BaseModel):
    repo_url: str
    base_ref: str = "main"
    branch: str = "auto-refactor"


def clone_repo(repo_url, branch):
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    clone_url = repo_url.replace(
        "https://",
        f"https://{GITHUB_TOKEN}@"
    )

    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)

    subprocess.run(["git", "clone", "-b", branch, clone_url], check=True)
    return repo_name


def get_changed_files(repo_path, base_ref):
    os.chdir(repo_path)
    cmd = f"git diff --name-only origin/{base_ref}...HEAD"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    files = [
        f for f in output.splitlines()
        if f.endswith(".kt") and os.path.exists(f)
    ]
    os.chdir("..")
    return files


def refactor_file(repo_path, filepath):
    full_path = os.path.join(repo_path, filepath)

    with open(full_path, "r", encoding="utf-8") as f:
        code = f.read()

    if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
        return f"{filepath} - No logs found"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a Kotlin expert."},
            {"role": "user", "content": code}
        ],
        "temperature": 0
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        new_code = response.json()["choices"][0]["message"]["content"].strip()
        new_code = re.sub(r"^```kotlin\s*|^```\s*", "", new_code)
        new_code = re.sub(r"\s*```$", "", new_code)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        return f"{filepath} refactored"
    else:
        return f"API Error {response.status_code}"


def commit_and_push(repo_path):
    os.chdir(repo_path)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Auto refactor logs"], check=True)
    subprocess.run(["git", "push"], check=True)
    os.chdir("..")
@app.get("/")
async def root():
    return {
        "message": "Refactor Agent API Is Active",
        "usage": "POST /refactor avec les headers x-api-key et Content-Type",
        "docs": "/docs"
    }

@app.post("/refactor")
def run_refactor(
    request: RefactorRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="Missing environment variables")

    repo_path = clone_repo(request.repo_url, request.branch)

    files = get_changed_files(repo_path, request.base_ref)

    results = []
    for f in files:
        result = refactor_file(repo_path, f)
        results.append(result)
        time.sleep(1)

    commit_and_push(repo_path)

    return {"processed_files": results}

