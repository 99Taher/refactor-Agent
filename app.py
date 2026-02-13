import os
import re
import time
import requests
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class RefactorRequest(BaseModel):
    base_ref: str = "main"


def get_changed_files(base_ref):
    try:
        cmd = f"git diff --name-only origin/{base_ref}...HEAD"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        return [
            f for f in output.splitlines()
            if f.endswith(".kt") and os.path.exists(f)
        ]
    except Exception as e:
        print(f"Erreur Git Diff: {e}")
        return []


def refactor(filepath, api_key):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    if not any(x in code for x in ["Log.", "Logr.", "AppSTLogger.", "AppLogger."]):
        return f"{filepath} - No logs found"

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are a Kotlin Refactoring Expert.\n"
        "Your mission: Clean up and deduplicate logging in this Android code.\n\n"
        "Return ONLY raw source code."
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a Kotlin expert."},
            {"role": "user", "content": f"{prompt}\n\nCODE:\n{code}"}
        ],
        "temperature": 0
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        new_code = response.json()["choices"][0]["message"]["content"].strip()

        # Remove markdown markers if present
        new_code = re.sub(r"^```kotlin\s*|^```\s*", "", new_code)
        new_code = re.sub(r"\s*```$", "", new_code)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_code)

        return f"{filepath} refactored"
    else:
        return f"API Error {response.status_code}"


@app.post("/refactor")
def run_refactor(request: RefactorRequest):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY")

    files = get_changed_files(request.base_ref)

    results = []
    for f in files:
        result = refactor(f, api_key)
        results.append(result)
        time.sleep(1)

    return {"processed_files": results}
