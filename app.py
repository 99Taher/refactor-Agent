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
MAX_FILES             = 10
MAX_WORKERS           = 1
GROQ_TIMEOUT          = 120
CLONE_DEPTH           = 500
REQUEST_DELAY         = 3         # ⬆️ 2 → 3 (pause entre chunks)
MAX_RETRIES           = 5
MAX_INTER_CHUNK_DELAY = 10        # ⬆️ 5 → 10 (max pause progressive)

CHUNK_SIZE            = 15000     # ⬇️ 35000 → 15000 (CRITIQUE)
MAX_FILE_SIZE         = 60000     # ⬆️ 40000 → 60000 (traiter plus de fichiers)
MAX_TOKENS_OUT        = 8000      # ⬇️ 32000 → 8000 (output plus court = plus rapide)

RATE_LIMIT_BASE_WAIT  = 15
MIN_CHUNK_SIZE_FOR_CHECKS = 100

NON_LOG_DIFF_MAX_PER_CHUNK = 3
NON_LOG_DIFF_MAX_FINAL     = 5

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
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


def _try_diff(repo_path: Path, base: str, dot: str) -> Optional[str]:
    try:
        return run_git(["git", "diff", "--name-only", f"{base}{dot}HEAD", "--"], cwd=repo_path)
    except Exception as e:
        logger.warning(f"  git diff {dot} échoué: {e}")
        return None


def get_changed_files(repo_path: Path, base_ref: str) -> List[str]:
    logger.info(f"Detecting changed files (base={base_ref})...")
    try:
        run_git([
            "git", "fetch", "origin",
            f"refs/heads/{base_ref}:refs/remotes/origin/{base_ref}",
            f"--depth={CLONE_DEPTH}"
        ], cwd=repo_path)
    except Exception:
        try:
            run_git(["git", "fetch", "--all", f"--depth={CLONE_DEPTH}"], cwd=repo_path)
        except Exception:
            pass

    branches_raw = run_git(["git", "branch", "-r"], cwd=repo_path, check=False)
    branches = [b.strip() for b in branches_raw.splitlines()]
    origin_branch = f"origin/{base_ref}"

    if origin_branch in branches:
        base_ref_for_diff = origin_branch
        logger.info(f"✅ Diff avec {origin_branch}")
    elif "origin/HEAD" in branches:
        base_ref_for_diff = "origin/HEAD"
        logger.warning("⚠️ Fallback origin/HEAD")
    else:
        base_ref_for_diff = "HEAD^"
        logger.warning("⚠️ Fallback HEAD^")

    diff = _try_diff(repo_path, base_ref_for_diff, "...")
    if diff is None:
        logger.warning("⚠️ Three-dot échoué — essai two-dot...")
        diff = _try_diff(repo_path, base_ref_for_diff, "..")
    if diff is None:
        logger.warning("⚠️ Two-dot échoué — deepen 500...")
        try:
            run_git(["git", "fetch", "--deepen=500", "origin", base_ref], cwd=repo_path, check=False)
        except Exception:
            pass
        diff = _try_diff(repo_path, base_ref_for_diff, "...") or _try_diff(repo_path, base_ref_for_diff, "..")

    if not diff:
        logger.info("Aucun fichier .kt modifié trouvé")
        return []

    files = [f for f in diff.splitlines() if f.strip().endswith(".kt")]
    logger.info(f"📝 {len(files)} fichiers .kt trouvés")
    return files[:MAX_FILES]


# ================= PATTERNS =================
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
    r"|^\s*//\s*rest of (the )?code\s*$"
    r"|^\s*//\s*\[truncated\]\s*$"
    r")",
    re.IGNORECASE | re.MULTILINE
)
APPLOGGER_LINE = re.compile(r"^\s*AppLogger\.[diwev]\(.*\)\s*$")


def is_log_related(line: str) -> bool:
    stripped = line.strip()
    return bool(LOG_PATTERNS.search(stripped)) or bool(IMPORT_PATTERNS.match(stripped))


def detect_truncation(code: str) -> bool:
    return bool(TRUNCATION_PATTERNS.search(code))


def get_non_log_lines(code: str) -> List[str]:
    """Retourne les lignes non-log non-vides, strippées."""
    return [l.strip() for l in code.splitlines() if l.strip() and not is_log_related(l)]


def diff_non_log(original: str, refactored: str) -> Tuple[List[str], List[str]]:
    """
    ⭐ Comparaison par SET de contenu — pas par count.
    Retourne (lignes_supprimées, lignes_ajoutées).
    """
    orig_set = set(get_non_log_lines(original))
    new_set  = set(get_non_log_lines(refactored))
    removed  = sorted(orig_set - new_set)
    added    = sorted(new_set  - orig_set)
    return removed, added


def deduplicate_applogger(code: str) -> Tuple[str, int]:
    """Filet de sécurité Python : doublons AppLogger consécutifs inter-chunks."""
    lines = code.splitlines(keepends=True)
    result = []
    removed = 0
    i = 0
    while i < len(lines):
        current = lines[i]
        if APPLOGGER_LINE.match(current.rstrip("\n")):
            j = i + 1
            while j < len(lines) and lines[j].rstrip("\n") == current.rstrip("\n"):
                j += 1
            removed += j - i - 1
            result.append(current)
            i = j
        else:
            result.append(current)
            i += 1
    return "".join(result), removed


# ================= VALIDATION =================
def check_chunk_non_log_loss(original: str, refactored: str, chunk_index: int) -> Tuple[bool, str]:
    """
    ⭐ Validation par SET : compte les lignes non-log réellement supprimées ou ajoutées.
    Une perte de count peut masquer des modifications réelles (ajout + suppression simultanés).
    """
    removed, added = diff_non_log(original, refactored)

    orig_log = sum(1 for l in original.splitlines()   if l.strip() and is_log_related(l))
    new_log  = sum(1 for l in refactored.splitlines() if l.strip() and is_log_related(l))

    logger.info(
        f"  📊 Chunk {chunk_index}: "
        f"non-log Δ={len(removed)} supprimées / {len(added)} ajoutées | "
        f"log {orig_log}→{new_log} (dedup LLM: {orig_log - new_log})"
    )

    total_diff = len(removed) + len(added)
    if total_diff <= NON_LOG_DIFF_MAX_PER_CHUNK:
        if removed or added:
            for l in removed[:2]: logger.warning(f"    - '{l[:80]}'")
            for l in added[:2]:   logger.warning(f"    + '{l[:80]}'")
        return True, "ok"

    for l in removed[:3]: logger.error(f"    - '{l[:80]}'")
    for l in added[:3]:   logger.error(f"    + '{l[:80]}'")
    return False, f"Δ non-log = {len(removed)} supprimées + {len(added)} ajoutées (max={NON_LOG_DIFF_MAX_PER_CHUNK})"


def validate_refactoring(original: str, refactored: str, filepath: str) -> Tuple[bool, str]:
    """Validation finale par SET sur le fichier complet assemblé."""
    removed, added = diff_non_log(original, refactored)
    total_diff = len(removed) + len(added)

    orig_log = sum(1 for l in original.splitlines()   if l.strip() and is_log_related(l))
    new_log  = sum(1 for l in refactored.splitlines() if l.strip() and is_log_related(l))

    logger.info(
        f"  🔍 Validation finale: Δ non-log = {len(removed)} supprimées / {len(added)} ajoutées | "
        f"log {orig_log}→{new_log} (dedup total: {orig_log - new_log})"
    )

    if total_diff <= NON_LOG_DIFF_MAX_FINAL:
        if removed or added:
            for l in removed[:3]: logger.warning(f"    - '{l[:80]}'")
            for l in added[:3]:   logger.warning(f"    + '{l[:80]}'")
        return True, "ok"

    for l in removed[:5]: logger.error(f"    - '{l[:80]}'")
    for l in added[:5]:   logger.error(f"    + '{l[:80]}'")
    return False, f"Δ non-log = {len(removed)} supprimées + {len(added)} ajoutées (max={NON_LOG_DIFF_MAX_FINAL})"


# ================= GROQ =================
def build_prompt(is_retry: bool = False) -> str:
    retry_prefix = (
        "⚠️ YOUR PREVIOUS RESPONSE WAS REJECTED because you modified non-log source lines.\n"
        "THIS TIME: output every single non-log line EXACTLY as it appears in the input.\n\n"
    ) if is_retry else ""

    return (
        retry_prefix +
        "You are a Kotlin Refactoring Expert.\n"
        "Your mission: Clean up and deduplicate logging in this Android code.\n\n"
        "🚨 CRITICAL — OUTPUT RULES:\n"
        " - Return THE COMPLETE CODE. Every single line. No truncation.\n"
        " - NEVER use '// ...', '...', or any placeholder.\n"
        " - DO NOT modify business logic, control flow, variable names, function signatures.\n\n"
        "1. CONVERSION RULES:\n"
        "   Log.d/i/w/e OR Logr.d/i/w/e → AppLogger.d/i/w/e(tag, msg)\n"
        "   AppSTLogger.appendLogST(...) → AppLogger.d/i/w/e selon le niveau\n\n"
        "2. DEDUPLICATION RULE (CRITICAL):\n"
        "   Merge tous les AppLogger consécutifs avec le même tag ET même message en UN SEUL.\n\n"
        "3. IMPORTS:\n"
        "   ADD: import com.honeywell.domain.managers.loggerApp.AppLogger\n"
        "   REMOVE: tous les anciens imports Log / Logr / AppSTLogger / STLevelLog\n\n"
        "Return ONLY raw source code. NO markdown, NO explanations."
    )


def call_groq(code: str, chunk_index: Optional[int] = None, is_retry: bool = False) -> str:
    url   = "https://api.groq.com/openai/v1/chat/completions"
    label = f"chunk {chunk_index}" if chunk_index is not None else "fichier"
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "messages": [
            {"role": "system", "content": "Output raw Kotlin code only. Return every line. Never truncate."},
            {"role": "user", "content": build_prompt(is_retry=is_retry) + "\n\nCODE:\n" + code}
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
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", RATE_LIMIT_BASE_WAIT * (2 ** attempt)))
                logger.warning(f"🚦 Rate limit {label} → attente {wait}s (tentative {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise Exception(f"Groq error {resp.status_code}: {resp.text[:200]}")
            choice = resp.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                logger.warning(f"  ⚠️ {label}: finish_reason=length — réponse tronquée par max_tokens")
            content = choice["message"]["content"]
            content = re.sub(r"```.*?\n", "", content).replace("```", "").strip()
            return content
        except requests.exceptions.Timeout:
            wait = RATE_LIMIT_BASE_WAIT * (2 ** attempt)
            logger.warning(f"⏱️ Timeout {label} → attente {wait}s (tentative {attempt+1})")
            time.sleep(wait)

    raise Exception(f"Max retries atteint sur {label}")


# ================= REFACTOR =================
def refactor_chunk_with_retry(chunk: str, chunk_index: int, total: int) -> str:
    logger.info(f"  → Chunk {chunk_index}/{total} ({len(chunk):,} chars) → Groq...")
    result = call_groq(chunk, chunk_index=chunk_index)

    if len(chunk.strip()) < MIN_CHUNK_SIZE_FOR_CHECKS:
        logger.info(f"  ✅ Chunk {chunk_index}/{total}: trivial — pas de validation")
        return result

    truncated      = detect_truncation(result)
    ok, reason     = check_chunk_non_log_loss(chunk, result, chunk_index)

    if truncated:
        logger.warning(f"  ⚠️ Chunk {chunk_index}/{total}: troncature détectée — retry...")
    elif not ok:
        logger.warning(f"  ⚠️ Chunk {chunk_index}/{total}: {reason} — retry...")

    if truncated or not ok:
        time.sleep(REQUEST_DELAY)
        result = call_groq(chunk, chunk_index=chunk_index, is_retry=True)
        ok2, reason2 = check_chunk_non_log_loss(chunk, result, chunk_index)
        if ok2:
            logger.info(f"  ✅ Chunk {chunk_index}/{total}: retry OK")
        else:
            logger.error(f"  ❌ Chunk {chunk_index}/{total}: retry échoué — {reason2} (on continue)")

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
            logger.info(f"📦 Fichier découpé en {total} chunks de ~{CHUNK_SIZE} chars")

            refactored_chunks = []
            for i, chunk in enumerate(chunks):
                result = refactor_chunk_with_retry(chunk, i + 1, total)
                refactored_chunks.append(result)
                if i < total - 1:
                    delay = min(REQUEST_DELAY * (i + 1), MAX_INTER_CHUNK_DELAY)
                    logger.info(f"  Pause {delay}s...")
                    time.sleep(delay)

            new_code = "\n".join(refactored_chunks)
        else:
            logger.info(f"  → Fichier complet ({len(original_code):,} chars) → Groq...")
            new_code = call_groq(original_code)
            if detect_truncation(new_code):
                logger.warning("⚠️ Troncature fichier entier — retry...")
                new_code = call_groq(original_code, is_retry=True)

        # Filet de sécurité : doublons inter-chunks
        new_code, dedup_count = deduplicate_applogger(new_code)
        if dedup_count:
            logger.info(f"  🔁 Filet Python: {dedup_count} doublon(s) AppLogger inter-chunks supprimé(s)")

        # Validation finale par SET
        valid, reason = validate_refactoring(original_code, new_code, filepath)
        if not valid:
            logger.error(f"🚫 {filepath} rejeté: {reason}")
            return f"{filepath} - rejeté: {reason}"

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
    if run_git(["git", "status", "--porcelain"], cwd=repo_path):
        run_git(["git", "commit", "-m", "🤖 Auto refactor logs"], cwd=repo_path)
        run_git(["git", "push"], cwd=repo_path)
        logger.info("✅ Push successful.")
    else:
        logger.info("Nothing to commit.")


# ================= API =================
@app.post("/refactor")
def run_refactor(request: RefactorRequest, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(403, "Unauthorized")
    if not GROQ_API_KEY or not GITHUB_TOKEN:
        raise HTTPException(500, "Missing env vars")

    repo_path = clone_repo(request.repo_url, request.branch)
    files     = get_changed_files(repo_path, request.base_ref)

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

    return {"status": "success", "processed_files": results}


@app.get("/")
def root():
    return {
        "message": "Refactor Agent API Active",
        "version": "16.4-stable",
        "groq_model": GROQ_MODEL
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "groq_model": GROQ_MODEL,
        "deduplication": "LLM + Python safety net",
        "chunk_size": CHUNK_SIZE,
        "max_file_size": MAX_FILE_SIZE,
        "max_tokens_out": MAX_TOKENS_OUT,
        "non_log_diff_max_per_chunk": NON_LOG_DIFF_MAX_PER_CHUNK,
        "non_log_diff_max_final": NON_LOG_DIFF_MAX_FINAL
    }

