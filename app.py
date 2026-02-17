#!/usr/bin/env python3
"""
Refactor Agent - Script complet pour refactoring Kotlin avec déduplication LLM.
Gère : chunking, validation robuste, retry, rate limit, doublons inter-chunks.
"""
import os
import re
import sys
import time
import requests
import subprocess
from typing import List, Tuple, Optional


# ================= CONFIG =================
GROQ_MODEL            = "llama-3.3-70b-versatile"
MAX_TOKENS_OUT        = 32000
GROQ_TIMEOUT          = 120
MAX_RETRIES           = 5
RATE_LIMIT_BASE_WAIT  = 15
REQUEST_DELAY         = 2

# Chunking
CHUNK_SIZE            = 35000  # chars par chunk (~8k tokens)
MAX_FILE_SIZE         = 40000  # seuil pour activer le chunking
MIN_CHUNK_SIZE        = 100    # chunks triviaux ignorés

# Validation
NON_LOG_DIFF_MAX_PER_CHUNK = 3  # modifications tolérées par chunk
NON_LOG_DIFF_MAX_FINAL     = 5  # modifications tolérées sur fichier complet


# ================= PATTERNS =================
LOG_PATTERNS = re.compile(
    r"(Log\.[diwev]\(|Logr\.[diwev]\(|AppSTLogger\.appendLogST\(|AppLogger\.[diwev]\()"
)
IMPORT_PATTERNS = re.compile(
    r"^import\s+(android\.util\.Log|com\.honeywell.*[Ll]og|com\.st.*[Ll]og|.*Logr|.*STLevelLog|.*AppLogger|.*AppSTLogger)"
)
TRUNCATION_PATTERNS = re.compile(
    r"(^\s*//\s*\.\.\.\s*$|^\s*/\*\s*\.\.\.\s*\*/\s*$|^\s*\.\.\.\s*$|^\s*//\s*rest of (the )?code\s*$|^\s*//\s*\[truncated\]\s*$)",
    re.IGNORECASE | re.MULTILINE
)
APPLOGGER_LINE = re.compile(r"^\s*AppLogger\.[diwev]\(.*\)\s*$")


def is_log_related(line: str) -> bool:
    stripped = line.strip()
    return bool(LOG_PATTERNS.search(stripped)) or bool(IMPORT_PATTERNS.match(stripped))


def detect_truncation(code: str) -> bool:
    return bool(TRUNCATION_PATTERNS.search(code))


# ================= GIT =================
def get_changed_files(base_ref: str) -> List[str]:
    """Récupère les fichiers .kt modifiés avec fallback three-dot → two-dot."""
    print(f"🔍 Détection des fichiers modifiés (base={base_ref})...")

    # Essai 1 : three-dot (nécessite merge base)
    try:
        cmd = f"git diff --name-only origin/{base_ref}...HEAD"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        files = [f for f in output.splitlines() if f.endswith('.kt') and os.path.exists(f)]
        if files:
            print(f"📝 {len(files)} fichiers trouvés (three-dot)")
            return files
    except Exception:
        pass

    # Essai 2 : two-dot (pas besoin de merge base)
    try:
        cmd = f"git diff --name-only origin/{base_ref}..HEAD"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        files = [f for f in output.splitlines() if f.endswith('.kt') and os.path.exists(f)]
        if files:
            print(f"📝 {len(files)} fichiers trouvés (two-dot)")
            return files
    except Exception as e:
        print(f"⚠️ Erreur Git Diff: {e}")

    # Essai 3 : fallback HEAD^ (commit précédent)
    try:
        cmd = "git diff --name-only HEAD^ HEAD"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        files = [f for f in output.splitlines() if f.endswith('.kt') and os.path.exists(f)]
        if files:
            print(f"📝 {len(files)} fichiers trouvés (HEAD^)")
            return files
    except Exception:
        pass

    return []


# ================= CHUNKING =================
def split_code(code: str) -> List[str]:
    """Découpe le code en chunks de CHUNK_SIZE chars max, coupant sur les limites de fonction."""
    chunks = []
    start = 0
    while start < len(code):
        end = min(start + CHUNK_SIZE, len(code))
        if end < len(code):
            # Couper sur une fonction si possible
            split = code.rfind("\nfun ", start, end)
            if split != -1 and split > start:
                end = split
            else:
                # Sinon couper sur une ligne vide
                split = code.rfind("\n\n", start, end)
                if split != -1 and split > start:
                    end = split
        chunks.append(code[start:end])
        start = end
    return chunks


# ================= VALIDATION =================
def get_non_log_lines(code: str) -> List[str]:
    """Retourne les lignes non-log non-vides, strippées."""
    return [l.strip() for l in code.splitlines() if l.strip() and not is_log_related(l)]


def diff_non_log(original: str, refactored: str) -> Tuple[List[str], List[str]]:
    """
    Comparaison par SET de contenu — pas par count.
    Retourne (lignes_supprimées, lignes_ajoutées).
    """
    orig_set = set(get_non_log_lines(original))
    new_set  = set(get_non_log_lines(refactored))
    removed  = sorted(orig_set - new_set)
    added    = sorted(new_set  - orig_set)
    return removed, added


def check_chunk_validity(original: str, refactored: str, chunk_index: int) -> Tuple[bool, str]:
    """Validation par SET : compte les lignes non-log réellement modifiées."""
    removed, added = diff_non_log(original, refactored)

    orig_log = sum(1 for l in original.splitlines()   if l.strip() and is_log_related(l))
    new_log  = sum(1 for l in refactored.splitlines() if l.strip() and is_log_related(l))

    print(f"    📊 Chunk {chunk_index}: Δ non-log={len(removed)} supprimées/{len(added)} ajoutées | "
          f"log {orig_log}→{new_log} (dedup LLM: {orig_log - new_log})")

    total_diff = len(removed) + len(added)
    if total_diff <= NON_LOG_DIFF_MAX_PER_CHUNK:
        return True, "ok"

    for l in removed[:2]: print(f"      - '{l[:70]}'")
    for l in added[:2]:   print(f"      + '{l[:70]}'")
    return False, f"Δ={total_diff} (max={NON_LOG_DIFF_MAX_PER_CHUNK})"


def validate_final(original: str, refactored: str) -> Tuple[bool, str]:
    """Validation finale par SET sur le fichier complet assemblé."""
    removed, added = diff_non_log(original, refactored)
    total_diff = len(removed) + len(added)

    print(f"  🔍 Validation finale: Δ non-log = {len(removed)} supprimées / {len(added)} ajoutées")

    if total_diff <= NON_LOG_DIFF_MAX_FINAL:
        if removed or added:
            for l in removed[:2]: print(f"    - '{l[:70]}'")
            for l in added[:2]:   print(f"    + '{l[:70]}'")
        return True, "ok"

    for l in removed[:3]: print(f"    - '{l[:70]}'")
    for l in added[:3]:   print(f"    + '{l[:70]}'")
    return False, f"Δ={total_diff} (max={NON_LOG_DIFF_MAX_FINAL})"


# ================= DEDUPLICATION PYTHON =================
def deduplicate_applogger(code: str) -> Tuple[str, int]:
    """Filet de sécurité : supprime les doublons AppLogger consécutifs inter-chunks."""
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


# ================= GROQ API =================
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
        "   AppSTLogger.appendLogST(STLevelLog.DEBUG, ...) → AppLogger.d(tag, msg)\n"
        "   AppSTLogger.appendLogST(STLevelLog.INFO, ...)  → AppLogger.i(tag, msg)\n"
        "   AppSTLogger.appendLogST(STLevelLog.WARN, ...)  → AppLogger.w(tag, msg)\n"
        "   AppSTLogger.appendLogST(STLevelLog.ERROR, ...) → AppLogger.e(tag, msg)\n\n"

        "2. DEDUPLICATION RULE (CRITICAL):\n"
        "   Merge ALL consecutive AppLogger calls with EXACT SAME tag AND message into ONE.\n\n"

        "3. IMPORTS:\n"
        "   ADD: 'import com.honeywell.domain.managers.loggerApp.AppLogger'\n"
        "   REMOVE: android.util.Log, Logr, STLevelLog, AppSTLogger imports\n\n"

        "Return ONLY raw source code. NO markdown, NO explanations."
    )


def call_groq(code: str, api_key: str, chunk_index: Optional[int] = None, is_retry: bool = False) -> Optional[str]:
    """Appelle Groq avec retry automatique sur rate limit."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    label = f"chunk {chunk_index}" if chunk_index is not None else "fichier"

    payload = {
        "model": GROQ_MODEL,
        "max_tokens": MAX_TOKENS_OUT,
        "messages": [
            {"role": "system", "content": "Output raw Kotlin code only. Return every line. Never truncate."},
            {"role": "user", "content": build_prompt(is_retry) + "\n\nCODE:\n" + code}
        ],
        "temperature": 0
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", RATE_LIMIT_BASE_WAIT * (2 ** attempt)))
                print(f"    🚦 Rate limit {label} — attente {retry_after}s (tentative {attempt+1}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue

            if resp.status_code != 200:
                print(f"    ❌ Groq error {resp.status_code}: {resp.text[:150]}")
                return None

            choice = resp.json()['choices'][0]
            if choice.get("finish_reason") == "length":
                print(f"    ⚠️ {label}: finish_reason=length — réponse tronquée par max_tokens")

            content = choice["message"]["content"].strip()
            content = re.sub(r'^```(kotlin)?\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\s*```$', '', content)
            return content

        except requests.exceptions.Timeout:
            print(f"    ⏱️ Timeout {label} (tentative {attempt+1}/{MAX_RETRIES})")
            time.sleep(10)
        except Exception as e:
            print(f"    ❌ Erreur réseau {label}: {e}")
            return None

    print(f"    ❌ Max retries atteint sur {label}")
    return None


# ================= REFACTORING =================
def refactor_chunk_with_retry(chunk: str, api_key: str, chunk_index: int, total: int) -> Optional[str]:
    """Refactorise un chunk avec validation et retry si nécessaire."""
    print(f"    → Chunk {chunk_index}/{total} ({len(chunk):,} chars) → Groq...")

    result = call_groq(chunk, api_key, chunk_index=chunk_index)
    if result is None:
        return None

    # Chunks triviaux : pas de validation
    if len(chunk.strip()) < MIN_CHUNK_SIZE:
        print(f"    ✅ Chunk {chunk_index}/{total}: trivial — pas de validation")
        return result

    # Check troncature
    truncated = detect_truncation(result)
    ok, reason = check_chunk_validity(chunk, result, chunk_index)

    if truncated:
        print(f"    ⚠️ Chunk {chunk_index}/{total}: troncature détectée — retry...")
    elif not ok:
        print(f"    ⚠️ Chunk {chunk_index}/{total}: {reason} — retry...")

    if truncated or not ok:
        time.sleep(REQUEST_DELAY)
        result = call_groq(chunk, api_key, chunk_index=chunk_index, is_retry=True)
        if result is None:
            return None

        ok2, reason2 = check_chunk_validity(chunk, result, chunk_index)
        if ok2:
            print(f"    ✅ Chunk {chunk_index}/{total}: retry OK")
        else:
            print(f"    ⚠️ Chunk {chunk_index}/{total}: retry partiel ({reason2}) — on continue")

    return result


def refactor_file(filepath: str, api_key: str) -> bool:
    """Refactorise un fichier Kotlin. Retourne True si succès."""
    print(f"\n🤖 Traitement de {filepath}...")

    # Lecture
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except Exception as e:
        print(f"  ❌ Erreur lecture: {e}")
        return False

    # Skip si pas de logs
    if not any(x in original_code for x in ["Log.", "Logr.", "AppSTLogger."]):
        print(f"  ⏭️  Skipped (pas de logs)")
        return False

    print(f"  📄 Taille: {len(original_code):,} chars")

    # Refactoring avec ou sans chunking
    try:
        if len(original_code) > MAX_FILE_SIZE:
            # Gros fichier → chunking
            chunks = split_code(original_code)
            total = len(chunks)
            print(f"  📦 Fichier découpé en {total} chunks de ~{CHUNK_SIZE} chars")

            refactored_chunks = []
            for i, chunk in enumerate(chunks):
                result = refactor_chunk_with_retry(chunk, api_key, i + 1, total)
                if result is None:
                    print(f"  ❌ Échec chunk {i+1}/{total} — fichier ignoré")
                    return False
                refactored_chunks.append(result)

                # Pause entre chunks
                if i < total - 1:
                    time.sleep(REQUEST_DELAY)

            new_code = "\n".join(refactored_chunks)

        else:
            # Petit fichier → entier
            print(f"  → Fichier complet → Groq...")
            new_code = call_groq(original_code, api_key)
            if new_code is None:
                print(f"  ❌ Échec Groq")
                return False

            if detect_truncation(new_code):
                print(f"  ⚠️ Troncature détectée — retry...")
                time.sleep(REQUEST_DELAY)
                new_code = call_groq(original_code, api_key, is_retry=True)
                if new_code is None:
                    print(f"  ❌ Échec retry")
                    return False

        # Filet de sécurité Python : doublons inter-chunks
        new_code, dedup_count = deduplicate_applogger(new_code)
        if dedup_count:
            print(f"  🔁 Filet Python: {dedup_count} doublon(s) AppLogger inter-chunks supprimé(s)")

        # Validation finale
        valid, reason = validate_final(original_code, new_code)
        if not valid:
            print(f"  ❌ Validation finale échouée: {reason}")
            return False

        # Écriture
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_code)

        print(f"  ✅ Refactorisé avec succès")
        return True

    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False


# ================= MAIN =================
def main():
    api_key  = os.getenv('GROQ_API_KEY')
    base_ref = os.getenv('BASE_REF', 'main')

    if not api_key:
        print("❌ Erreur: GROQ_API_KEY manquant dans les variables d'environnement")
        sys.exit(1)

    print(f"🚀 Refactor Agent - Démarrage")
    print(f"🤖 Modèle: {GROQ_MODEL}")
    print(f"📦 Chunking: {CHUNK_SIZE} chars (seuil: {MAX_FILE_SIZE})")
    print(f"🔁 Déduplication: LLM + Python safety net")

    files = get_changed_files(base_ref)

    if not files:
        print("\n📭 Aucun fichier .kt à traiter")
        sys.exit(0)

    print(f"\n📋 {len(files)} fichiers à refactoriser:")
    for f in files:
        print(f"  - {f}")

    # Traitement
    success_count = 0
    skip_count = 0
    failed_count = 0

    for filepath in files:
        result = refactor_file(filepath, api_key)
        if result is True:
            success_count += 1
        elif result is False and "Skipped" in open(filepath, 'r').read():
            skip_count += 1
        else:
            failed_count += 1

        # Pause entre fichiers
        time.sleep(REQUEST_DELAY)

    # Résumé
    print(f"\n" + "="*60)
    print(f"✅ Succès:  {success_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print(f"❌ Échecs:  {failed_count}")
    print(f"📊 Total:   {len(files)}")
    print("="*60)

    if failed_count > 0:
        sys.exit(1)
    else:
        print("\n🎉 Refactoring terminé avec succès")
        sys.exit(0)


if __name__ == "__main__":
    main()
