import os
import re
import time
import requests
import subprocess


def get_changed_files(base_ref):
    """Récupère les fichiers .kt modifiés avec fallback two-dot si three-dot échoue."""
    try:
        # Essai 1 : three-dot (merge base)
        cmd = f"git diff --name-only origin/{base_ref}...HEAD"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        files = [f for f in output.splitlines() if f.endswith('.kt') and os.path.exists(f)]
        if files:
            print(f"📝 {len(files)} fichiers trouvés (three-dot)")
            return files
    except Exception:
        pass

    try:
        # Essai 2 : two-dot (pas besoin de merge base)
        cmd = f"git diff --name-only origin/{base_ref}..HEAD"
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')
        files = [f for f in output.splitlines() if f.endswith('.kt') and os.path.exists(f)]
        print(f"📝 {len(files)} fichiers trouvés (two-dot)")
        return files
    except Exception as e:
        print(f"⚠️ Erreur Git Diff: {e}")
        return []


def call_groq_with_retry(code, api_key, max_retries=3):
    """
    Appelle Groq avec retry automatique sur rate limit (429).
    Retourne le code refactorisé ou None en cas d'échec.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = (
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

        "2. DEDUPLICATION RULE (CRITICAL - 100% LLM):\n"
        "   Merge ALL consecutive AppLogger calls with EXACT SAME tag AND message into ONE.\n"
        "   Example: Multiple 'AppLogger.e(TAG, \"error\")' → ONE 'AppLogger.e(TAG, \"error\")'\n\n"

        "3. IMPORTS:\n"
        "   ADD: 'import com.honeywell.domain.managers.loggerApp.AppLogger'\n"
        "   REMOVE: android.util.Log, Logr, STLevelLog, AppSTLogger imports\n\n"

        "Return ONLY raw source code. NO markdown, NO explanations."
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 32000,
        "messages": [
            {"role": "system", "content": "Output raw Kotlin code only. Return every line. Never truncate."},
            {"role": "user", "content": f"{prompt}\n\nCODE:\n{code}"}
        ],
        "temperature": 0
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)

            if response.status_code == 429:
                # Rate limit : attendre le délai suggéré ou backoff exponentiel
                retry_after = int(response.headers.get("retry-after", 15 * (2 ** attempt)))
                print(f"  🚦 Rate limit — attente {retry_after}s (tentative {attempt+1}/{max_retries})")
                time.sleep(retry_after)
                continue

            if response.status_code != 200:
                print(f"  ❌ Groq error {response.status_code}: {response.text[:150]}")
                return None

            new_code = response.json()['choices'][0]['message']['content'].strip()
            # Nettoyage des balises markdown
            new_code = re.sub(r'^```(kotlin)?\s*', '', new_code, flags=re.MULTILINE)
            new_code = re.sub(r'\s*```$', '', new_code)
            return new_code

        except requests.exceptions.Timeout:
            print(f"  ⏱️ Timeout (tentative {attempt+1}/{max_retries})")
            time.sleep(10)
        except Exception as e:
            print(f"  ❌ Erreur réseau: {e}")
            return None

    print(f"  ❌ Max retries atteint après {max_retries} tentatives")
    return None


def refactor(filepath, api_key):
    """Refactorise un fichier Kotlin avec déduplication 100% LLM."""
    print(f"🤖 Traitement de {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except Exception as e:
        print(f"  ❌ Erreur lecture: {e}")
        return

    # Skip si pas de logs
    if not any(x in original_code for x in ["Log.", "Logr.", "AppSTLogger."]):
        print(f"  ⏭️  Skipped (pas de logs)")
        return

    # Appel Groq avec retry
    new_code = call_groq_with_retry(original_code, api_key)

    if new_code is None:
        print(f"  ❌ Échec du refactoring")
        return

    # Validation basique : le code refactorisé ne doit pas être vide ou trop court
    if len(new_code) < len(original_code) * 0.5:
        print(f"  ⚠️ Code refactorisé trop court ({len(new_code)} vs {len(original_code)} chars) — ignoré")
        return

    # Écriture du fichier refactorisé
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_code)
        print(f"  ✅ Refactorisé (100% LLM)")
    except Exception as e:
        print(f"  ❌ Erreur écriture: {e}")


if __name__ == "__main__":
    api_key  = os.getenv('GROQ_API_KEY')
    base_ref = os.getenv('BASE_REF', 'main')

    if not api_key:
        print("❌ Erreur: GROQ_API_KEY manquant dans les variables d'environnement")
        exit(1)

    print(f"🔍 Détection des fichiers modifiés (base={base_ref})...")
    files = get_changed_files(base_ref)

    if not files:
        print("📭 Aucun fichier .kt à traiter")
        exit(0)

    print(f"\n📦 {len(files)} fichiers à refactoriser:\n" + "\n".join(f"  - {f}" for f in files))
    print()

    for filepath in files:
        refactor(filepath, api_key)
        time.sleep(2)  # Pause entre fichiers pour éviter le rate limit

    print("\n✅ Refactoring terminé")
