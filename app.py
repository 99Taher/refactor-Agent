name: AI Logger Refactor avec Notifications

on:
  push:
    branches:
      - test
      - develop
      - feature/**
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  refactor:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write

    steps:
      - name: 📥 Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: 🔍 Détecter fichiers Kotlin modifiés
        id: detect_files
        run: |
          if [ "${{ github.event_name }}" == "pull_request" ]; then
            BASE_REF="${{ github.event.pull_request.base.ref }}"
            HEAD_REF="${{ github.event.pull_request.head.ref }}"
            echo "base_ref=$BASE_REF" >> $GITHUB_OUTPUT
            echo "head_ref=$HEAD_REF" >> $GITHUB_OUTPUT
            echo "📍 PR: Base=$BASE_REF, Head=$HEAD_REF"
            git fetch origin "$BASE_REF"
            CHANGED_FILES=$(git diff --name-only origin/$BASE_REF...HEAD | grep '\.kt$' || true)
          else
            HEAD_REF="${{ github.ref_name }}"
            echo "base_ref=HEAD^" >> $GITHUB_OUTPUT
            echo "head_ref=$HEAD_REF" >> $GITHUB_OUTPUT
            echo "📍 Push sur: $HEAD_REF"
            CHANGED_FILES=$(git diff --name-only HEAD^ HEAD | grep '\.kt$' || true)
          fi

          if [ -z "$CHANGED_FILES" ]; then
            echo "kt_count=0" >> $GITHUB_OUTPUT
            echo "✅ Aucun fichier Kotlin modifié"
          else
            KT_COUNT=$(echo "$CHANGED_FILES" | wc -l | tr -d ' ')
            echo "kt_count=$KT_COUNT" >> $GITHUB_OUTPUT
            echo "📝 $KT_COUNT fichiers Kotlin modifiés:"
            echo "$CHANGED_FILES"
            echo "$CHANGED_FILES" > /tmp/changed_files.txt
          fi

      - name: 🔥 Réveiller le service Render
        if: steps.detect_files.outputs.kt_count > 0
        run: |
          echo "🌐 Vérification du service Render..."
          HEALTH_CODE=$(curl -s -w "%{http_code}" -o /dev/null --max-time 15 \
            https://refactor-agent.onrender.com/health || echo "000")

          if [ "$HEALTH_CODE" == "200" ]; then
            echo "✅ Service déjà actif"
          else
            echo "🔄 Service en hibernation, réveil en cours..."
            for i in 1 2 3; do
              echo "⏰ Tentative $i/3 (attente 15s)..."
              sleep 15
              HEALTH_CODE=$(curl -s -w "%{http_code}" -o /dev/null --max-time 20 \
                https://refactor-agent.onrender.com/health || echo "000")
              if [ "$HEALTH_CODE" == "200" ]; then
                echo "✅ Service réveillé!"
                break
              fi
            done
          fi

      - name: 🤖 Appeler Refactor Agent API
        id: refactor
        if: steps.detect_files.outputs.kt_count > 0
        run: |
          START_TIME=$(date +%s)

          if [ "${{ github.event_name }}" == "pull_request" ]; then
            API_BASE_REF="${{ steps.detect_files.outputs.base_ref }}"
          else
            API_BASE_REF="feature/centralized-logging-system"
          fi

          echo "🚀 Envoi requête de refactoring..."
          echo "📍 Base: $API_BASE_REF, Branch: ${{ steps.detect_files.outputs.head_ref }}"

          HTTP_CODE=$(curl -s -w "%{http_code}" -o /tmp/response.json \
            --max-time 900 \
            --connect-timeout 60 \
            --retry 3 \
            --retry-delay 30 \
            --retry-max-time 1200 \
            -X POST "https://refactor-agent.onrender.com/refactor" \
            -H "Content-Type: application/json" \
            -H "X-API-Key: ${{ secrets.REFACTOR_API_SECRET }}" \
            -d "{
              \"repo_url\": \"${{ github.event.repository.clone_url }}\",
              \"base_ref\": \"$API_BASE_REF\",
              \"branch\": \"${{ steps.detect_files.outputs.head_ref }}\"
            }")

          END_TIME=$(date +%s)
          DURATION=$((END_TIME - START_TIME))

          echo "http_code=$HTTP_CODE" >> $GITHUB_OUTPUT
          echo "duration=$DURATION" >> $GITHUB_OUTPUT
          echo "📊 Code HTTP: $HTTP_CODE"
          echo "⏱️  Durée: ${DURATION}s"

          if [ -f /tmp/response.json ]; then
            echo "📄 Réponse API:"
            cat /tmp/response.json | jq '.' 2>/dev/null || cat /tmp/response.json
          fi

          if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
            echo "✅ Refactoring terminé avec succès"
            echo "success=true" >> $GITHUB_OUTPUT

            # ⭐ FIX: Vérifier processed_files avant de parser
            HAS_FILES=$(cat /tmp/response.json | jq 'has("processed_files")' 2>/dev/null || echo "false")

            if [ "$HAS_FILES" == "true" ]; then
              PROCESSED=$(cat /tmp/response.json | jq -r '.processed_files[]' 2>/dev/null || echo "")
              echo "$PROCESSED" > /tmp/processed_files.txt

              if [ -n "$PROCESSED" ]; then
                SUCCESS_COUNT=$(echo "$PROCESSED" | grep -c "refactored" 2>/dev/null || echo "0")
                FAILED_COUNT=$(echo "$PROCESSED" | grep -cE "rate limit|error|trop gros" 2>/dev/null || echo "0")
              else
                SUCCESS_COUNT=0
                FAILED_COUNT=0
              fi
            else
              API_MSG=$(cat /tmp/response.json | jq -r '.message // .status // "terminé"' 2>/dev/null || echo "terminé")
              echo "ℹ️  Message: $API_MSG"
              echo "" > /tmp/processed_files.txt
              SUCCESS_COUNT=0
              FAILED_COUNT=0
            fi

            # ⭐ FIX: Valeurs toujours définies
            echo "success_count=$SUCCESS_COUNT" >> $GITHUB_OUTPUT
            echo "failed_count=$FAILED_COUNT" >> $GITHUB_OUTPUT
          else
            echo "❌ Échec du refactoring (HTTP $HTTP_CODE)"
            echo "success=false" >> $GITHUB_OUTPUT
            echo "success_count=0" >> $GITHUB_OUTPUT
            echo "failed_count=0" >> $GITHUB_OUTPUT

            if [ "$HTTP_CODE" == "502" ] || [ "$HTTP_CODE" == "503" ]; then
              echo "⚠️  Service en cours de démarrage - Attendre 2 min et relancer"
            fi
            exit 1
          fi

      - name: 💬 Commenter sur la PR - Succès
        if: success() && github.event_name == 'pull_request' && steps.refactor.outputs.success == 'true'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');

            let processedFiles = '';
            try {
              processedFiles = fs.readFileSync('/tmp/processed_files.txt', 'utf8');
            } catch (e) {
              processedFiles = '';
            }

            const successCount = parseInt('${{ steps.refactor.outputs.success_count }}') || 0;
            const failedCount = parseInt('${{ steps.refactor.outputs.failed_count }}') || 0;
            const totalCount = parseInt('${{ steps.detect_files.outputs.kt_count }}') || 0;
            const duration = parseInt('${{ steps.refactor.outputs.duration }}') || 0;
            const successRate = totalCount > 0 ? Math.round((successCount / totalCount) * 100) : 0;

            let statusEmoji = successRate >= 80 ? '✅' : successRate >= 50 ? '⚠️' : '❌';

            const lines = processedFiles.split('\n').filter(l => l.trim());
            let resultsTable = '';
            if (lines.length > 0) {
              resultsTable = '**📝 Détail des fichiers:**\n\n| Fichier | Status |\n|---------|--------|\n';
              lines.forEach(line => {
                const match = line.match(/(.+?) → (.+)/);
                if (match) {
                  const shortFile = match[1].split('/').slice(-2).join('/');
                  const status = match[2];
                  let emoji = status.includes('refactored') ? '✅' :
                               status.includes('rate limit') ? '⏳' :
                               status.includes('trop gros') ? '📦' : '❌';
                  resultsTable += '| `' + shortFile + '` | ' + emoji + ' ' + status + ' |\n';
                }
              });
              resultsTable += '\n';
            } else {
              resultsTable = '**ℹ️  Aucun fichier .kt modifié détecté par l\'API**\n\n';
            }

            const comment = '## ' + statusEmoji + ' Refactoring Automatique - Résultat\n\n' +
              '**📊 Statistiques:**\n' +
              '- ✅ Succès: **' + successCount + '/' + totalCount + '** fichiers (' + successRate + '%)\n' +
              '- ❌ Échecs: **' + failedCount + '** fichiers\n' +
              '- ⏱️ Durée: **' + duration + 's**\n' +
              '- 🤖 Modèle: `llama-3.3-70b-versatile`\n\n' +
              resultsTable +
              '**🔗 Liens:**\n' +
              '- [Logs du workflow](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }})\n\n' +
              '---\n*🤖 Généré automatiquement par Refactor Agent v6*';

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

      - name: 💬 Commenter sur la PR - Échec
        if: failure() && github.event_name == 'pull_request' && steps.detect_files.outputs.kt_count > 0
        uses: actions/github-script@v7
        with:
          script: |
            const httpCode = '${{ steps.refactor.outputs.http_code }}' || 'N/A';
            const duration = '${{ steps.refactor.outputs.duration }}' || 'N/A';

            let errorMessage = 'Erreur inconnue';
            if (httpCode === '403') errorMessage = '🔐 API_SECRET incorrect';
            else if (httpCode === '429') errorMessage = '⏳ Rate limit Groq - Attendre 1-2h';
            else if (httpCode === '500') errorMessage = '🔧 Erreur serveur - Vérifier logs Render';
            else if (httpCode === '502' || httpCode === '503') errorMessage = '🔄 Service Render en hibernation';
            else if (httpCode === '28' || httpCode === '000') errorMessage = '⏱️ Timeout connexion';

            const comment = '## ❌ Refactoring Automatique - Échec\n\n' +
              '- 🔴 Code: `' + httpCode + '`\n' +
              '- ⏱️ Durée: `' + duration + 's`\n' +
              '- 🔍 Erreur: ' + errorMessage + '\n\n' +
              '[Voir les logs](https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }})\n\n' +
              '---\n*🤖 Refactor Agent v6*';

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });

      - name: 📊 GitHub Actions Summary
        if: always() && steps.detect_files.outputs.kt_count > 0
        run: |
          echo "# 🤖 Refactor Agent - Résumé" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          if [ "${{ steps.refactor.outputs.success }}" == "true" ]; then
            echo "## ✅ Succès" >> $GITHUB_STEP_SUMMARY
            echo "- **Fichiers:** ${{ steps.refactor.outputs.success_count }}/${{ steps.detect_files.outputs.kt_count }}" >> $GITHUB_STEP_SUMMARY
            echo "- **Durée:** ${{ steps.refactor.outputs.duration }}s" >> $GITHUB_STEP_SUMMARY
          else
            echo "## ❌ Échec" >> $GITHUB_STEP_SUMMARY
            echo "- **HTTP:** ${{ steps.refactor.outputs.http_code }}" >> $GITHUB_STEP_SUMMARY
          fi
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "*$(date -u '+%Y-%m-%d %H:%M:%S UTC')*" >> $GITHUB_STEP_SUMMARY
