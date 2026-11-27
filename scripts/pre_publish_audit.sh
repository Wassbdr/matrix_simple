#!/usr/bin/env bash
# pre_publish_audit.sh - Quick audit before pushing the repo public.
# Run from repo root: bash scripts/pre_publish_audit.sh

set -uo pipefail

cd "$(dirname "$0")/.."
RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; RST=$'\033[0m'

fail=0
ok()   { echo "${GRN}OK${RST}   $*"; }
warn() { echo "${YLW}WARN${RST} $*"; }
err()  { echo "${RED}FAIL${RST} $*"; fail=1; }

echo "=== Pre-publish audit ==="

# 1. License
[[ -f LICENSE ]] && ok "LICENSE present" || err "LICENSE missing"

# 2. Citation
[[ -f CITATION.cff ]] && ok "CITATION.cff present" || warn "CITATION.cff missing"

# 3. Security policy
[[ -f SECURITY.md ]] && ok "SECURITY.md present" || warn "SECURITY.md missing"

# 4. Secret-like strings in tracked files
echo ""
echo "--- Scanning tracked files for secret-like patterns ---"
hits=$(git grep -nIE '(api[_-]?key|secret[_-]?key|bearer\s+[a-z0-9]|ghp_[a-z0-9]+|sk-[a-z0-9]{20,}|aws_access_key|password\s*=\s*["\x27][^"\x27]+["\x27])' \
    -- ':!*.md' ':!.gitignore' ':!.env.example' 2>/dev/null | grep -vi 'placeholder\|example\|test_\|fake\|dummy' || true)
if [[ -n "$hits" ]]; then
  warn "Possible secrets - review:"
  echo "$hits" | head -20
else
  ok "No obvious hardcoded secrets"
fi

# 5. Sandbox-only credentials flagged in SECURITY.md ?
if grep -q 'mypassword' k8s/app/db.yaml k8s/app/backend.yaml 2>/dev/null; then
  if grep -q 'mypassword' SECURITY.md 2>/dev/null; then
    ok "Sandbox password documented in SECURITY.md"
  else
    err "k8s/app/{db,backend}.yaml contains 'mypassword' - undocumented"
  fi
fi

# 6. Files >500KB tracked
echo ""
echo "--- Tracked files larger than 500KB ---"
big=$(git ls-files | xargs -I{} du -k "{}" 2>/dev/null | awk '$1>500{print}')
if [[ -n "$big" ]]; then
  warn "Large files (consider Releases/Zenodo):"
  echo "$big"
else
  ok "No tracked file >500KB"
fi

# 7. Generated artifacts that should be gitignored
echo ""
echo "--- Suspect generated artifacts in git ---"
suspects=$(git ls-files | grep -E '\.(pt|pkl|h5|hdf5)$|graph_heatmap|train_new_checkpoint\.log|infer_new_checkpoint\.log' || true)
[[ -z "$suspects" ]] && ok "No generated artifacts tracked" \
                     || { warn "Untrack these:"; echo "$suspects"; }

# 8. Internal-jury docs flagged?
echo ""
echo "--- Internal docs scope ---"
# Internal docs in docs/internal/ are explicitly scoped (acceptable). Anywhere else is a leak.
leaked=$(git ls-files | grep -E 'rapport_stage|ACCOMPAGNEMENT|SOUTENANCE|Week[0-9]+\.md' \
         | grep -v '^docs/internal/' || true)
if [[ -n "$leaked" ]]; then
  err "Internal docs outside docs/internal/:"
  echo "$leaked"
else
  scoped=$(git ls-files docs/internal/ 2>/dev/null | wc -l)
  ok "Internal docs all under docs/internal/ (${scoped} files)"
fi

# 9. TODOs/FIXMEs in code
todo_count=$(grep -rn 'TODO\|FIXME\|XXX\|HACK' rca_pipeline/ --include='*.py' 2>/dev/null | wc -l)
[[ $todo_count -eq 0 ]] && ok "No TODO/FIXME in rca_pipeline/" \
                        || warn "$todo_count TODO/FIXME markers in rca_pipeline/"

# 10. CI workflow present
[[ -f .github/workflows/ci.yml ]] && ok "CI workflow present" || err "CI workflow missing"

# 11. README badges link to placeholder?
if grep -q 'YOUR_USERNAME/YOUR_REPO' README.md 2>/dev/null; then
  warn "README still has YOUR_USERNAME/YOUR_REPO placeholders"
else
  ok "README placeholders replaced"
fi

# 12. Tests pass?
echo ""
echo "--- Test suite ---"
if command -v pytest >/dev/null 2>&1; then
  if python -m pytest -q rca_pipeline/tests --co >/dev/null 2>&1; then
    ok "pytest collection succeeds"
  else
    warn "pytest collection failed - run 'python -m pytest rca_pipeline/tests'"
  fi
else
  warn "pytest not installed"
fi

echo ""
if [[ $fail -ne 0 ]]; then
  echo "${RED}Audit failed - fix the FAIL items before publishing.${RST}"
  exit 1
fi
echo "${GRN}Audit passed (review WARN items).${RST}"
