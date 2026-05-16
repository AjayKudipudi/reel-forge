#!/usr/bin/env bash
# Push this repo to a personal GitHub account WITHOUT touching global git config.
#
# Edit the variables in the CONFIG block, then run:
#     bash scripts/push_to_github.sh
#
# After it completes:
#   1. Verify the repo appears at https://github.com/<GH_USERNAME>/<REPO_NAME>
#   2. Revoke the PAT at https://github.com/settings/tokens (security hygiene
#      — the token was used inline and lives in your shell history)
#   3. Delete this script if you don't need to re-run it
set -euo pipefail

# ─── CONFIG ───────────────────────────────────────────────────────────────
GH_USERNAME="AjayKudipudi"
GH_EMAIL="venkatajay903@gmail.com"
COMMIT_AUTHOR_NAME="Venkat Ajay K"
REPO_NAME="reel-forge"
REPO_DESCRIPTION="AI dance video generator for Instagram Reels — SteadyDancer-14B + RIFE + GFPGAN on AWS spot GPUs"
REPO_VISIBILITY="public"   # "public" or "private"
DEFAULT_BRANCH="main"

# PAT — REPLACE THIS with your fresh token after rotating the one you pasted earlier.
# Paste a new one here right before running, then delete this line / rotate again
# after push completes.
GITHUB_TOKEN="${GITHUB_TOKEN:-ghp_REPLACE_ME}"

# ─── DERIVED ──────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/8] Repo root: $REPO_ROOT"

# Sanity-check the token is set
if [[ "$GITHUB_TOKEN" == *"REPLACE_ME"* ]] || [[ -z "$GITHUB_TOKEN" ]]; then
    echo "ERROR: edit GITHUB_TOKEN at the top of this script (or export GITHUB_TOKEN=...)" >&2
    exit 1
fi

# ─── 2. Verify .env is NOT going to be committed ──────────────────────────
echo "[2/8] Verifying .env is ignored..."
if [[ -f .env ]]; then
    if ! grep -qE '^\.env$|^\.env\b' .gitignore; then
        echo "ERROR: .env exists but is not in .gitignore. Aborting to prevent secret leak." >&2
        exit 1
    fi
    echo "      OK — .env exists locally but is gitignored"
fi

# ─── 3. Create the GitHub repo via API (idempotent) ───────────────────────
echo "[3/8] Creating GitHub repo $GH_USERNAME/$REPO_NAME (visibility=$REPO_VISIBILITY)..."
PRIVATE_FLAG=$([[ "$REPO_VISIBILITY" == "private" ]] && echo "true" || echo "false")
HTTP_CODE=$(curl -sS -o /tmp/gh_create.json -w "%{http_code}" \
    -X POST "https://api.github.com/user/repos" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -d "{\"name\":\"$REPO_NAME\",\"description\":\"$REPO_DESCRIPTION\",\"private\":$PRIVATE_FLAG,\"auto_init\":false}")
case "$HTTP_CODE" in
    201) echo "      OK — repo created" ;;
    422) echo "      OK — repo already exists, continuing" ;;
    401) echo "ERROR: GitHub returned 401 Unauthorized. Token is invalid or revoked." >&2; cat /tmp/gh_create.json >&2; exit 1 ;;
    *)   echo "ERROR: GitHub API returned HTTP $HTTP_CODE"; cat /tmp/gh_create.json >&2; exit 1 ;;
esac
rm -f /tmp/gh_create.json

# ─── 4. Initialize the local git repo (if not already) ────────────────────
echo "[4/8] Initializing git..."
if [[ ! -d .git ]]; then
    git init -b "$DEFAULT_BRANCH"
else
    echo "      already a git repo — using existing"
    git checkout -B "$DEFAULT_BRANCH"
fi

# ─── 5. Set repo-LOCAL identity (does NOT touch ~/.gitconfig) ─────────────
echo "[5/8] Setting repo-local commit identity..."
git config --local user.name "$COMMIT_AUTHOR_NAME"
git config --local user.email "$GH_EMAIL"
echo "      committing as: $(git config --local user.name) <$(git config --local user.email)>"

# Final guard — confirm .env is not in the index
if git ls-files --others --cached | grep -qE '^\.env$'; then
    echo "ERROR: .env is tracked. Aborting." >&2
    exit 1
fi

# ─── 6. Stage + commit ────────────────────────────────────────────────────
echo "[6/8] Staging files..."
git add -A
git status --short | head -30
echo

if git diff --cached --quiet; then
    echo "      nothing to commit — skipping"
else
    git commit -m "Initial public release

AI dance video generator for Instagram Reels using SteadyDancer-14B
(image-to-video), Practical-RIFE (frame interpolation), GFPGAN (face
restoration), and AWS spot GPUs. See docs/ for architecture, settings
audit, findings, and bug history."
fi

# ─── 7. Set remote with PAT embedded (cleaned up in step 8) ───────────────
echo "[7/8] Pushing to GitHub..."
REMOTE_URL_WITH_TOKEN="https://$GH_USERNAME:$GITHUB_TOKEN@github.com/$GH_USERNAME/$REPO_NAME.git"
REMOTE_URL_CLEAN="https://github.com/$GH_USERNAME/$REPO_NAME.git"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL_WITH_TOKEN"
else
    git remote add origin "$REMOTE_URL_WITH_TOKEN"
fi

git push -u origin "$DEFAULT_BRANCH"

# ─── 8. Scrub the token from .git/config ──────────────────────────────────
echo "[8/8] Scrubbing token from .git/config..."
git remote set-url origin "$REMOTE_URL_CLEAN"

echo
echo "Done — pushed to https://github.com/$GH_USERNAME/$REPO_NAME"
echo "Repo-local git config (does not affect global):"
git config --local --list | grep -E '^user\.|^remote\.' || true
echo
echo "Global git config (untouched):"
git config --global --list | grep -E '^user\.' || echo "  (no global user config — fine)"
echo
echo "REMINDER: revoke the PAT at https://github.com/settings/tokens now."
