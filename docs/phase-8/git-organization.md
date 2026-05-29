# Phase 8: Git Repository Organization

**Generated:** 2026-02-15

## Current State

- **Default Branch:** `main`
- **Active Feature Branches:** 8 total
- **Remote:** origin/main

## Proposed Branch Strategy

### Branch Structure

```
main (protected)
├── Production-ready code
├── Merged via PR only
└── Must pass CI checks

dev (default)
├── Integration branch
├── Feature branches merge here
└── Regular merges to main

feature/* (short-lived)
├── Individual features
├── Branch from dev
└── Merge to dev via PR (optional for solo developer)
```

### Branch Naming Convention

| Pattern | Use Case | Example |
|---------|----------|---------|
| `main` | Production releases | - |
| `dev` | Development integration | - |
| `feature/*` | New features | `feature/performer-similarity` |
| `fix/*` | Bug fixes | `fix/vision-analysis-timeout` |
| `cleanup/*` | Refactoring | `cleanup/consolidate-utils` |
| `docs/*` | Documentation only | `docs/update-readme` |

## Implementation Steps

### 1. Create Dev Branch

```bash
# Create dev from main
git checkout main
git pull origin main
git checkout -b dev
git push -u origin dev
```

### 2. Set Dev as Default (GitHub)

```bash
# Using gh CLI
gh repo edit --default-branch dev
```

**Note:** This requires admin access to the repository.

### 3. Add Branch Protection to Main

```bash
# Using gh CLI (requires admin access)
gh api repos/{owner}/{repo}/branches/main/protection \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  -f required_pull_request_reviews.required_approving_review_count=0 \
  -f required_pull_request_reviews.dismiss_stale_reviews=false \
  -f enforce_admins=false \
  -f required_status_checks.strict=true \
  -f required_status_checks.contexts[]=type-check \
  -f required_status_checks.contexts[]=lint \
  -f restrictions=null
```

**Simpler alternative for solo developer:**

```bash
# Minimal protection - require PR but no approvals
gh api repos/{owner}/{repo}/branches/main/protection \
  -X PUT \
  -H "Accept: application/vnd.github+json" \
  -F "required_pull_request_reviews[require_code_owner_reviews]=false" \
  -F "enforce_admins=false" \
  -F "required_status_checks=null" \
  -F "restrictions=null"
```

## CLAUDE.md Updates

### Add Branch Development Section

```markdown
## Development Workflow

### Branch Strategy

This repository uses a two-branch development model:

- **main**: Production-ready code (protected)
- **dev**: Development integration branch (default)

### Creating Feature Branches

1. Always branch from `dev`:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/my-feature
   ```

2. Work on your feature with frequent commits

3. Merge to `dev` when ready:
   ```bash
   git checkout dev
   git merge feature/my-feature
   git push origin dev
   ```

4. Periodically merge `dev` to `main` for releases:
   ```bash
   git checkout main
   git merge dev
   git push origin main
   ```

### Pull Requests (Optional)

Since this is a single-developer project, PRs are optional but recommended for:
- Major features
- Breaking changes
- When you want AI code review

### Branch Cleanup

After merging feature branches, delete them:
```bash
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```
```

## Cleanup Recommendations

### Stale Remote Branches

The following remote branches may be candidates for cleanup:

| Branch | Last Commit | Status |
|--------|-------------|--------|
| `claude/add-embedding-recommendations-F2dOq` | Unknown | Review |
| `claude/review-todo-list-3QFo1` | Unknown | Review |
| `claude/setup-docker-github-actions-oBr7E` | Unknown | Review |
| `claude/show-model-reasoning-4KVBn` | Unknown | Review |
| `add-claude-github-actions-1767238637175` | Unknown | Review |

**Command to delete merged branches:**
```bash
# List merged branches
git branch -r --merged origin/main | grep -v "main\|dev\|HEAD"

# Delete specific branch
git push origin --delete branch-name
```

### Local Branches to Merge or Delete

| Branch | Status | Recommendation |
|--------|--------|----------------|
| `cleanup/dead-code-removal` | Partial | Complete or merge |
| `cleanup/productionalize-2026-02-13` | Active | This branch |
| `feat/tag-gap-detection` | Unknown | Review |
| `feature/multi-stage-vision-analysis` | Unknown | Review |

## Verification

After completing Phase 8:

1. [ ] `dev` branch exists
2. [ ] `dev` is default branch on GitHub
3. [ ] `main` has branch protection
4. [ ] CLAUDE.md updated with branch workflow
5. [ ] Stale branches reviewed/cleaned

## Notes

- This is a single-developer repository
- PR approvals not required but recommended for major changes
- CI checks should still run on PRs to main
