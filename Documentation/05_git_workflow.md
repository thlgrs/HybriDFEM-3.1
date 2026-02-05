# Git Workflow Guide

This guide explains the Git workflow for contributing to HybriDFEM. Following these practices ensures a clean, maintainable codebase.

## Overview

HybriDFEM uses a **feature branch workflow** with the following principles:

1. The `main` branch is always stable and deployable
2. All development happens on feature branches
3. Changes are merged via Pull Requests (PRs)
4. Code review is required before merging

## Quick Reference

```bash
# Start new feature
git checkout main
git pull origin main
git checkout -b feature/my-feature

# Work on feature
git add <files>
git commit -m "Add feature X"

# Keep up to date
git fetch origin main
git rebase origin/main

# Push and create PR
git push -u origin feature/my-feature
# Create PR via GitHub/GitLab UI

# After PR is merged
git checkout main
git pull origin main
git branch -d feature/my-feature
```

## Detailed Workflow

### 1. Setting Up Your Local Repository

```bash
# Clone the repository
git clone <repository-url> HybriDFEM
cd HybriDFEM

# Configure your identity
git config user.name "Your Name"
git config user.email "your.email@uclouvain.be"

# Verify remote
git remote -v
```

### 2. Creating a Feature Branch

Always create a new branch for your work:

```bash
# Ensure you're on main and up to date
git checkout main
git pull origin main

# Create and switch to feature branch
git checkout -b feature/descriptive-name

# Alternative: create from a specific commit
git checkout -b feature/my-feature abc1234
```

**Branch naming conventions:**

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/description` | `feature/mortar-coupling` |
| Bug fix | `fix/description` | `fix/stiffness-assembly` |
| Documentation | `docs/description` | `docs/api-reference` |
| Refactor | `refactor/description` | `refactor/solver-interface` |
| Test | `test/description` | `test/coupling-methods` |

### 3. Making Commits

**Commit early and often** with clear, descriptive messages:

```bash
# Stage specific files
git add Core/Objects/Coupling/Mortar.py
git add tests/test_coupling.py

# Or stage all changes in a directory
git add Core/Objects/Coupling/

# Commit with a clear message
git commit -m "Add mortar coupling for non-matching meshes

- Implement MortarCoupling class with Gauss integration
- Add interface detection for block-FEM boundaries
- Support horizontal and vertical interface orientations"
```

**Commit message guidelines:**

```
<type>: <short summary> (50 chars max)

<body - explain what and why, not how> (wrap at 72 chars)

<footer - references to issues, breaking changes>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```bash
# Good commit messages
git commit -m "feat: Add Quad9 element with 9-node Lagrangian formulation"
git commit -m "fix: Correct Jacobian sign for clockwise node ordering"
git commit -m "docs: Add coupling method comparison table"
git commit -m "test: Add convergence tests for hybrid coupling"

# Bad commit messages (avoid these)
git commit -m "fix bug"
git commit -m "WIP"
git commit -m "updates"
```

### 4. Keeping Your Branch Updated

Regularly sync with main to avoid large merge conflicts:

```bash
# Fetch latest changes (doesn't modify your files)
git fetch origin main

# Rebase your branch onto main
git rebase origin/main

# If conflicts occur:
# 1. Edit conflicted files to resolve
# 2. Stage resolved files
git add <resolved-files>
# 3. Continue rebase
git rebase --continue

# If you need to abort
git rebase --abort
```

**Rebase vs Merge:**

- **Rebase** (preferred): Creates a linear history, cleaner
- **Merge**: Preserves branch history, creates merge commits

```bash
# Rebase (preferred for feature branches)
git rebase origin/main

# Merge (alternative, but creates merge commits)
git merge origin/main
```

### 5. Pushing Your Branch

```bash
# First push (sets upstream)
git push -u origin feature/my-feature

# Subsequent pushes
git push

# After rebasing (force push required - BE CAREFUL)
git push --force-with-lease
```

**Important**: Only force-push your own feature branches, never `main`.

### 6. Creating a Pull Request

1. Push your branch to the remote
2. Go to the repository on GitHub/GitLab
3. Click "New Pull Request" or "Create Merge Request"
4. Select your branch as source, `main` as target
5. Fill in the PR template:

```markdown
## Description
Brief description of what this PR does.

## Changes
- Added X
- Modified Y
- Fixed Z

## Testing
How was this tested?
- [ ] Unit tests pass (`pytest`)
- [ ] Manual testing performed
- [ ] New tests added

## Screenshots (if applicable)
[Add images for UI changes]

## Related Issues
Fixes #123
```

### 7. Code Review Process

**As a PR author:**

1. Request review from team members
2. Respond to feedback promptly
3. Make requested changes as new commits (easier to review)
4. Once approved, squash commits if needed before merge

**As a reviewer:**

1. Check code quality and standards
2. Run tests locally if needed
3. Provide constructive feedback
4. Approve or request changes

### 8. Merging

After approval, merge your PR:

1. **Squash and merge** (preferred): Combines all commits into one
2. **Rebase and merge**: Keeps commits separate but linear
3. **Merge commit**: Creates a merge commit (avoid unless necessary)

```bash
# If merging locally (usually done via UI)
git checkout main
git pull origin main
git merge --squash feature/my-feature
git commit -m "feat: Add mortar coupling (#123)"
git push origin main
```

### 9. Cleaning Up

After your PR is merged:

```bash
# Switch to main and update
git checkout main
git pull origin main

# Delete local branch
git branch -d feature/my-feature

# Delete remote branch (usually done automatically by PR merge)
git push origin --delete feature/my-feature
```

## Common Scenarios

### Scenario 1: Working on Multiple Features

```bash
# Create first feature branch
git checkout -b feature/feature-a
# ... work on feature A ...
git commit -m "Add feature A"
git push -u origin feature/feature-a

# Switch to work on feature B
git checkout main
git pull origin main
git checkout -b feature/feature-b
# ... work on feature B ...
```

### Scenario 2: Undoing Commits

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes
git reset --hard HEAD~1

# Revert a specific commit (creates new commit)
git revert abc1234
```

### Scenario 3: Stashing Work

```bash
# Save work temporarily
git stash

# Switch branches, do other work
git checkout main

# Come back and restore
git checkout feature/my-feature
git stash pop
```

### Scenario 4: Resolving Merge Conflicts

```bash
# During rebase, if conflicts occur
# 1. Open conflicted files, look for conflict markers:
# <<<<<<< HEAD
# your changes
# =======
# incoming changes
# >>>>>>> branch-name

# 2. Edit to resolve (remove markers, keep correct code)

# 3. Stage resolved files
git add <resolved-file>

# 4. Continue
git rebase --continue
```

### Scenario 5: Updating a PR After Feedback

```bash
# Make changes based on review feedback
git add <modified-files>
git commit -m "Address review: clarify variable naming"

# Push update to PR
git push

# If you rebased
git push --force-with-lease
```

## Best Practices

### DO:

- **Commit often**: Small, focused commits are easier to review and revert
- **Write clear messages**: Future you will thank present you
- **Keep branches short-lived**: Merge within days, not weeks
- **Pull before push**: Always sync with remote before pushing
- **Use `.gitignore`**: Don't commit generated files, secrets, or IDE configs

### DON'T:

- **Don't commit to main directly**: Always use feature branches
- **Don't force-push shared branches**: Only force-push your own branches
- **Don't commit secrets**: API keys, passwords, credentials
- **Don't commit large binaries**: Use Git LFS if needed
- **Don't rewrite published history**: After pushing, commits are "published"

## .gitignore

HybriDFEM's `.gitignore` should include:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build
dist/
build/
*.egg-info/

# Results (generated files)
Examples/*/Results/
*.h5
*.png
*.pdf

# OS
.DS_Store
Thumbs.db
```

## Troubleshooting

### "Permission denied" when pushing

```bash
# Check your SSH key is added
ssh -T git@github.com

# Or use HTTPS with credential manager
git config --global credential.helper cache
```

### "Your branch is behind"

```bash
git fetch origin main
git rebase origin/main
# Resolve any conflicts
git push --force-with-lease
```

### Accidentally committed to main

```bash
# Move commit to a new branch
git branch feature/my-accidental-work
git reset --hard origin/main
git checkout feature/my-accidental-work
```

### Need to change last commit message

```bash
# Only if not pushed yet
git commit --amend -m "New message"

# If already pushed (changes history!)
git commit --amend -m "New message"
git push --force-with-lease
```

---

*Previous: [Development Guide](04_development_guide.md) | Next: [Examples Guide](06_examples_guide.md)*
