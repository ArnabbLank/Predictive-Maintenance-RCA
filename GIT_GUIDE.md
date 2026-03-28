# Git Guide for This Project

> A practical, step-by-step guide for using Git in this RUL Prediction + Ops Copilot project.

---

## Table of Contents
1. [Initial Setup](#1-initial-setup)
2. [Daily Workflow](#2-daily-workflow)
3. [Branching Strategy](#3-branching-strategy)
4. [Commit Message Convention](#4-commit-message-convention)
5. [Working with Branches](#5-working-with-branches)
6. [Handling Merge Conflicts](#6-handling-merge-conflicts)
7. [Undoing Mistakes](#7-undoing-mistakes)
8. [Working with Remote (GitHub)](#8-working-with-remote-github)
9. [Pull Requests](#9-pull-requests)
10. [Git Best Practices](#10-git-best-practices)
11. [Useful Commands Cheat Sheet](#11-useful-commands-cheat-sheet)
12. [Large Files & Data](#12-large-files--data)

---

## 1. Initial Setup

### First-time Git configuration
```bash
# Set your identity (use your real name and university email)
git config --global user.name "Your Name"
git config --global user.email "your.email@university.edu"

# Set default branch name
git config --global init.defaultBranch main

# Set VS Code as the default editor
git config --global core.editor "code --wait"

# Enable colored output
git config --global color.ui auto
```

### Clone the repo
```bash
git clone <repo-url> rul-copilot
cd rul-copilot
```

### Or initialize from scratch
```bash
cd rul-copilot
git init
git add .
git commit -m "feat: initial project scaffold"
git remote add origin <repo-url>
git push -u origin main
```

---

## 2. Daily Workflow

The basic cycle you'll repeat every day:

```bash
# 1. Pull latest changes (if working with others)
git pull origin main

# 2. Create a branch for your work (see naming below)
git checkout -b week01/eda-notebook

# 3. Do your work... edit files, run notebooks, etc.

# 4. Check what changed
git status
git diff

# 5. Stage your changes
git add week01-eda/01_eda.ipynb
git add src/data_loader.py
# OR stage everything:
git add .

# 6. Commit with a meaningful message
git commit -m "feat(week01): add EDA notebook with sensor distributions"

# 7. Push your branch
git push origin week01/eda-notebook

# 8. When done, merge into main (via PR or locally)
git checkout main
git merge week01/eda-notebook
git push origin main
```

---

## 3. Branching Strategy

### Branch naming convention
```
<week>/<short-description>

Examples:
  week01/eda-notebook
  week02/baseline-models
  week04/lstm-model
  week06/transformer-uq
  week09/copilot-agent
  week11/streamlit-app
  fix/data-loader-bug
  docs/update-readme
```

### Branch hierarchy
```
main (stable, always working)
 ├── week01/eda-notebook
 ├── week02/baseline-models
 ├── week04/lstm-model
 ├── ...
 └── week12/final-report
```

### Rules
- **`main`** = always working, clean code. Never commit broken code here.
- **Week branches** = your working branches. Merge into `main` when a week is complete.
- **Fix branches** = for bug fixes. Prefix with `fix/`.
- **Doc branches** = for documentation. Prefix with `docs/`.

---

## 4. Commit Message Convention

We follow **Conventional Commits** (simplified):

```
<type>(<scope>): <short description>

Types:
  feat     — new feature or functionality
  fix      — bug fix
  data     — data processing changes
  model    — model architecture/training changes
  exp      — experiment results
  docs     — documentation only
  refactor — code restructuring (no new features)
  test     — adding or fixing tests
  chore    — maintenance (deps, configs)

Scope (optional):
  week01, week02, ..., src, copilot, app, etc.

Examples:
  feat(week01): add sensor trend plots to EDA notebook
  model(week06): implement CNN-Transformer with MC dropout
  fix(src): correct RUL capping logic in preprocess.py
  docs: update README with baseline results table
  exp(week11): add ablation study — SG vs raw inputs
  chore: update requirements.txt with streamlit
```

---

## 5. Working with Branches

### Create and switch to a new branch
```bash
git checkout -b week02/baselines
```

### Switch between branches
```bash
git checkout main
git checkout week02/baselines
```

### List all branches
```bash
git branch          # local branches
git branch -a       # including remote branches
```

### Delete a branch (after merging)
```bash
git branch -d week01/eda-notebook    # safe delete (checks if merged)
git branch -D week01/eda-notebook    # force delete
```

### Merge a branch into main
```bash
git checkout main
git merge week02/baselines
# If no conflicts, push:
git push origin main
```

---

## 6. Handling Merge Conflicts

Conflicts happen when two branches edit the same lines. Don't panic!

```bash
# 1. Git will tell you which files have conflicts
git status

# 2. Open the conflicting file — you'll see markers:
<<<<<<< HEAD
your changes on main
=======
changes from the branch
>>>>>>> week02/baselines

# 3. Edit the file to keep what you want (remove the markers)

# 4. Stage the resolved file
git add <resolved-file>

# 5. Complete the merge
git commit -m "fix: resolve merge conflict in data_loader.py"
```

### Tips to avoid conflicts
- Pull frequently: `git pull origin main`
- Keep branches short-lived (merge weekly)
- Don't edit the same files on multiple branches

---

## 7. Undoing Mistakes

### Oops, I haven't committed yet
```bash
# Discard all unstaged changes
git checkout -- .

# Discard changes to a specific file
git checkout -- src/data_loader.py

# Unstage a file (keep changes, just un-add)
git reset HEAD src/data_loader.py
```

### Oops, I committed but haven't pushed
```bash
# Undo the last commit, keep changes staged
git reset --soft HEAD~1

# Undo the last commit, keep changes unstaged
git reset HEAD~1

# Undo the last commit, discard changes (DANGEROUS)
git reset --hard HEAD~1
```

### Oops, I pushed
```bash
# Create a new commit that undoes the bad one (safe)
git revert <commit-hash>
git push
```

### I want to save my work temporarily
```bash
git stash                    # save current changes
git stash list               # see saved stashes
git stash pop                # restore the most recent stash
git stash drop               # delete the most recent stash
```

---

## 8. Working with Remote (GitHub)

### Connect to a remote repository
```bash
git remote add origin https://github.com/your-username/rul-copilot.git
git push -u origin main
```

### Push / Pull
```bash
git push origin main               # push main to GitHub
git push origin week03/health-index # push a branch
git pull origin main                # get latest from GitHub
```

### Check remote info
```bash
git remote -v
```

---

## 9. Pull Requests

When your weekly branch is ready:

1. Push the branch: `git push origin week05/paper-a`
2. Go to GitHub → your repo → "Compare & pull request"
3. Fill in:
   - **Title**: `Week 5: Paper A — CNN+LSTM replication`
   - **Description**: What you did, results, any issues
4. Request a review from your mentor
5. After approval, merge via GitHub (use "Squash and merge" for clean history)
6. Delete the branch on GitHub after merging
7. Locally: `git checkout main && git pull origin main`

---

## 10. Git Best Practices

### DO ✅
- Commit early and often (small, focused commits)
- Write meaningful commit messages
- Pull before you push
- Use `.gitignore` to keep the repo clean
- Review your changes with `git diff` before committing
- Keep `main` always in a working state

### DON'T ❌
- Don't commit large data files (use `.gitignore`)
- Don't commit API keys or secrets (use `.env` + `.gitignore`)
- Don't commit notebook outputs (clear outputs or use `nbstripout`)
- Don't force push to `main`
- Don't leave uncommitted changes for days

### Notebook-specific tips
```bash
# Install nbstripout to auto-clear notebook outputs before commit
pip install nbstripout
nbstripout --install            # sets up a git filter
nbstripout --install --attributes .gitattributes
```

---

## 11. Useful Commands Cheat Sheet

| Command | Description |
|---------|-------------|
| `git status` | See current state |
| `git log --oneline -10` | Last 10 commits (compact) |
| `git log --oneline --graph --all` | Visual branch history |
| `git diff` | See unstaged changes |
| `git diff --staged` | See staged changes |
| `git blame <file>` | See who changed each line |
| `git show <commit>` | See a specific commit |
| `git stash` | Temporarily save changes |
| `git tag v1.0 -m "Week 12 final"` | Tag a release |
| `git clean -fd` | Remove untracked files (careful!) |

---

## 12. Large Files & Data

### Why we don't commit data to Git
- Git is designed for code, not large binary files
- The CMAPSS dataset (~12 MB zipped) is manageable, but we still exclude it
- Model checkpoints (`.pt`, `.pth`) can be very large

### Our approach
- Data files are in `.gitignore`
- A download script (`data/download_data.sh`) recreates the data folder
- Model checkpoints go in `reports/` or a separate storage (not in git)

### If you need Git LFS (for very large files)
```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.zip"

# Make sure .gitattributes is tracked
git add .gitattributes
```

---

## Quick Reference Card

```
┌─────────────────── Git Workflow ───────────────────┐
│                                                     │
│  1. git pull origin main                            │
│  2. git checkout -b week<N>/<description>           │
│  3. ... do your work ...                            │
│  4. git add .                                       │
│  5. git commit -m "type(scope): message"            │
│  6. git push origin week<N>/<description>           │
│  7. Open Pull Request on GitHub                     │
│  8. After merge: git checkout main && git pull      │
│                                                     │
└─────────────────────────────────────────────────────┘
```
