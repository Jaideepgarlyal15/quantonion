# GitHub Setup and Deployment Guide

This guide covers how to set up Git version control, connect to GitHub, and deploy the application to Streamlit Cloud.

---

## Initial Git Setup

If you have not already initialised Git in your repository, run the following commands:

```bash
# Initialise git repository
git init

# Add all files to staging
git add .

# Commit with initial message
git commit -m "Initial commit"

# Add remote repository (replace with your actual repository URL)
git remote add origin https://github.com/yourusername/Regime-Switching-Risk-Dashboard-Streamlit-HMM-.git

# Push to GitHub
git push -u origin main
```

If your default branch is named "master" instead of "main", rename it first:

```bash
git branch -M main
git push -u origin main
```

---

## GitHub Authentication

### Option A: Personal Access Token (Recommended)

1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Generate a new token with the "repo" scope
3. Copy the generated token
4. When pushing to GitHub, use the token as your password

### Option B: GitHub CLI

```bash
# Install GitHub CLI
brew install gh

# Authenticate with GitHub
gh auth login

# Set default repository
gh repo set-default
```

---

## Regular Git Operations

### Push Changes to GitHub

```bash
# Stage all changes
git add .

# Commit with a message
git commit -m "Description of changes"

# Push to GitHub
git push
```

### Pull Latest Changes

```bash
git pull
```

### Check Repository Status

```bash
git status
```

---

## Visual Studio Code Integration

Visual Studio Code has built-in Git support:

1. Open the Source Control tab (Ctrl+Shift+G on Linux/Windows, Cmd+Shift+G on Mac)
2. Review changed files
3. Stage changes by clicking the plus icon
4. Enter a commit message and press Ctrl+Enter (Cmd+Enter on Mac)
5. Sync changes by clicking the sync icon

---

## Deploying to Streamlit Cloud

1. Push your code to a public GitHub repository
2. Navigate to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository from the list
6. Choose the branch and specify the main file path (app.py)
7. Click "Deploy"

Your application will be available at:
`https://yourusername-regime-switching-risk-dashboard.streamlit.app`

---

## Common Git Commands Reference

| Command | Description |
|---------|-------------|
| `git status` | Show modified and staged files |
| `git add filename` | Stage a specific file |
| `git add .` | Stage all modified files |
| `git commit -m "message"` | Commit staged changes |
| `git push` | Push committed changes to remote |
| `git pull` | Fetch and merge remote changes |
| `git log` | Show commit history |
| `git diff` | Show unstaged changes |

---

## Creating a New Repository

If you need to create a fresh repository:

1. Create a new repository on GitHub.com (without initialising with README)
2. Clone the empty repository to your local machine
3. Copy the project files into the cloned directory
4. Commit and push:

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

