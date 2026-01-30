# GitHub Setup & VSCode Connection Guide

## Step 1: Initialize Git & Push to GitHub

Run these commands in your terminal:

```bash
# Navigate to your project folder
cd /Users/jaideepgarlyal/Desktop/regime-switching-dashboard

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit with a message
git commit -m "Update: Multi-timeframe ML forecasts, remove premium references"

# Add your GitHub repository
git remote add origin https://github.com/Jaideepgarlyal15/Regime-Switching-Risk-Dashboard-Streamlit-HMM-.git

# Push to GitHub
git push -u origin main
```

**If you get an error about branches:**
```bash
# Check current branch name
git branch

# If it's "master" instead of "main":
git branch -M main
git push -u origin main
```

---

## Step 2: Connect GitHub to VSCode (Already Connected!)

Your VSCode is already connected to GitHub. Here's how to sync:

### Push Updates (Send changes to GitHub)
```bash
git add .
git commit -m "Your message here"
git push
```

### Pull Updates (Get latest from GitHub)
```bash
git pull
```

### Check Status
```bash
git status
```

---

## Step 3: Set Up GitHub Authentication

### Option A: GitHub Token (Recommended)
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → Select "repo" scope
3. Copy the token
4. When pushing, use token as password:
   - Username: your GitHub username
   - Password: paste the token

### Option B: GitHub CLI
```bash
# Install GitHub CLI
brew install gh

# Login
gh auth login

# Then push with:
gh repo set-default
```

---

## Step 4: VSCode Git Integration

VSCode has built-in Git support:

1. **Source Control tab** (Ctrl+Shift+G) shows changes
2. **Stage changes** by clicking +
3. **Commit** by typing message and pressing Ctrl+Enter
4. **Sync/Push** by clicking the sync icon

---

## Step 5: Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Deploy!

Your app URL will be: `https://yourusername-regime-switching-risk-dashboard.streamlit.app`

---

## Useful Git Commands

| Command | Description |
|---------|-------------|
| `git status` | See changed files |
| `git add filename` | Stage specific file |
| `git add .` | Stage all files |
| `git commit -m "msg"` | Commit changes |
| `git push` | Push to GitHub |
| `git pull` | Pull from GitHub |
| `git log` | See commit history |
| `git diff` | See file differences |

---

## If You Need to Create a New Repository

```bash
# Create repository on GitHub.com first
# Then run:
cd /Users/jaideepgarlyal/Desktop/regime-switching-dashboard
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

