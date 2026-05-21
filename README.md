# [Project Title]

## Overview
Brief description of the research project.

## Repository Structure

| Folder | Purpose | Who uses it |
|---|---|---|
| `paper/` | LaTeX source files | Overleaf sync |
| `data/raw/` | Original, unmodified data | Read-only |
| `data/processed/` | Cleaned/transformed data | Scripts |
| `code/` | Analysis scripts | Claude Code / terminal |
| `outputs/` | Figures, tables, results | Scripts |

## Setup

### 1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

### 2. Install dependencies
# Python example:
pip install -r requirements.txt

### 3. Set up API keys (when needed)
Create a `.env` file in the root (never commit this):
MY_API_KEY=your-key-here

Load it in Python with:
from dotenv import load_dotenv
load_dotenv()

### 4. Overleaf sync
- In Overleaf: Menu → GitHub → link to this repo → select the `paper/` folder
- Push/pull between Overleaf and GitHub using the Overleaf menu

## Collaboration Workflow

git pull                     # always pull before starting work
# ... make changes ...
git add .
git commit -m "clear description of what changed"
git push

## Coauthors
- [Name] — GitHub: @username
- [Name] — GitHub: @username
