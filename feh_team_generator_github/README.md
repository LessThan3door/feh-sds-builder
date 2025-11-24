# FEH Team Generator (Hosted-ready)

This repository contains a production-ready FastAPI backend and a clean frontend for the FEH Team Generator.

**Original uploaded script (reference):** `/mnt/data/FEH SDS Autobuilder.py`

## One-click Deploy to Render

[![Deploy to Render](https://deploy.render.com/buttons/new-button.svg)](https://dashboard.render.com/deploy?repo={REPO_URL})

> After you push this repo to GitHub, replace `{REPO_URL}` above with your repository URL to enable the Deploy button.

## Quick start locally

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
# open http://127.0.0.1:8000/frontend/index.html
```

## Docker

```bash
docker build -t feh_team_generator .
docker run -p 8000:8000 feh_team_generator
```

---
