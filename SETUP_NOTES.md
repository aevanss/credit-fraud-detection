# Setup Notes

Notes from initial project setup, kept for future reference (new machine, onboarding
a collaborator, or just remembering why something is configured the way it is).

## Environment

- **Python version used:** [e.g. 3.10.6, from `/Library/Frameworks/Python.framework/...`]
- **Why this interpreter specifically:** [note the conda `base` auto-activation issue —
  what it was, how you noticed it, how you fixed it]
- **Venv creation:** `python3 -m venv venv`
- **Activation:** `source venv/bin/activate`

## Known issues & fixes

### Issue: [short title, e.g. "Kaggle CLI stuck on old version"]
- **Symptom:** [what error/behavior you saw — paste the key error line, not the whole traceback]
- **Cause:** [root cause once diagnosed — e.g. version/Python constraint]
- **Fix:** [exact command(s) that resolved it]
- **Why it matters going forward:** [e.g. "kaggle>=1.8.0 requires Python 3.11+; this
  project is pinned to 3.10.6, so kaggle is pinned to 1.7.4.5 in requirements.txt"]

### Issue: [second one — e.g. Kaggle auth / legacy credentials]
- **Symptom:**
- **Cause:**
- **Fix:**
- **Why it matters going forward:**

## Credentials

- **Kaggle auth method used:** [legacy `kaggle.json`, not the newer token-only method —
  note why]
- **Location:** `~/.kaggle/kaggle.json` (not tracked in git, not stored in the repo)
- **Permissions:** `chmod 600` applied — [why]

## Reproducing this environment from scratch

```bash
git clone <repo-url>
cd credit_fraud
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Kaggle credentials required separately — see "Credentials" above
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```