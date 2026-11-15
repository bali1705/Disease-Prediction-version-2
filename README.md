# MedSight — Multiple Disease Prediction System

A production-ready Flask web app for **preliminary, educational** multi-disease screening. It combines:

- **Symptoms**: a 41-disease symptom classifier with adaptive follow-up questions.
- **Lab reports**: upload a PDF or photo, or paste text. Vitals and biomarkers are extracted automatically (Gemini vision, or a built-in PDF/text reader).
- **Both combined**: lab findings re-weight the symptom-based ranking, and trained diabetes and heart-disease models add risk estimates.

Findings are pinned to an **interactive anatomical body map**. Organ callouts are colour-coded (green = normal, amber = borderline, red = high risk), and clicking one opens plain-language explanations with range bars. A **context-aware AI assistant** (Google Gemini via the `google-genai` SDK) explains the results and asks one clarifying question at a time. It falls back to built-in guidance if no API key is set.

> **Medical disclaimer.** MedSight is for preliminary educational screening only. It does not provide a diagnosis and does not replace a qualified clinician. In an emergency, call your local emergency number.

---

## 1. Model upgrade: before vs. after

All numbers come from `python -m ml_pipeline.train` and are saved in `reports/backtest_results.json` and `reports/backtest_report.md`. The app also shows them on the **Model accuracy** page.

**Protocol.** Legacy and upgraded models are scored on **identical outer stratified 5-fold splits**. Upgraded models are tuned with **nested CV**: the model family and hyper-parameters are chosen using the training folds only. All preprocessing runs inside the scikit-learn / imbalanced-learn pipeline, so nothing is fitted on test data. That preprocessing covers zero-as-missing handling, median imputation with missing-indicators, scaling, one-hot encoding and SMOTE.

| Model | Evaluation | Accuracy before → after | Δ accuracy | F1 before → after | ROC-AUC before → after | Deployed model |
|---|---|---|---|---|---|---|
| Diabetes | Stratified 5-fold CV | 77.1% → 76.8% | **-0.26 pp** (-0.3%) | 63.3% → 63.1% | 83.2% → 83.7% | Logistic regression |
| Heart disease | Stratified 5-fold CV | 81.9% → 86.5% | **+4.61 pp** (+5.6%) | 79.0% → 83.9% | 88.8% → 91.8% | Logistic regression |
| Parkinson's disease | Person-grouped 5-fold CV (leakage-free) | 83.0% → 82.4% | **-0.62 pp** (-0.8%) | 89.2% → 89.1% | 83.7% → 81.5% | Logistic regression |
| Parkinson's disease | Random 5-fold CV (legacy protocol, leaks people across folds) | 85.6% → 93.8% | **+8.21 pp** (+9.6%) | 90.9% → 96.0% | 88.6% → 97.9% | Logistic regression |
| Symptom-based disease classifier (41 diseases) | All symptoms entered (unique profiles, 5-fold) | 79.4% → 97.9% | **+18.46 pp** (+23.2%) | 73.8% → 98.4% | 93.7% → 100.0% | Logistic regression |
| Symptom-based disease classifier (41 diseases) | Partial & shuffled symptoms (realistic input, 5-fold) | 10.4% → 91.8% | **+81.42 pp** (+783.1%) | 8.1% → 94.0% | 63.1% → 99.9% | Logistic regression |

Full per-metric tables (accuracy, precision, recall, F1 and ROC-AUC with absolute and relative change) are in [`reports/backtest_report.md`](reports/backtest_report.md).

**Reading these results honestly**

- **Heart disease and the symptom classifier improved substantially.** The symptom classifier reaches 98% on complete profiles, and 92% when users enter only some symptoms in any order, where the legacy model scored 10%.
- **Diabetes and Parkinson's are at parity.** On these small, well-studied datasets the legacy linear SVM was already near the achievable ceiling (~77% for Pima diabetes). The upgraded pipelines match it and add proper missing-value handling, so users can leave fields blank. They also produce calibrated probabilities for the risk gauges, with no leakage.
- **Parkinson's random-split number is misleading.** The +8 pp jump comes from leakage (the same person in train and test). The person-grouped row is the honest estimate.
- **An earlier selection rule overfit.** It always picked the top inner-CV model and chose complex ensembles that scored worse on these small datasets. Its results are kept in `reports/backtest_report_v1_selection.md`. The final rule picks the simplest model within one standard error of the best.

### Data problems found and fixed

| Issue in the legacy project | Fix |
|---|---|
| `heart.csv` (Kaggle copy) stores the label **inverted**: `target = 1` is the healthier group. The old app told healthy-looking profiles they had heart disease. | Label corrected to `disease = 1 − target`. Both models are scored on the corrected label. A unit test checks the direction. |
| Heart model ignored resting BP, resting ECG and vessel count, and learned the invalid codes `ca = 4` and `thal = 0`. | All 13 features are used; invalid codes are treated as missing and imputed. |
| Diabetes data uses `0` for missing glucose, BP, skin thickness, insulin and BMI. The SVM learned those zeros as real values. | Zeros become missing inside the pipeline, then imputed with missing-indicators. |
| Parkinson's data has ≈6 recordings per person, and random splits put the same voice in train and test (leakage). | Person-grouped CV (`StratifiedGroupKFold`). The leaky random-split number is shown only for comparison. |
| Symptom data: 94% of the 4,920 rows are duplicates, so any random split scores ≈100%. The legacy model encodes symptoms by **list position**, retrains a forest on every request, and only works for 3–17 symptoms entered in one fixed order. | Evaluated on unique profiles and on partial, shuffled symptom lists (how people actually enter them). The new model is order-independent, trained once, and robust to incomplete input. |

---

## 2. Project structure

```
app.py                     Flask app: routes, validation, rate limits, security headers
ml_pipeline/
  config.py                Feature specs shared by training, validation and UI
  transformers.py          Picklable preprocessing steps (zero→missing, code cleaning)
  legacy.py                Faithful reproductions of the original models (for backtesting)
  symptom_catalog.py       131 symptoms, categories, organ mapping, disease guidance
  train.py                 Backtest + nested-CV training + export (.joblib)
  predictor.py             Production inference + information-gain follow-up questions
services/
  biomarkers.py            40 vitals/lab markers: ranges, organ mapping, unit conversion
  report_parser.py         Report text parser + Gemini output normalisation
  gemini_client.py         google-genai wrapper (chat + structured report extraction)
  assessment.py            Fusion of symptoms + labs + ML risks, red flags, next steps
  assistant_offline.py     Built-in assistant used when Gemini is unavailable
  uploads.py               Type sniffing, size limits, guaranteed temp-file cleanup
  validation.py            Strict input validation for every endpoint
models/                    Exported pipelines (*.joblib) — ready for inference
data/                      Training datasets
reports/                   Backtest results (JSON + Markdown)
static/ templates/         Frontend (vanilla JS, no CDN scripts)
tests/test_app.py          End-to-end API tests (pytest)
Procfile  render.yaml  Dockerfile  requirements.txt  .python-version  .env.example
```

## 3. Run locally

Requires **Python 3.13**.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini key for local use (optional; the app works without it)
cp .env.example .env                # Windows: copy .env.example .env
#   then edit .env and set GEMINI_API_KEY=...

# 4. Start the app
python app.py                       # http://localhost:5000
```

Run the tests with `pip install -r requirements-dev.txt && pytest -q`.

Retrain and re-run the backtest (≈30 min on 2 CPUs) with `python -m ml_pipeline.train`. This overwrites `models/*.joblib` and `reports/*`.

## 4. Deploy publicly on Render (free tier works)

1. Push this folder to a **GitHub** repository using `git`. `.gitignore` keeps `.env` out of the repo. Don't drag-and-drop files in the GitHub web UI, because that bypasses `.gitignore`.
   ```bash
   git init && git add . && git commit -m "MedSight" && git branch -M main
   git remote add origin https://github.com/<you>/medsight.git && git push -u origin main
   ```
2. Go to <https://dashboard.render.com> and choose **New → Blueprint**. Pick the repo, and Render reads `render.yaml`. (Alternatively, choose **New → Web Service** and fill in the settings below.)
   - Runtime: **Python** (version taken from `.python-version` = 3.13)
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 2 --threads 4 --timeout 120 --preload --bind 0.0.0.0:$PORT`
   - Health check path: `/healthz`
3. In **Environment**, add `GEMINI_API_KEY` with your key. It is stored as a secret and never committed.
4. Click **Deploy**. When the status shows *Live*, your public URL is `https://<service-name>.onrender.com`.
5. Check it: open `https://<service-name>.onrender.com/healthz`. It should return `{"status":"ok","ai":true}`.

Notes:
- On the free plan, the service sleeps after ~15 min idle, so the first request then takes ~30–60 s.
- If you see out-of-memory restarts on the 512 MB plan, change the start command to `--workers 1 --threads 8`.
- Rate limits are per process, held in memory. For multi-instance deployments, set `RATELIMIT_STORAGE_URI=redis://...`.

### Railway
New Project → Deploy from GitHub repo. Railway uses the `Procfile` (or the `Dockerfile`). Add `GEMINI_API_KEY` under **Variables**, then **Settings → Networking → Generate Domain**.

### Hugging Face Spaces
Create a Space with **SDK: Docker**, then push this repo to it. Add this front-matter block at the very top of the Space's `README.md`:
```yaml
---
title: MedSight
sdk: docker
app_port: 7860
---
```
Add `GEMINI_API_KEY` under **Settings → Variables and secrets** (as a *secret*).

## 5. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Enables the Gemini assistant and photo/scan reading |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Preferred model; falls back to `gemini-2.5-flash`, then `gemini-flash-latest` |
| `PORT` | `5000` | Port to bind (set automatically by hosts) |
| `MAX_UPLOAD_MB` | `8` | Max report size |
| `RATELIMIT_STORAGE_URI` | `memory://` | Rate-limit store (`redis://…` for multiple instances) |
| `PROXY_HOPS` | `1` | Reverse proxies to trust for the client IP |

## 6. Security & privacy

- The API key is read only from the environment (`os.environ.get("GEMINI_API_KEY")`), and `.env` is git-ignored.
- **Rate limits per IP** (Flask-Limiter):
  - Report reading: 6/min and 60/day.
  - Chat: 12/min and 150/day.
  - Assessments: 40/min.
  - Global: 600/h.
- **Uploads**: extension whitelist, magic-byte sniffing, an 8 MB cap and Pillow verification with a decompression-bomb limit. Files go to a private random-named temp file that is deleted in a `finally` block, and a sweeper removes any stragglers.
- Strict server-side validation of every field. Content-Security-Policy, `X-Frame-Options`, `nosniff` and a referrer policy are set on every response.
- Nothing users enter is stored.

## 7. API

| Method | Path | Body |
|---|---|---|
| GET | `/api/meta` | – symptom catalogue, biomarker catalogue, model input specs |
| POST | `/api/assess` | `{mode, symptoms[], denied[], biomarkers{}, patient{}, clinical{}}` |
| POST | `/api/followup` | `{symptoms[], denied[], exclude[]}` |
| POST | `/api/report` | multipart `file`, or JSON `{text}` |
| POST | `/api/predict/<diabetes\|heart\|parkinsons>` | model features |
| POST | `/api/chat` | `{message, history[], assessment{}}` |
| GET | `/api/performance` | backtest comparison |
| GET | `/healthz` | – |
