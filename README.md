# Explainable AI-Based Phishing Email Detection and Risk Scoring System

Phishing e-postalarını tespit eden, risk skoru hesaplayan ve SHAP ile açıklanabilirlik sağlayan bir sistem.

## Kurulum

```bash
pip install -r requirements.txt
```

## Proje Yapısı

```
├── phishing_ai/       # ML paketi (config, data, features, models, risk, explain)
├── api/               # FastAPI backend
├── web/               # Tek sayfalık UI
├── tests/             # Unit testler
├── scripts/           # train.py, download_dataset.py
└── models/            # Kaydedilen modeller
```

## Kullanım

### 1. Dataset indir ve model eğit

```bash
python scripts/download_dataset.py
python scripts/train.py
```

### 2. API başlat

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Web arayüzü

`web/index.html` dosyasını tarayıcıda açın veya bir static server ile sunun.

## API

- `GET /health` — Sağlık kontrolü
- `POST /analyze_email` — E-posta analizi (JSON body: `{"text": "..."}`)

## Test

```bash
pytest tests/ -v
```
