# Explainable AI-Based Phishing Email Detection and Risk Scoring System

Phishing e-postalarını tespit eden, risk skoru hesaplayan ve hem risk breakdown hem de kelime bazlı pozitif/negatif göstergeler döndüren bir sistem.

## Kurulum

```bash
pip install -r requirements.txt
```

## Dataset

Bu proje Faz 1 için Kaggle `naserabdullahalam/phishing-email-dataset` içindeki `CEAS_08.csv` dosyasını kullanır.

Kaggle API kullanmak için:

1. Kaggle hesabından API token oluştur.
2. `kaggle.json` dosyasını `~/.kaggle/kaggle.json` altına koy.
3. Dosya iznini sınırla:

```bash
chmod 600 ~/.kaggle/kaggle.json
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

### 1. Dataset indir

```bash
python scripts/download_dataset.py
```

Bu komut `data/raw/CEAS_08.csv` dosyasını hazırlar.

### 2. Faz 1 modeli eğit

```bash
python scripts/train.py
```

### 3. API başlat

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Web arayüzü

API ile birlikte sunulan arayüz için `http://127.0.0.1:8000/web/` adresini açın.

## API

- `GET /health` — Sağlık kontrolü
- `POST /analyze_email` — E-posta analizi (JSON body: `{"text": "..."}`)

Örnek response alanları:

- `prediction`
- `probability`
- `risk_score`
- `risk_level`
- `risk_components`
- `top_indicators_pos`
- `top_indicators_neg`

## Faz 2: Opsiyonel embedding deneyi

Faz 2 için `sentence-transformers` ile hibrit özellik uzayı kullanılabilir:

- TF-IDF + security features + embeddings
- Logistic Regression hibrit modeli
- çekirdek Faz 1 modeline karşı performans kıyası

Eğitmek için:

```bash
python scripts/train.py --phase2
```

Daha hızlı deney (örneklem ile, önerilen):

```bash
python scripts/train.py --phase2-only --phase2-max-rows 8000
```

API’de hibrit modeli seçmek için:

- `POST /analyze_email?model=hybrid`

## Test

```bash
pytest tests/ -v
```
