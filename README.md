# AI Adaptive Onboarding Engine

Two fully integrated layers in one project:

| Layer | What it is |
|---|---|
| **PyTorch model** | Multi-Task Siamese Transformer trained from scratch — resume–JD similarity + multi-label skill extraction |
| **Streamlit app** | Dark-themed web UI — upload PDFs, see match score, skill gaps, dependency-ordered roadmap, resume rewrite suggestions |

---

## Project Structure

```
ai_onboarding_engine/
├── data/
│   ├── skill_vocab.py        # 128-skill vocabulary + aliases + dependency graph
│   └── dataset.py            # Synthetic dataset pipeline
├── models/
│   ├── transformer.py        # Siamese Transformer from scratch (MHSA + FFN + mean pool)
│   └── losses.py             # CosineEmbeddingLoss + BCEWithLogitsLoss + metrics
├── training/
│   ├── train.py              # Multi-GPU fp16 training loop + checkpointing
│   ├── inference.py          # OnboardingEngine high-level wrapper
│   └── kaggle_train.ipynb    # Ready-to-run on Kaggle 2x T4 GPUs
├── utils/
│   ├── tokenizer.py          # Word-level tokenizer (from scratch)
│   ├── text_utils.py         # PDF extraction + text cleaning
│   └── roadmap.py            # BFS prerequisites + topological sort
├── app/
│   ├── engine.py             # Rule-based engine (80+ skills, alias map, roadmap)
│   └── streamlit_app.py      # Full Streamlit UI
├── requirements.txt
└── README.md
```

---

```bash
pip install streamlit pdfplumber
streamlit run app/streamlit_app.py
```
Runs in rule-based mode automatically when no checkpoint is present.

---
## Run with Trained Model

```bash
mkdir checkpoints
cp best_model.pt checkpoints/
cp tokenizer.json checkpoints/
streamlit run app/streamlit_app.py
```

---

## Architecture

```
Resume ──┐                        ┌── Similarity head  → score [0,1]
         ├── SharedTextEncoder ───┤
JD     ──┘  (shared weights)      └── Skill head × 2  → 128-dim logits

Encoder: Embedding → Sinusoidal PE → 3x Pre-LN Transformer → Mean pool → (B,256)
Params:  ~6.5M  |  hidden=256  |  layers=3  |  heads=4  |  max_len=256
```

## Loss
```
L = CosineEmbeddingLoss  +  0.5 × BCEWithLogitsLoss (resume + JD skill heads)
```
