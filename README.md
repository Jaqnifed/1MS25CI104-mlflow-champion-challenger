# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI104 &nbsp;·&nbsp; Skill Lab (MLOps with MLflow)

## 1. Task and models

| | |
|---|---|
| Task | Binary sentiment classification (positive / negative) |
| Dataset | SST-2 validation split (`stanfordnlp/sst2`), movie-review sentences with human labels |
| Subset size | 200 sentences, sampled with `random_state=42` (saved in `data/eval_inputs.csv`) |
| Model A (current Champion) | `distilbert-base-uncased-finetuned-sst-2-english` (~67M params) |
| Model B (Challenger) | `siebert/sentiment-roberta-large-english` (~355M params) |

Both models output POSITIVE/NEGATIVE, which `hf_wrapper.py` maps to `positive`/`negative`.
Both are wrapped in a custom `mlflow.pyfunc.PythonModel` (as in Lab 9), so any logged
version can be loaded from MLflow and used without writing Hugging Face code.

## 2. How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# terminal 1: tracking server (leave it running)
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# terminal 2
python evaluate.py --n 10          # quick debug run
python evaluate.py                 # final run, 200 examples
python register_champion.py        # registers A and B, sets champion/challenger aliases
python predict_from_mlflow.py      # loads models:/sentiment-model@champion, predicts on data/new_data.csv

# stretch goal: swap the aliases and rerun the SAME script without editing it
python register_champion.py --flip
python predict_from_mlflow.py
```

| File | What it does |
|---|---|
| `config.py` | USN, model IDs, dataset, seed, and the Champion rule constants |
| `hf_wrapper.py` | pyfunc wrapper around the Hugging Face pipeline + label mapping |
| `evaluate.py` | runs both models on the same 200 inputs, logs runs, metrics, artifacts, models and the decision |
| `register_champion.py` | finds the latest `compare_sentiment` run by name (no hard-coded run IDs), registers both models, sets aliases |
| `predict_from_mlflow.py` | loads the Champion by alias only; never imports transformers |

## 3. Champion rule (written before the final run)

1. **Primary metric:** macro-F1 on the 200-sentence evaluation set. **Higher is better.**
2. **Minimum improvement margin:** the Challenger replaces the Champion only if its macro-F1 is
   at least **1.0 percentage point** higher (delta ≥ 0.01). With 200 examples one sentence is
   0.5 pp, so a 1 pp margin means the Challenger must get at least two more sentences right;
   a smaller gap could be noise from which sentences happened to be sampled.
3. **Guardrail:** reject the Challenger if its average latency is more than **10×** the
   Champion's, even if it is more accurate. Model B is about 5× bigger, so some slowdown is
   expected and acceptable; 10× would be too slow for interactive use.
4. **Tie-breaker:** if the difference is inside the margin, the Champion stays, because it is
   smaller and faster.

Direction of every metric: accuracy, macro-F1, macro-precision, macro-recall → higher is better.
`avg_latency_ms`, `total_inference_s`, `latency_ratio_b_over_a` → lower is better.

## 4. Results

*(fill in from `outputs/metrics_summary.json` after the final run)*

| Metric | A: distilbert-sst2 | B: roberta-large | B − A |
|---|---|---|---|
| Accuracy | | | |
| Macro-F1 (primary) | | | |
| Macro-precision | | | |
| Macro-recall | | | |
| Avg latency (ms/example) | | | |
| Total inference (s) | | | |

Plots: `outputs/compare_metrics.png`, `outputs/compare_latency.png`,
`outputs/confusion_matrix_*.png`. Disagreements: `outputs/disagreements.csv`.

### Qualitative examples (5 cases where the models disagree)

| # | Sentence | Gold | A | B | Comment |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

## 5. Decision

*(150–250 words, written after the final run. Answer: does the Challenger improve on the
Champion, by how much, is it meaningful for 200 samples, and what are the trade-offs in speed,
size and errors.)*

## 6. Limitations and declaration

- 200 sentences is a small sample; differences of 1–2 pp could change with a different seed.
- SST-2 sentences are short movie-review snippets; results may not carry over to long reviews,
  tweets or Indian-English text.
- Model A was fine-tuned on the SST-2 *training* split, so it is evaluated on in-domain data;
  Model B was trained on a mix of 15 sentiment datasets. This favours Model A slightly.
- Latency was measured on my laptop CPU, batch size 16; numbers will differ on other machines.
- **AI assistance:** I used Claude (an AI assistant) to help draft the scripts and this README
  structure. I ran every script myself, checked the outputs, and can explain each line.
