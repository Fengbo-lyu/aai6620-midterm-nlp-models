# Comparing Three NLP Models — Bigram vs LSTM vs Transformer

Course: AAI 6620 Applied Natural Language Processing
Author: Fengbo Lyu
Midterm Project — individual submission**

A character-level language-modelling study: given previous text, predict the next character. I train
and compare three architectures on the same dataset and the same task to see how the choice of
architecture changes what a model can learn, and what self-attention actually adds.

---

## What NLP task did I solve?

Character-level next-token prediction (autoregressive language modelling). The input is a window of
previous characters; the output is a probability distribution over the next character. Models are
scored by cross-entropy loss (lower is better) and judged qualitatively by the text they generate.

## What dataset did I use?

A public-domain Shakespeare corpus (`data/input.txt`, 249,959 characters, 69-character vocabulary),
derived from the course-provided `Shakespeare_data.csv` (the per-line "Shakespeare plays" dataset):
the `PlayerLine` column is concatenated into plain text. A deterministic 90/10 train/validation split
is shared by all three models. The dataset file is included in this repo, so the project is fully
reproducible.

## What three models did I compare, and how does each work?

| Model | How it works | Context it uses |
|-------|--------------|-----------------|
| Bigram | A `(vocab × vocab)` lookup table of next-char logits | only the current character |
| LSTM | Embedding → LSTM (recurrent hidden state) → linear head | a hidden state summarising the past |
| Transformer | Token + positional embeddings → masked multi-head self-attention + feed-forward blocks → output head | every earlier position, directly, via attention |

All three share the same tokenizer, data split, batching, loss, and generation code (see the notebook)
so the comparison is fair.

## Results (this run)

| Model | Parameters | Train Loss | Val Loss | Train Time (s) |
|-------|-----------:|-----------:|---------:|---------------:|
| Bigram | 4,761 | 2.459 | 2.474 | 3.4 |
| LSTM | 242,661 | 1.434 | 1.644 | 120.0 |
| Transformer | 242,757 | 1.658 | 1.762 | 117.1 |

*(Char-level cross-entropy; numbers come from the executed notebook and may vary slightly on re-run.)*

Generated text samples for each model are printed in the notebook (Section 5). The bigram produces
plausible letter pairs but no real words; the LSTM and transformer both produce word-like,
locally-grammatical Shakespeare pastiche.

## Which model performed best? What did attention improve?

On this small corpus the LSTM reached the lowest validation loss, just ahead of the small transformer,
and both were far below the bigram. The honest finding is that at this scale (~225K training
characters, 2 transformer layers, 1,500 steps) the transformer's advantage had not appeared yet:
self-attention's benefit (direct, parallel access to all earlier tokens, with no recurrent bottleneck)
grows with more data, depth, and training. The lesson of the project is that transformers are powerful
at scale, not automatically at every size.

## What did I learn?

Moving from bigram to LSTM to transformer is a move from no memory, to compressed sequential memory, to
direct attention over the whole context. Attention removes the RNN information bottleneck and is
parallelisable, which is why it scales to large language models; but a tiny transformer does not
automatically beat a tuned LSTM on a small dataset. See `Midterm_Project.ipynb` Section 6 for the full
reflection.

## How did I use Claude/ChatGPT responsibly?

See [`ai_use_summary.md`](ai_use_summary.md) for the breakdown of what AI assisted with and what I
reviewed, ran, and verified myself.

---

## Repository contents

```
.
├── Midterm_Project.ipynb   # full notebook: data, 3 models, training, comparison, reflection
├── data/
│   └── input.txt           # Shakespeare corpus (character level)
├── README.md
└── ai_use_summary.md       # responsible AI-use documentation
```

## How to run

```bash
# Python 3.12, with: torch, pandas, matplotlib
pip install torch pandas matplotlib
jupyter nbconvert --to notebook --execute --inplace Midterm_Project.ipynb
# or open Midterm_Project.ipynb in Jupyter / Colab and "Run all"
```

Everything is small enough to train on a laptop CPU or free Colab in a few minutes.
