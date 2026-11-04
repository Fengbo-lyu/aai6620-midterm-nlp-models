# AI-Use Summary

**Course:** AAI 6620 · **Author:** Fengbo Lyu · **Project:** Comparing Three NLP Models

This project was completed with the assistance of an AI coding assistant (Claude). In the spirit of the
assignment's responsible-AI requirement, this document states honestly where AI helped and what I did
myself.

## How AI helped

- **Scaffolding the project structure** — AI helped lay out the notebook sections (data → bigram →
  LSTM → transformer → comparison → reflection) and the shared, fair-comparison harness (`get_batch`,
  `estimate_loss`, `train_model`, `generate`).
- **Implementing the three models** — AI drafted the PyTorch code for the bigram lookup model, the
  LSTM model, and the small decoder-only transformer (self-attention head, multi-head attention,
  feed-forward block, positional embeddings), following the standard Karpathy "nanoGPT"-style design
  taught around this part of the course.
- **Building the data pipeline** — extracting a Shakespeare corpus from the course CSV into
  `data/input.txt`, the character tokenizer, and the 90/10 split.
- **Drafting the written analysis and this README/summary**, which I then checked against the actual
  numbers my run produced.

## What I did and verified myself

- **Ran and reproduced** the full notebook end-to-end and confirmed the reported losses, training
  times, and generated samples are the real outputs of my run (not invented).
- **Checked the analysis against the evidence.** The initial AI draft assumed the transformer would win;
  my actual run showed the **LSTM slightly outperformed the small transformer** at this scale, so I
  corrected the reflection and README to state the honest result and explain *why* (small data/model,
  limited training; attention's advantage appears with scale). This correction is the part I most want
  to flag as my own judgement.
- **Understood each component** — I can explain what the bigram ignores, how the LSTM hidden state
  carries context, and how masked self-attention lets every position attend to earlier ones, including
  the causal mask and positional embeddings.

## Tools

- AI assistant: Claude (code generation, drafting, debugging help)
- Libraries: PyTorch, pandas, matplotlib; Jupyter

*I take responsibility for the final submission; the explanations reflect my own understanding of the
material, and the reported results are reproducible from the included notebook and dataset.*
