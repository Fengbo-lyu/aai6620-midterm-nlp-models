# AI-Use Summary

Course: AAI 6620 · Author: Fengbo Lyu ·
Project: Comparing Three NLP Models

I used an AI coding assistant (Claude) on parts of this project. As the assignment's responsible-AI
requirement asks, this document states honestly what I did myself and where AI helped.

## What I did myself

- **Designed the experiment.** I chose the task (character-level language modelling), the Shakespeare
  dataset, and the three models to compare, and I set the hyperparameters — model sizes, learning rates,
  number of training steps, block size and batch size — to keep the comparison fair and small enough to
  run on a laptop.
- **Debugged and ran everything end-to-end** on my own machine, and reproduced the reported losses,
  training times, and generated samples, so the numbers in the notebook and README are my real outputs
  rather than invented ones.
- **Wrote the results analysis myself** — what the numbers mean, whether they matched what I expected,
  and why. The AI's first draft assumed the transformer would win; my actual run showed the LSTM
  slightly beat the small transformer at this scale, so I corrected the interpretation and explained the
  reason (small data and model, limited training; attention's advantage shows up with scale). That
  analysis and conclusion are my own.

## Where AI helped

- **Drafting the PyTorch code** for the three models (the bigram lookup table, the LSTM, and the small
  decoder-only transformer with self-attention, multi-head attention, a feed-forward block, and
  positional embeddings) in the standard nanoGPT-style design covered in the course, plus the shared
  comparison harness (`get_batch`, `estimate_loss`, `train_model`, `generate`).
- **Initial notebook scaffolding** and the script to extract the Shakespeare corpus from the
  course-provided `Shakespeare_data.csv` (concatenating the `PlayerLine` column) into `data/input.txt`.
- **Language polishing** of the written sections after I had decided what they should say.

## What I can explain

I can walk through each component: what the bigram ignores, how the LSTM hidden state carries context,
and how masked self-attention lets every position attend to earlier ones, including the causal mask and
positional embeddings. Choosing the settings, debugging the runs, and writing the analysis is where I
built that understanding.

## Tools

- AI assistant: Claude (code drafting, debugging help, language polishing)
- Libraries: PyTorch, pandas, matplotlib; Jupyter

*I take responsibility for the final submission. The explanations and conclusions are my own, and the
reported results are reproducible from the included notebook and dataset.*
