# English → Hinglish Neural Machine Translation (LSTM Encoder–Decoder)

A sequence-to-sequence model that translates English sentences into **Hinglish** (Hindi written in the Latin script, mixed with English) using an LSTM-based encoder–decoder trained with teacher forcing. Built with TensorFlow/Keras and evaluated with BLEU.

> **Use case:** helping language-technology products support multilingual communication for users who naturally write in Hinglish.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Overview

This project implements a baseline neural machine translation (NMT) system:

1. Load and preprocess paired English / Hinglish sentences
2. Tokenize both languages, build vocabularies, and pad sequences
3. Design and implement an LSTM encoder–decoder model
4. Train with teacher forcing and evaluate translation quality using BLEU

## Dataset

**English–Hinglish Machine Translation** from Kaggle: [`yash9439/eng-hinglish-machine-translation`](https://www.kaggle.com/datasets/yash9439/eng-hinglish-machine-translation)

The dataset is downloaded automatically with `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("yash9439/eng-hinglish-machine-translation")
```

The notebook auto-detects the English and Hinglish columns. If detection picks the wrong columns for your copy of the data, set `en_col` and `hi_col` manually in the loading cell.

> You may need to configure Kaggle credentials (`~/.kaggle/kaggle.json`) or run in Kaggle/Colab, where access is simpler.

## Project Pipeline

| Step | Description |
|------|-------------|
| **Preprocessing** | Lowercasing, punctuation separation, removal of odd symbols, deduplication, length filtering (≤ 25 words), optional subsampling |
| **Tokenization** | Separate word-level Keras tokenizers for English and Hinglish (20k vocab cap, `<unk>` for out-of-vocabulary words) |
| **Target tokens** | `<start>` and `<end>` markers added to Hinglish sentences |
| **Padding** | Post-padding to the 99th-percentile sequence length |
| **Teacher forcing** | Decoder input = target[:-1], decoder output = target[1:] |

## Model Architecture

```
English tokens ──► Embedding ──► LSTM (512) ──► [h, c] ──┐
                                                         ▼
<start> + Hinglish tokens ──► Embedding ──► LSTM (512, init=[h, c]) ──► Dense (softmax) ──► next-word probabilities
```

- **Encoder:** Embedding (256-d, masked) → Dropout → LSTM (512 units), final hidden and cell states passed to the decoder
- **Decoder:** Embedding (256-d, masked) → Dropout → LSTM (512 units) → Softmax over the Hinglish vocabulary
- **Loss / optimizer:** Sparse categorical cross-entropy, Adam (lr = 1e-3) with early stopping and learning-rate reduction
- **Inference:** Separate encoder and one-step decoder models with greedy decoding

## Getting Started

### Prerequisites
- Python 3.9+
- A GPU is recommended (Google Colab or Kaggle notebooks work well)

### Installation

```bash
git clone https://github.com/Muruganandam-2007/Hinglish-Translator.git
cd Hinglish-Translator
pip install kagglehub tensorflow pandas numpy scikit-learn nltk matplotlib jupyter
```

### Run

```bash
jupyter notebook english_to_hinglish_lstm_nmt.ipynb
```

Then run all cells top to bottom.

## Usage

After training, translate your own sentences with the helper function in the notebook:

```python
translate_batch([
    "what are you doing today?",
    "please send me the report by evening.",
])
```

## Evaluation

Translation quality is measured with **corpus BLEU** (NLTK, with smoothing) on a held-out test set:

- BLEU-1
- BLEU-2
- BLEU-4

The notebook also prints side-by-side examples of English input, reference Hinglish, and model prediction.

> Hinglish has no standard spelling, so a single sentence has many valid renderings. BLEU therefore tends to underestimate quality, and scores for a plain LSTM baseline are expected to be modest.

## Configuration

Key hyperparameters (editable in the notebook):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_SAMPLES` | 60000 | Number of sentence pairs used (`None` = all) |
| `MAX_WORDS` | 25 | Maximum words per sentence |
| `MAX_VOCAB` | 20000 | Vocabulary cap per language |
| `EMB_DIM` | 256 | Embedding dimension |
| `UNITS` | 512 | LSTM hidden units |
| `batch_size` | 128 | Training batch size |
| `epochs` | 20 | Max epochs (early stopping enabled) |

## Project Structure

```
.
├── english_to_hinglish_lstm_nmt.ipynb   # End-to-end notebook
├── best_seq2seq.keras                   # Best checkpoint (generated after training)
└── README.md
```

## Limitations & Future Work

- Add **Bahdanau / Luong attention** to handle longer sentences
- Use a **bidirectional or stacked LSTM** encoder
- Replace greedy decoding with **beam search**
- Try **subword tokenization** (SentencePiece / BPE) to reduce out-of-vocabulary words and handle spelling variation
- Report **sacreBLEU**, chrF, or human evaluation in addition to NLTK BLEU
- Fine-tune a pretrained multilingual transformer (e.g. mT5, IndicBART) for comparison

## License

Released under the MIT License. The dataset is subject to its own license on Kaggle.

## Acknowledgements

- Dataset: [yash9439 / English–Hinglish Machine Translation](https://www.kaggle.com/datasets/yash9439/eng-hinglish-machine-translation) on Kaggle
- Sequence-to-sequence learning: Sutskever et al., *Sequence to Sequence Learning with Neural Networks* (2014)
