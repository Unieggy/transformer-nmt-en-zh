
# English → Chinese Transformer NMT (SentencePiece+PyTorch)

A minimal, clean English→Chinese Neural Machine Translation system:

- Transformer encoder–decoder in **PyTorch**
- **SentencePiece** BPE tokenization
- **Gradio** app for quick interactive translation
- Scripts for **training**, **CLI translation**, and basic evaluation

> Works locally and in **Google Colab**. The app auto-detects Colab paths when possible.

---

## Contents
- [1. Using (Pretrained Inference)](#1-using-pretrained-inference)
  - [Local](#local)
  - [Google Colab](#google-colab)
  - [CLI without UI](#cli-without-ui)
- [2. Training (From Scratch)](#2-training-from-scratch)
  - [Data Format](#data-format)
  - [Train SentencePiece Tokenizers](#train-sentencepiece-tokenizers)
  - [Configure](#configure)
  - [Run Training](#run-training)
  - [Evaluate (BLEU)](#evaluate-bleu)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 1. Using (Pretrained Inference)

### Local

1) **Clone & create env**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate 
````

2. **Install minimal inference deps**

```bash
pip install torch gradio sentencepiece
```

3. **Put your artifacts in place**

```
weights/tmodel_19.pt
tokenizers/tokenizer_en.model
tokenizers/tokenizer_zh.model
```

4. **Launch the app**

```bash
python app.py
```

Open the local URL printed in terminal (e.g., `http://127.0.0.1:7860`).

* Click **Load model** → you should see `✅ Model loaded`.
* Enter English text → **Translate**.

> The app also supports auto-loading if it finds the default files.

---

### Google Colab

1. **Install deps**

```python
!pip install torch sentencepiece gradio
```

2. **Mount Drive (recommended)**

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Place files in Drive**

```
/content/drive/MyDrive/weights/tmodel_19.pt
/content/drive/MyDrive/tokenizer_en.model
/content/drive/MyDrive/tokenizer_zh.model
```

4. **Run**

```python
!python app.py
```

Click the **public `.gradio.live` link** Colab prints.

> `app.py` prefers **Colab paths** if they exist, otherwise falls back to local defaults.

---

### CLI without UI

You can translate from the command line (uses `translate.py`):

```bash
python translate.py \
  --ckpt weights/tmodel_19.pt \
  --sp_src tokenizers/tokenizer_en.model \
  --sp_tgt tokenizers/tokenizer_zh.model \
  --text "The weather is nice today; shall we take a walk by the beach?"
```

---

## 2. Training (From Scratch)

> You’ll need a GPU for reasonable speed. The defaults target **EN→ZH**.

### Data Format

Use aligned, line-by-line parallel text:

```
data/
  train.en   train.zh
  valid.en   valid.zh
  test.en    test.zh   # optional, for evaluation
```

Each line in `train.en` corresponds to the same line number in `train.zh` (and so on).

> If you prefer Hugging Face datasets (e.g., **opus-100**), you can modify `train.py` to load via `datasets` instead of raw files.

### Train SentencePiece Tokenizers

Create separate tokenizers for EN and ZH (example vocab size = 32k).
**Important:** include special tokens.

```bash
spm_train \
  --input=data/train.en \
  --model_prefix=tokenizer_en \
  --vocab_size=32000 \
  --model_type=bpe \
  --character_coverage=1.0 \
  --user_defined_symbols="[PAD],[SOS],[EOS]"

spm_train \
  --input=data/train.zh \
  --model_prefix=tokenizer_zh \
  --vocab_size=32000 \
  --model_type=bpe \
  --character_coverage=1.0 \
  --user_defined_symbols="[PAD],[SOS],[EOS]"
```

Move the resulting files into `tokenizers/`:

```
tokenizers/tokenizer_en.model
tokenizers/tokenizer_zh.model
```

### Configure

Open **config.py** and adjust as needed:

```python
def get_config():
    return {
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 1e-4,
        "dmodel": 512,
        "seq_len": 128,
        "lang_src": "en",
        "lang_tgt": "zh",
        "model_basename": "tmodel_",
        "model_folder": "weights",
        "tokenizer_file": "tokenizers/tokenizer_{0}.model",
        "experiment_name": "runs/tmodel",
        "preload": None, 
    }
```

> Paths in the app are Colab-aware; for training you can keep everything relative inside the repo.

### Run Training

```bash
pip install -r requirements.txt  # (see "Dependencies" section)
python train.py
```

Checkpoints are saved in `weights/` as `tmodel_<epoch>.pt`.

### Evaluate (BLEU)

Install `sacrebleu` and compute BLEU on your test set predictions:

```bash
pip install sacrebleu
# Suppose you generated translations to outputs/test.zh.hyp
sacrebleu data/test.zh -i outputs/test.zh.hyp -m bleu -b
```

---

## Project Structure

```
.
├── app.py                 # Gradio app (Colab-aware defaults, greedy decode)
├── config.py              # Hyperparams & paths (used by train and app)
├── dataset.py             # Dataset + padding/masks
├── model.py               # Transformer encoder–decoder
├── train.py               # Training loop
├── translate.py           # CLI translator
├── tokenizers/
│   ├── tokenizer_en.model
│   └── tokenizer_zh.model
└── weights/
    └── tmodel_19.pt       # Example checkpoint
```

---

## Dependencies

### Inference-only (minimum)

```bash
pip install torch sentencepiece gradio
```

### Full (training + tools)

```bash
pip install datasets tokenizers torchmetrics tensorboard tqdm sentencepiece torch gradio
```
### Weight file
https://drive.google.com/file/d/1efKsvJC5EC97WuDPRw6eoP057wScTdJT/view?usp=drive_link

Optionally:

```bash
pip install sacrebleu   # BLEU evaluation
```

> Python 3.10–3.11 recommended.
> GPU: CUDA-enabled PyTorch for training (CPU works but is slow).

---

## Troubleshooting

**UI shows just “Error”. How do I see details?**
`app.py` prints the full traceback into a **Status** box and to the console. If you don’t see it, check the Colab cell output.

**“Please click ‘Load model’ first.”**
You clicked **Translate** before loading, or loading failed (bad paths). Fix paths, click **Load model** and wait for `✅ Model loaded`.

**`FileNotFoundError` for weights/tokenizers**
Ensure paths match where the files actually are.

* Colab (Drive): `/content/drive/MyDrive/...`
* Local: `weights/tmodel_19.pt`, `tokenizers/tokenizer_en.model`, etc.

**`NotImplementedError: Module [Transformer] is missing the required "forward" function`**
Your `model.py` doesn’t implement `forward`. Either:

* Add `forward(self, src, tgt, src_mask=None, tgt_mask=None)` that calls `encode`, `decode`, and `projection`, **or**
* Use the masked `greedy()` in `app.py` that calls `model.encode`, `model.decode`, `model.projection` directly (the app already includes a version that doesn’t require a `forward`).

**`index out of range` in embeddings**
Likely `[SOS]/[EOS]/[PAD]` weren’t added to SentencePiece; their IDs are `-1`. Re-train SP with `--user_defined_symbols="[PAD],[SOS],[EOS]"`. The app’s greedy decoder also guards for missing SOS/EOS.

**`size mismatch` when loading checkpoint**
You changed hyperparams (e.g., `dmodel`, heads, vocab size) relative to the checkpoint. Use the same config or retrain.

**GPU not used**
Ensure PyTorch sees CUDA and your runtime/GPU is enabled (Colab: Runtime → Change runtime type → GPU).

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

* Vaswani et al., *“Attention Is All You Need.”* 2017
* Google’s **SentencePiece**
* Hugging Face **datasets/tokenizers** (optional for training)

```

---
```
