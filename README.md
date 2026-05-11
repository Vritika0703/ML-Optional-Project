# SVG Scaling Laws — CS-GY 6923 Optional Project

Empirical scaling laws for decoder-only Transformer language models trained on SVG (Scalable Vector Graphics) code.

---

## Project Structure

```
svg_scaling_laws/
├── configs/
│   └── model_configs.yaml      # All model sizes + training hyperparameters
├── data/                       # Created after preprocessing
│   └── processed/
│       ├── train.npy           # Token IDs (uint16)
│       ├── val.npy
│       ├── test.npy
│       ├── tokenizer.json      # HuggingFace BPE tokenizer
│       ├── dataset_stats.json
│       └── sample_svgs.json
├── models/
│   ├── transformer.py          # Decoder-only GPT (SP + µP)
│   └── data_loader.py          # Memory-mapped token streams
├── scripts/
│   ├── 01_preprocess.py        # Part 1: Data pipeline
│   ├── 01b_dataset_stats.py    # Dataset statistics + SVG render grid
│   ├── 02_train.py             # Part 2: SP training (single model)
│   ├── 02b_lr_sweep.py         # Part 2: LR sweep on tiny model
│   ├── 02c_scaling_plot.py     # Part 2: Power-law scaling plot
│   ├── 03_train_mup.py         # Part 3: µP training (single model)
│   ├── 03b_compare_scaling.py  # Part 3: SP vs µP comparison
│   ├── 04_generate.py          # Part 4: Sample generation
│   └── 05_evaluate.py          # Part 4: Quantitative evaluation
├── checkpoints/                # Created during training
│   ├── sp/                     # Standard parameterization runs
│   └── mup/                    # µP runs
├── results/                    # Plots, metrics, generated SVGs
├── run_all.py                  # End-to-end pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create a virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# Optional: install CairoSVG for SVG→PNG rendering
pip install cairosvg          # may require: brew install cairo (macOS)
```

---

## Quick Start

### Full pipeline (all steps in order)
```bash
python run_all.py
```

### Step-by-step

#### Part 1: Preprocess Data
```bash
python scripts/01_preprocess.py \
    --output_dir data/processed \
    --vocab_size 4096 \
    --max_token_len 512

python scripts/01b_dataset_stats.py \
    --data_dir data/processed \
    --output_dir results
```

#### Part 2: LR Sweep + SP Scaling Study
```bash
# LR sweep on tiny model (≈7 training runs)
python scripts/02b_lr_sweep.py --data_dir data/processed --out_dir results/lr_sweep

# Train all 5 model sizes (use best LR from sweep, e.g. 3e-4)
for MODEL in tiny small medium large xl; do
    python scripts/02_train.py --model $MODEL --data_dir data/processed --lr 3e-4
done

# Generate scaling plot + power-law fit
python scripts/02c_scaling_plot.py \
    --checkpoints_dir checkpoints/sp \
    --out_dir results \
    --extra_n_params 880000000
```

#### Part 3: µP Scaling Study
```bash
# Train all sizes with µP (same LR transfers from tiny)
for MODEL in tiny small medium large xl; do
    python scripts/03_train_mup.py --model $MODEL --data_dir data/processed --lr 3e-4
done

# Compare SP vs µP
python scripts/03b_compare_scaling.py \
    --sp_dir checkpoints/sp \
    --mup_dir checkpoints/mup \
    --out_dir results
```

#### Part 4: Generation + Evaluation
```bash
# Generate samples from best model
python scripts/04_generate.py \
    --checkpoint checkpoints/sp/xl/best_checkpoint.pt \
    --tokenizer data/processed/tokenizer.json \
    --out_dir results/generated_samples \
    --n_unconditional 10 \
    --n_conditional 5

# Evaluate (perplexity, XML validity, render rate, grid)
python scripts/05_evaluate.py \
    --checkpoint checkpoints/sp/xl/best_checkpoint.pt \
    --tokenizer data/processed/tokenizer.json \
    --test_data data/processed/test.npy \
    --samples_dir results/generated_samples \
    --out_dir results/evaluation
```

---

## Model Configurations

| Name   | ≈Params | d_model | n_layers | n_heads | d_ff  |
|--------|---------|---------|----------|---------|-------|
| Tiny   | 1M      | 128     | 4        | 4       | 512   |
| Small  | 3M      | 192     | 6        | 6       | 768   |
| Medium | 10M     | 384     | 6        | 6       | 1536  |
| Large  | 30M     | 512     | 10       | 8       | 2048  |
| XL     | 88M     | 768     | 12       | 12      | 3072  |

---

## Training Setup

| Hyperparameter     | Value                         |
|--------------------|-------------------------------|
| Optimizer          | AdamW (β₁=0.9, β₂=0.95)      |
| LR Schedule        | Cosine with linear warmup     |
| Warmup steps       | 2,000                         |
| Batch size         | ~128K tokens                  |
| Context length     | 512 tokens                    |
| Weight decay       | 0.1                           |
| Gradient clip      | 1.0                           |
| Training epochs    | 1 (scaling study), more (best)|
| Tokenizer          | BPE (vocab=4096)              |

---

## Key Results (Illustrative)

> **Note:** The results listed below are currently *illustrative* and are produced by the figure generation pipeline (`scripts/00_generate_figures.py`) for structural demonstration of the report formatting. They will be replaced with final experimental outputs once the full 3-epoch XL training run completes on a full GPU node.

- **Scaling exponent α (SP)**: `0.0835`
- **Scaling exponent α (µP)**: `0.0962`
- **Best val loss (XL, 1 epoch)**: `2.89`
- **Test perplexity**: `78`
- **XML validity rate**: `91.0%`
- **Render rate**: `85.0%`
- **Extrapolated loss @ 10×XL**: SP=`2.43`, µP=`2.23`

---

## Dependencies

See `requirements.txt`. Key packages:
- `torch >= 2.0` — training
- `datasets` — HuggingFace dataset loading
- `tokenizers` — BPE tokenizer
- `mup` — Maximal Update Parameterization
- `lxml` — XML validity checking
- `cairosvg` — SVG→PNG rendering
- `scipy` — power-law curve fitting
- `matplotlib` — plots

---

## Attribution

Model architecture adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy).
µP implementation via [microsoft/mup](https://github.com/microsoft/mup).

### Datasets
- `starvector/svg-icons-simple` (primary)
- `starvector/svg-emoji-simple` (supplementary)

### References
- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* https://arxiv.org/abs/2001.08361
- Hoffmann et al. (2022). *Training Compute-Optimal LLMs (Chinchilla).* https://arxiv.org/abs/2203.15556
- Yang et al. (2022). *Tensor Programs V: µP.* https://arxiv.org/abs/2203.09789
- Rodriguez et al. (2023). *StarVector.* https://arxiv.org/abs/2312.11556
