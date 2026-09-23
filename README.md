# FSU-FNO

FSU-FNO is a compact neural operator for chip thermal-field prediction and
few-shot adaptation. It combines axis-factorized spectral operators,
context-conditioned U-Net refinement, and a Fourier-residual training
objective. The released configuration uses width 36 and 10 Fourier modes.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended.

## Data

Download the benchmark datasets:

- [S1 datasets: 11 source tasks (Google Drive folder)](https://drive.google.com/drive/folders/1WzjpOAgeua03F3iLodHlVbTsRhXn1lMA)
- [S2–S5 datasets (~4.6 GB)](https://drive.google.com/file/d/15Do8Raf070VseV9cn44j1hdVpD3Rz-Un/view?usp=sharing)
- [Pretrained FSU-FNO weights](https://drive.google.com/file/d/1xpSTo0A3wZTeUCmk761u-WKMk0aXtatP/view?usp=sharing)

Extract the pretrained weights into `checkpoints/` to use the evaluation and
few-shot scripts without retraining.

Set `FSU_DATA` to the IC-ThermBench data root. The expected layout is:

```text
$FSU_DATA/
├── level2_steady/
├── level3_steady/
├── level4_steady/
└── level5_steady/
```

Each directory must contain one input `.mat` file and one output `.mat` file.
Both files use the HDF5 key `data`. Inputs are stored as `(B,P,Z,Y,X)` and
targets as `(B,Z,Y,X)`.

```bash
export FSU_DATA=/path/to/ic-thermbench
```

## Reproduce the paper experiments

Train the S2--S4 source models:

```bash
bash script/train_all.sh
```

Evaluate S2--S4 and the S5 zero-shot setting:

```bash
bash script/test_all.sh
```

Run S5 Spectrally Consistent Fine-Tuning (SCFT) for
`K = 0, 10, 50, 100, 250, 500` labels per target case:

```bash
bash script/finetune_all.sh
```

Run the three-seed S5 objective ablation comparing MSE-only, fixed spectral
weight, and SCFT at `K = 5, 10, 50`:

```bash
bash script/fewshot_curriculum.sh
```

The commands create `checkpoints/`, `results/`, and `logs/` as needed.

## Project structure

```text
fsu_fno.py         model definition
data.py            dataset loading, splitting, and normalization
losses.py          spatial and Fourier-domain objectives
metrics.py         field and hotspot metrics
checkpoint.py      checkpoint serialization
train.py           source training
test.py            source and zero-shot evaluation
finetune.py        S5 few-shot adaptation
fewshot_sweep.py   SCFT objective ablation
script/            paper experiment launchers
```

## Core configuration

- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-4`
- Batch size: `20`
- Source training: `100` epochs with StepLR `(step=2, gamma=0.9)`
- Source objective: spatial MSE plus Fourier-residual loss, `lambda=0.1`
- S5 adaptation: `50` epochs, learning rate `1e-4`
- SCFT schedule: `lambda_e = 0.1 min(1, e/10)`
- S4/S5 inputs: per-channel normalization

All reported metrics are computed after inverse normalization.
