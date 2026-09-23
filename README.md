# Learning to Access Latent Memory: Generalizable ECG Restoration under Distribution Shift

This repository contains only the LUNet implementation for the manuscript's two-stage method.

<div align="center">
  <img src="Figure1_main.jpg"/>
</div>

Stage 1, Clean-Memory Curation, trains the clean teacher pathway end to end:

`x_C -> E_T -> z_T -> memory router -> key-value memory -> memory readout -> D_T -> x_hat_C_T`

Stage 2, Memory Access Alignment (MAA), loads the Stage 1 checkpoint and freezes the teacher LUNet, curated key-value memory, value-mixing head, and decoder. Only the student LUNet bottleneck and student memory router are trainable. Inference uses the noisy ECG only:

`x_N -> E_S -> z_S -> student memory router -> frozen key-value memory -> frozen D_T -> x_hat_C_S`

The released seed-42 configurations preserve the LUNet dimensions, no-skip architecture, memory geometry, optimizer, scheduler, preprocessing, and loss coefficients from the corresponding experiments. Stage 2 computes the student reconstruction objective for reporting, while its configured training weight remains `0.0`, matching the manuscript experiment. Its optimized objective is bottleneck MSE plus memory-readout L1 alignment.

## Training

Run commands from repository root.

```bash
python train_stage1.py --config configs/stage1/stage1_Memory_Curation_from_the_Clean_ECG.yaml
python train_stage2.py --config configs/stage2/stage2_MAA.yaml
```

Use `--out-dir` to select a run directory. Use `--resume` with `latest.pt` to continue training. Stage 2 accepts `--stage1-checkpoint` when the Stage 1 checkpoint is outside the configured path.

## Evaluation

```bash
python test_stage1.py \
  --config configs/stage1/stage1_Memory_Curation_from_the_Clean_ECG.yaml \
  --checkpoint results/stage1/seed42/best_stage1.pt

python test_stage2.py \
  --config configs/stage2/stage2_MAA.yaml \
  --checkpoint results/stage2/seed42/best_stage2.pt
```

`test_stage2.py` invokes only `student_pathway(x_N)` for prediction. Paired clean ECG is used only as the evaluation target.

## Data contract

Each split directory contains matching `*_noisy.npy`, `*_noise.npy`, and `*_clean.npy` files plus `metadata.json`. Arrays are stored under the existing `instance_noisy_zscore` protocol and must satisfy `noisy == clean + noise`. Dataset files are not repartitioned or renormalized at runtime.

Default dataset paths are repository-relative placeholders under `datasets/`. Update them to local dataset locations before running training or evaluation.

## Checkpoints

Stage 1 checkpoints retain separate `E_T`, `R_T`, `C_T`, `K`, `V`, and `D` components. Stage 2 checkpoints store the student encoder states and `R_S`; frozen Stage 1 components remain referenced through the Stage 1 checkpoint and are not duplicated.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```
