# Fine-Tuning SDXL for Naruto Style — Technical Write-Up

## 1. High-level overview

The goal was to fine-tune `stabilityai/stable-diffusion-xl-base-1.0` to generate images in
the Naruto anime art style, using the `lambdalabs/naruto-blip-captions` dataset, entirely
within a free-tier Colab T4 GPU (~16GB VRAM) including at SDXL's native 1024x1024
training resolution, which is the resolution the base model naively will not fit at without
the optimizations described below.

**Approach:** rather than full fine-tuning which requires storing gradients and optimizer
state for all ~2.6B UNet parameters well beyond 16GB, the UNet is kept frozen and small
low-rank adapters are trained on top of it. To use the limited ~5-hour Colab GPU budget
efficiently, training is staged in two phases:

1. **Screening** — LoRA and DoRA, two hyperparameter variants each (rank 8 and rank 16), for
   a short 100 steps on a 200-example subset, at the real target resolution (so a memory
   problem shows up in minutes, not after a long run). QLoRA was also attempted at this
   stage (see Section 2.7).
2. **Full training** — only the best-performing config per method (by validation loss)
   proceeds to a full 600-step run on the complete dataset (~1,037 training images, ~183
   held out for validation).

The single best checkpoint by validation loss was then evaluated with:

- A fixed prompt, fixed seed comparison against the base (un-fine-tuned) model.
- Quantitative scoring: CLIP score (prompt adherence) and FID (style fidelity against real,
  held-out Naruto images).
- A short inference optimization benchmark (latency/VRAM across a few configurations).

## 2. Techniques used to fit training in 16GB VRAM

### 2.1 Parameter-efficient fine-tuning (LoRA / DoRA)

Instead of updating all ~2.6B UNet parameters, small low-rank adapter matrices are injected
into the attention projection layers (`to_q`, `to_k`, `to_v`, `to_out.0`) and only those are
trained — **~11–24M parameters depending on rank, under 0.1% of the UNet**. This is the
single biggest approach: full fine-tuning needs gradient + optimizer-state memory for every
parameter (roughly 4x a parameter's own size in mixed precision with Adam), which for 2.6B
parameters alone exceeds a T4's entire VRAM before a single activation is computed.
LoRA/DoRA collapse that cost by over 99%.

**DoRA** (Weight-Decomposed Low-Rank Adaptation) is the same idea with one refinement. it
decomposes the weight update into a magnitude component and a direction component, and only
applies the low-rank update to direction. In our results it slightly outperformed plain LoRA
at the same rank (Section 4).

### 2.2 fp16 mixed precision, everywhere

UNet, VAE, and both text encoders all run in fp16. This roughly halves memory for weights and
activations compared to fp32, at negligible quality cost for this task. diffusion models are
routinely trained and run in fp16/bf16.

### 2.3 Precompute-then-free (the second-biggest lever)

The VAE and both CLIP text encoders are **frozen**. they never receive gradients. Rather
than keeping them loaded in VRAM for the entire training run, they are run **once** over the
whole dataset to precompute image latents and text embeddings, which are cached to CPU
memory, and then the VAE and text encoders are **deleted from VRAM entirely**. From that
point on, every training step only needs the UNet + adapter in memory no repeated VAE/text
encoder forward passes, and no VAE/text-encoder memory footprint competing with the UNet's
during training.

### 2.4 Gradient checkpointing

Trades compute for memory: instead of storing every intermediate activation for the backward
pass, activations are recomputed on the fly during backpropagation. This is essential at
1024x1024, where activation memory (not parameter memory) becomes the dominant cost.

### 2.5 8-bit AdamW (`bitsandbytes`)

Adam-family optimizers keep two extra state tensors (first and second moment) per trainable
parameter. `bitsandbytes`' 8-bit AdamW quantizes that optimizer state, cutting its memory
roughly in half, a real saving even though the LoRA/DoRA parameter count is already small,
and effectively free given the batch-size-1 training loop is not compute-bound.

### 2.6 xformers / SDPA memory-efficient attention

Standard attention has quadratic memory scaling in sequence length. Memory-efficient
attention kernels (xformers, or PyTorch's built-in SDPA / Flash-Attention-2 as a fallback if
xformers fails to install) reduce this substantially, important at 1024x1024, where the
spatial token count is 4x that of 512x512.

### 2.7 Batch size 1 + gradient accumulation, and why QLoRA was attempted but not used

Batch size 1 keeps per-step activation memory to a minimum; gradient accumulation (8 steps)
recovers a reasonable effective batch size without ever holding multiple samples' worth of
activations simultaneously.

**QLoRA** (LoRA with the base model in 4-bit) was also tried because it can reduce the memory used by the base model. However, it did not work in our runs because of a data type problem in the attention part of the model. We found the issue was related to how PEFT and 4-bit training work together. We found the one-line fix, but it was not included in the final results and will be tested separately.

### 2.8 Native 1024x1024 resolution

With the above stack in place, training proceeds directly at SDXL's native 1024x1024
resolution rather than a reduced resolution, full fine-tuning cannot do this on a T4, but
LoRA/DoRA-class adapters, combined with the memory techniques above, can.

## 3. Motivation for these choices

The guiding constraint was simple: **16GB is not enough for full fine-tuning of a 2.6B-
parameter UNet at any reasonable batch size, and the free Colab GPU budget (~5 hours) is not
enough to brute-force a large hyperparameter search.** Every technique above was chosen to
address one of those two constraints directly:

- LoRA/DoRA + 8-bit Adam + gradient checkpointing address the **memory** constraint at the
  parameter/optimizer level.
- fp16 + precompute-then-free + memory-efficient attention address the **memory** constraint
  at the activation/intermediate-tensor level.
- The **screen-then-commit training strategy** running short tests across several configurations and fully training only the best ones, helps address the time constraint. It avoids spending hours of limited free GPU time fully training configurations that a short run has already shown to be weaker.

## 4. Results

### Screening (100 steps, 200-example subset, 1024x1024)

| Config              | Peak VRAM | Time  | Final val loss      |
| ------------------- | --------- | ----- | ------------------- |
| LoRA A (r=8, α=16)  | 6.44 GB   | 660s  | 0.0894              |
| LoRA B (r=16, α=32) | 6.56 GB   | 663s  | **0.0874** (winner) |
| DoRA A (r=8, α=16)  | 6.45 GB   | 1103s | 0.0903              |
| DoRA B (r=16, α=32) | 6.57 GB   | 1104s | **0.0868** (winner) |

### Full training (600 steps, full dataset, 1024x1024)

| Config                  | Peak VRAM | Time            | Final train loss | Final val loss    |
| ----------------------- | --------- | --------------- | ---------------- | ----------------- |
| LoRA B (r=16, α=32)     | 6.56 GB   | 2233s (~37 min) | 0.0870           | 0.0878            |
| **DoRA B (r=16, α=32)** | 6.57 GB   | 3659s (~61 min) | 0.1170           | **0.0786 (best)** |

**Best checkpoint: DoRA, rank 16, alpha 32.**

### Inference optimization (best checkpoint, 1024x1024, 30 steps)

| Config                                                             | Latency     | Peak VRAM   |
| ------------------------------------------------------------------ | ----------- | ----------- |
| Baseline (SDPA — PyTorch's default Flash-Attention-2-class kernel) | 28.4s/image | 10.84 GB    |
| + VAE tiling/slicing                                               | 29.6s/image | 10.85 GB    |
| + Model CPU offload                                                | 34.6s/image | **5.98 GB** |

CPU offload trades roughly 20% more latency for nearly half the peak VRAM, a reasonable
choice if generating on tighter hardware than the training GPU. VAE tiling/slicing showed
no meaningful benefit at this scale/resolution combination in our runs.

### Quantitative style/prompt evaluation (15 held-out real Naruto images, same captions)

| Model                                  | CLIP score (prompt adherence) | FID (style fidelity vs. real Naruto images) |
| -------------------------------------- | ----------------------------- | ------------------------------------------- |
| Base SDXL (no fine-tune)               | 31.37                         | 346.97                                      |
| **Fine-tuned (DoRA, best checkpoint)** | 30.93                         | **283.30**                                  |

**Effectiveness:** FID got much better (347.0 → 283.3, about an 18% decrease). This means the fine-tuned model's outputs became more similar to real Naruto-style images, showing that the model learned the target style. The CLIP score only dropped a little (31.4 → 30.9, about 1.4%). This is a small and expected trade-off when changing a general model to focus more on one specific style, while the model still follows prompts well.

## 5. Repository contents

```
notebooks/

  train.ipynb
      Training process: first we do short tests, then we fully train the better settings.
      We tried LoRA and DoRA.

  inference.ipynb
      Compare the base model and the fine-tuned model using the 3 test prompts.
      Also includes CLIP and FID scores.

full_runs/
      Saved LoRA weights from the configs that were fully trained.

training_curves.png
comparison_grid.png
      Training graphs and image comparisons.

README.md
      This file.
```

## 6. Known limitations

- We kept the screening and hyperparameter testing small (2 settings for each method and 100 short training steps) because of the limited free GPU time. The results give us a general idea of which setting is better, not a complete search.

- We tried QLoRA, but it is not included in the final results.

- We used 600 steps for full training, which is not a lot for a dataset with around 1,037 images. The loss curves suggest that the model could improve more if we had more GPU time.
