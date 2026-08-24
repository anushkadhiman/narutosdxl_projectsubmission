# Fine-Tuning SDXL for Naruto Style

## 1. Overview

The goal was to fine-tune `stabilityai/stable-diffusion-xl-base-1.0` to generate images in a Naruto-style anime look. We used the `lambdalabs/naruto-blip-captions` dataset and trained everything on a Colab T4 GPU with 16 GB VRAM.

Full fine-tuning was too memory-intensive, so we used **LoRA and DoRA**, which train only a small number of additional parameters while keeping the main UNet frozen.

We first tested four configurations for 100 steps on 200 images, then trained the best LoRA and DoRA configurations for 600 steps on the full dataset.

## 2. How we reduced memory usage

To make 1024×1024 training possible on a T4, we used:

* **LoRA/DoRA:** Only about 11–24M parameters were trained instead of the full 2.6B-parameter UNet.
* **FP16:** Reduced memory usage for model weights and activations.
* **Precomputed latents and text embeddings:** The VAE and text encoders were removed from GPU memory during training.
* **Gradient checkpointing:** Reduced activation memory by recomputing some activations during backpropagation.
* **8-bit AdamW:** Reduced optimizer memory.
* **Memory-efficient attention:** Used xformers/SDPA to reduce attention memory.
* **Batch size 1 + gradient accumulation:** Kept GPU memory low while giving an effective batch size of 8.

We also tried QLoRA, but it ran into a data-type issue with the attention layers, so it wasn't included in the final results.

## 3. Results

In the initial screening, **rank-16 configurations performed best** for both LoRA and DoRA.

For the full training:

| Model         | Time        | Peak VRAM   | Validation Loss |
| ------------- | ----------- | ----------- | --------------- |
| LoRA r=16     | ~37 min     | 6.56 GB     | 0.0878          |
| **DoRA r=16** | **~61 min** | **6.57 GB** | **0.0786**      |

So the best model was **DoRA with rank 16 and alpha 32**.

For inference, the normal setup generated an image in about **28.4 seconds** using 10.84 GB VRAM. CPU offloading reduced VRAM to **5.98 GB**, but increased generation time to about 34.6 seconds.

## 4. Quantitative evaluation

We compared the original SDXL model with our fine-tuned DoRA model using 15 held-out Naruto images.

| Model               | CLIP Score | FID        |
| ------------------- | ---------- | ---------- |
| Base SDXL           | 31.37      | 346.97     |
| **Fine-tuned DoRA** | 30.93      | **283.30** |

The biggest improvement was the FID, which dropped by about **18%**. This indicates that the fine-tuned model's images became more similar to the Naruto-style reference images.

The CLIP score only decreased by about **1.4%**, so prompt-following ability remained fairly stable.

## 5. Conclusion

The main result is that **LoRA and DoRA made it possible to fine-tune SDXL at its native 1024×1024 resolution on a 16 GB T4 GPU**.

Among the configurations we tested, **DoRA rank 16 with alpha 32 performed best**, giving the strongest improvement in Naruto-style similarity while keeping VRAM usage around 6.6 GB.
