# Muhammed Yusuf

Building a real-time **frame generation** system for games — training the model,
distilling it, and shipping it as a desktop app.

Currently focused on: optical-flow based video frame interpolation under
**severe hardware constraints**.

---

## RDR2 Frame Generation

A desktop application that captures a running game window, generates
intermediate frames with a neural network, and presents them with frame
pacing — 30 fps → 60 fps, or 60 → 120.

**Status:** working Python application, packaged and verified as a Windows
`.exe`. Model training is ongoing.

| **Model** | RIFE-style IFNet (coarse-to-fine optical flow + refinement U-Net), trained from scratch on gameplay footage |
| **Runtime** | ONNX Runtime — CPU / GPU / NPU. No PyTorch dependency at inference |
| **Capture** | Windows Graphics Capture (per-window) and DXGI Desktop Duplication |
| **App** | CustomTkinter GUI, live measured metrics, 1×/2×/3× modes, hotkey toggle |

> Demo video, benchmarks and downloads: 



## The constraint that shaped this project

The entire model is trained on an **AMD Radeon 780M integrated GPU** via
DirectML — a chip with *no dedicated VRAM*, where every allocation comes out of
system memory.

Measured on this hardware:

```
~2.1 s / optimizer step        (256² crops, effective batch 8)
~62,000 steps  ≈  1.5 days     of uninterrupted training
142,000+       steps trained to date
```

Working inside that budget forced a discipline I would not have learned on a
rented A100: **every change had to be measured before it was kept.** Most of
the engineering below exists because guessing was too expensive.



## Engineering highlights

Selected findings from building the training pipeline. Each was found by
measurement, not intuition.

**Gradient clipping does not protect against `inf`.**
A single infinite gradient destroyed a training run and went unnoticed for
13,500 steps. The cause: `clip_grad_norm_` computes `max_norm / total_norm`;
when `total_norm` is `inf` the scale factor becomes `0`, and `inf * 0 = NaN`.
Clipping does not clip an infinite gradient — it converts it to `NaN` and
writes it into the weights. Verified in isolation, then fixed by skipping the
optimizer step on non-finite gradients.

**A data filter was discarding 99.66% valid training data.**
An audit of the triplet pipeline showed the "static frame" threshold was set
30× higher than the data supported. Of 75,994 rejected samples, only 259 were
actual duplicate frames; the rest carried real, learnable motion — including
the static-camera / moving-subject cases most valuable for occlusion handling.
Recovering them grew the dataset by 21.8% at zero cost.

**Learning rate, measured rather than guessed.**
An 8-hour controlled search: a coarse scan across a 300× range, a refinement
pass, and a confirmation pass on a second data seed. Every candidate started
from identical weights and optimizer state and saw identical batches, so the
comparison was paired. Noise decomposition showed the seed effect was a
constant offset (0.038 dB) rather than per-candidate noise, making the true
ranking resolution 0.004 dB — enough to show that the previously used learning
rate was past the point of usefulness.

**A metric with a blind spot.**
End-point error was discarding frames whose reference flow was near zero — 60%
of the static validation band. Those were exactly the frames where the model
behaved worst, inventing several pixels of motion in scenes that were still.
The gate became a routing decision instead of a rejection, and the excluded
frames are now reported as their own metric.



## Measurement infrastructure

Because the hardware budget was tight, the tooling had to answer questions
cheaply and honestly:

- **Banded validation** — quality reported separately for still / slow /
  medium / fast motion, on a fixed frame set, so a gain in easy scenes cannot
  mask a regression in hard ones
- **Controlled A/B harness** — identical weights, identical batch order,
  two-seed confirmation
- **Training health monitor** — hang detection, gradient and loss diagnostics,
  automatic restart supervision (DirectML leaks VRAM across long runs)
- **Interactive metric dashboard** — every validation metric over the full
  training history, with divergence detection



## Tech

`PyTorch` · `torch-directml` · `ONNX Runtime` · `OpenCV` · `NumPy` ·
`CustomTkinter` · `PyInstaller` · `Windows Graphics Capture` / `DXGI`


*Note: this project is source-available for review on request but not open
source. Benchmarks, methodology, and builds are published; model weights and
training code are not.*

-- You can contact me from;
📫 "xmyusufs34@gmail.com"
