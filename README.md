# Melody Diffuser

**Compose a melody by moving your finger.** Melody Diffuser is a 32M-parameter discrete diffusion transformer that generates monophonic melodies, conditioned on hand gestures through cross-attention.

**[▶ Try the demo](https://dl8ai.com/melody-diffuser)** · [Paper](docs/paper.pdf) · [Model weights](https://huggingface.co/DuncanLarz/Melody-Diffuser)

This repository holds the model and training code. The website and live demo (webcam gesture capture, playback, MIDI export) live in [DL8-AI-Portfolio](https://github.com/slamduncaan747/DL8-AI-Portfolio).

---

## Why

Text-to-music models produce convincing style and timbre but give little control over melodic and rhythmic structure. Melody Diffuser makes the *shape* of a melody the control signal: you draw it with your hand, and the model fills in the notes.

## Data

- **10M+ four-measure excerpts**, quantised to the sixteenth note: 64 steps per melody.
- **Vocabulary of 130 tokens:** MIDI pitches 0–127, `128` = hold the previous note, `129` = rest.
- **Gesture labels are generated automatically.** No gesture-labelled music dataset exists, so the pitch interval at each step is converted into one of 8 gesture tokens:

| Token | Gesture | Meaning |
|---:|---|---|
| 0 | HOLD | sustain |
| 1 | REPEAT | same pitch again |
| 2 / 3 / 4 | UP / UP+ / UP++ | small, medium, large step up |
| 5 / 6 / 7 | DOWN / DOWN+ / DOWN++ | small, medium, large step down |

At inference time the demo tracks your index finger with MediaPipe and turns its movement into the same tokens, so the model sees the same kind of signal it was trained on.

## Model

A transformer denoiser over the 64-token melody (`model.py`):

| | |
|---|---|
| Width / depth / heads | 512 / 6 blocks / 8 |
| Block | RMSNorm → self-attention → RMSNorm → **cross-attention to the gesture sequence** → RMSNorm → SwiGLU (inner 2048) |
| Conditioning | gesture embedding + learned gesture position embedding, attended to by every block |
| Timestep | sinusoidal encoding → 2-layer MLP, added to every token |
| Output | logits over the 130-token vocabulary, tied to the input embedding |
| Parameters | 32.2M |

```python
# in every transformer block
cond = gesture_emb + gesture_pos_emb
attn, _ = self.cross_attn(h, cond, cond)
x = x + attn
```

## Diffusion

Discrete diffusion with uniform replacement (`diffusion_utils.py`):

- **Forward process.** Over T = 64 steps with a linear β schedule from 10⁻⁴ to 0.05, each token is replaced by a uniformly random token with probability 1 − ∏(1 − βₜ).
- **Training objective.** The network predicts the clean melody from the noised one (cross-entropy over the vocabulary).
- **Classifier-free guidance.** The gesture condition is dropped for 15% of training examples, so generation can trade off between following the gesture and producing a natural melody (`cfg_scale`).

## Training

`train.py`:

| | |
|---|---|
| Optimiser | AdamW, learning rate 2·10⁻⁴, weight decay 0.01 |
| Schedule | linear warm-up for 2 epochs, then cosine decay |
| Batch | 2,048 melodies |
| Epochs | 10 |
| Gradient clipping | 1.0 |
| Validation | 100,000 held-out excerpts, every 2,000 steps |
| Hardware | NVIDIA A100 (Google Colab); logged with Weights & Biases |

## Repository

| File | |
|---|---|
| `model.py` | `MelodyDiffusor`: the gesture-conditioned transformer |
| `diffusion_utils.py` | noise schedule, forward corruption, loss, learning-rate schedule |
| `train.py` | training loop with validation and checkpointing |
| `docs/paper.pdf` | research write-up |

```bash
pip install -r requirements.txt
python train.py   # expects the tokenised melodies and gesture labels as pickles (paths at the top of the file)
```

Pretrained weights: [`Best-Melody-Diffuser.pth`](https://huggingface.co/DuncanLarz/Melody-Diffuser/resolve/main/Best-Melody-Diffuser.pth)

```python
import torch
from model import MelodyDiffusor

model = MelodyDiffusor(vocab_size=130, seq_len=64, dim=512, n_layers=6, n_heads=8, ffn_inner_dim=2048)
model.load_state_dict(torch.load("Best-Melody-Diffuser.pth", map_location="cpu"))
model.eval()
```

## Next steps

Polyphony: a variational autoencoder over four-part Bach chorales, enabling latent diffusion over harmony.

---

Independent research by Duncan Larzelere. Questions or collaboration: duncan.larzelere@gmail.com
