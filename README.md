# Audio Source Separation - Open-Unmix Implementation

Complete from-scratch implementation of an Open-Unmix-like LSTM architecture for music source separation.

## ✓ Implementation Complete

All components built from scratch (no external Open-Unmix imports):

- **Model**: 3-layer Bidirectional LSTM with multiplicative masking (`code/model.py`)
- **Audio Processing**: STFT → magnitude spectrogram → phase-aware reconstruction (`code/model.py`)
- **Dataset**: MUSDB18 loader with augmentation and balanced sampling (`code/dataset.py`)
- **Training**: MSE loss, Adam optimizer, learning rate scheduling, early stopping (`code/train.py`)
- **Inference**: Separation and evaluation on MUSDB18 (`code/inference.py`)

## Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy librosa musdb soundfile tqdm
```

### 2. Test Implementation
```bash
python3 code/model.py        # Test model architecture
python3 code/dataset.py      # Test dataset loader (downloads MUSDB18)
```

### 3. Train Model
```bash
python3 code/train.py
```

### 4. Separate Audio
```bash
python3 code/inference.py --audio input.wav --output vocals.wav --target vocals
```

## Architecture Overview

```
Audio → STFT → Magnitude Spectrogram → 3-Layer BiLSTM → Mask → Separated Audio
         ↓                                                        ↑
      Phase (preserved for reconstruction) ----------------------┘
```

**Key Details**:
- Input: Magnitude spectrogram (frames × channels × freq_bins)
- Model: 3-layer BiLSTM (512 hidden units) with sigmoid output
- Output: Multiplicative mask in [0, 1] range
- Loss: MSE in magnitude domain
- Reconstruction: mask × mixture_magnitude + original_phase

## Techniques & Data Flow

- **Per-frequency normalization**: Compute mean/std per frequency bin on MUSDB18 magnitudes after averaging stereo channels; clamp std ≥ 1e-8. This stabilizes training and keeps loud/quiet bands balanced.
- **Channel+frequency concatenation**: Magnitudes are normalized, then channels × freq bins are flattened for the BiLSTM so it can exploit inter-channel context while still outputting channel-wise masks.
- **Phase-preserving masking**: Model predicts a [0,1] mask on magnitudes; original mixture phase is reused for reconstruction to keep timing cues intact.
- **Stereo handling**: STFT is per channel; stats are computed on averaged stereo magnitudes only for normalization, but forward/inference keep full stereo magnitudes and apply channel-wise masks.

### Dataset (`code/dataset.py`)
- Builds per-frequency mean/std from MUSDB18 by averaging stereo magnitudes, concatenating chunks, then storing stats for reuse.
- Each sample: random 6–7 s stereo crop → optional gain + channel swap → STFT per channel → magnitude normalized with saved stats for both mixture and target.
- Output shape: `(frames, channels, freq_bins)` ready for the model.

### Model (`code/model.py`)
- Loads the saved mean/std into buffers and re-normalizes any input magnitude.
- Flattens `(channels × freq_bins)` → linear → 3-layer BiLSTM → linear → sigmoid mask shaped like the input magnitude.
- Mask is multiplied elementwise with the (unnormalized) mixture magnitude to form the estimate.

### Training (`code/train.py`)
- Injects dataset normalization stats into the model at startup so train/val share the same scaling.
- Loss: MSE between predicted and target magnitudes; optimizer: Adam; gradient clipping; ReduceLROnPlateau scheduler; early stopping with checkpointing.
- DataLoader yields normalized magnitudes; targets remain aligned frame/channel/frequency-wise.

### Inference (`code/inference.py`)
- Loads checkpointed model + stored mean/std; resamples/forces stereo if needed.
- AudioProcessor computes STFT per channel → magnitude/phase; magnitude normalized with the model’s stats.
- Model predicts mask; mask × original (unnormalized) magnitude; ISTFT with original phase reconstructs separated audio.

### How sections relate
- Dataset produces normalized mixture/target magnitudes and the normalization stats.
- Model consumes normalized magnitudes, applies learned mask on the original scale.
- Training ties them together: injects stats, optimizes the mask via MSE, and checkpoints weights.
- Inference mirrors dataset preprocessing (normalization) and model forward, then reconstructs with preserved phase.

## Files

- `code/model.py` - OpenUnmixLSTM model + AudioProcessor
- `code/dataset.py` - MUSDB18Dataset with augmentation
- `code/train.py` - Full training loop
- `code/inference.py` - Inference and evaluation
- `IMPLEMENTATION_GUIDE.md` - Detailed documentation

## How It Works

### 1. Audio → Spectrogram
```python
audio (stereo): (2, 44100)           # 1 second @ 44.1 kHz
        ↓ STFT (n_fft=4096, hop=1024)
magnitude: (2, 2049, 44)             # 2049 freq bins, 44 frames
phase: (2, 2049, 44)                 # Preserved for later
```

### 2. Model Forward Pass
```python
Input: (batch, frames, channels, freq_bins)
    → Normalize (per-frequency mean/std)
    → Linear: flatten channels+freqs → 512 hidden
    → 3-Layer BiLSTM: 512 → 1024
    → Linear: 1024 → channels×freq_bins
    → Sigmoid → Mask [0,1]
```

### 3. Separation & Reconstruction
```python
separated_magnitude = mask × mixture_magnitude
separated_audio = ISTFT(separated_magnitude × e^(i×mixture_phase))
```

## Training Details

- **Batch size**: 4
- **Chunk duration**: 6 seconds
- **Learning rate**: 0.001 (decay by 0.3 on plateau)
- **Augmentation**: Random gain [0.25, 1.25], channel swap
- **Early stopping**: 140 epochs without improvement

## Training Behavior & Tips

- Early runs: train loss ≈0.20 with validation bottom near 0.366 (epoch 4), then drifting toward 0.40 → mild overfitting or LR now too high for further gains.
- ReduceLROnPlateau will lower LR after patience epochs without validation gains; shorten patience (e.g., 2–3) or manually drop LR to ~3e-4 if convergence stalls.
- Consider early stopping around the first validation minimum to avoid overfitting; reduce early-stopping patience for faster convergence.
- Use a single set of normalization stats (computed on train) for train/val/test/inference to keep scaling consistent.

## Algorithm Overview & Trade-offs

This repo uses an Open-Unmix–style 3-layer BiLSTM that predicts a [0,1] mask on magnitude spectrograms and reuses mixture phase for reconstruction.

**Pros (speed/efficiency)**
- LSTM on spectrograms is compute- and memory-light compared to large Conv/Transformer separators; runs on CPU or modest GPUs.
- Spectrogram masking processes fewer points than time-domain models at high sample rates, reducing inference cost.
- ~23M parameters keep checkpoints and latency manageable.

**Cons (speed/efficiency)**
- BiLSTM time steps are sequential, less parallel than fully convolutional or attention models on GPU/TPU.
- STFT/ISTFT adds overhead; tight hop sizes increase cost.
- Quality typically trails time-domain ConvNets (e.g., Demucs/Conv-TasNet) or Transformer hybrids that capture richer temporal detail but need more compute.

**Compared to alternatives**
- Time-domain ConvNets (Conv-TasNet/Demucs): higher SDRs and better GPU parallelism but heavier compute/memory; larger models.
- Transformer separators: strongest quality, highest memory/compute; slower to train/infer.
- This BiLSTM spectrogram approach: balanced, simpler, lighter, predictable latency; quality is moderate but hardware-friendly.

## Expected Performance

After full training on MUSDB18:
- Vocals: ~6-7 dB SDR
- Drums: ~5-6 dB SDR
- Bass: ~4-5 dB SDR
- Other: ~3-4 dB SDR

## References

- Open-Unmix Paper: https://arxiv.org/abs/1810.12947
- MUSDB18: https://github.com/sigsep/sigsep-mus-db
