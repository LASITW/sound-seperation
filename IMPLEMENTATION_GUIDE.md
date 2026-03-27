# Open-Unmix Audio Source Separation Implementation

Complete implementation of an Open-Unmix-like architecture for music source separation, including model architecture, data pipeline, training loop, and inference.

## Overview

This implementation provides:
- **3-layer Bidirectional LSTM** model for source-specific separation
- **Audio preprocessing pipeline**: STFT → magnitude spectrogram → normalized input
- **Phase-aware reconstruction**: Original phase + separated magnitude → output audio
- **Training framework**: MSE loss, Adam optimizer, learning rate scheduling, early stopping
- **Dataset loader**: MUSDB18 with balanced sampling, augmentation, and normalization

## Architecture Details

### Model: OpenUnmixLSTM

```
Input: Magnitude spectrogram (batch, frames, channels=2, freq_bins=2049)
        ↓
    Normalization (per-frequency mean/std)
        ↓
    Linear: (2*2049) → 512 (compress frequency + channel)
        ↓
    3-Layer Bidirectional LSTM: 512 → 1024 (processes time dimension)
        ↓
    Linear: 1024 → (2*2049) (expand back to frequency bins)
        ↓
    Sigmoid activation → Mask ∈ [0, 1]
        ↓
Output: Multiplicative mask (same shape as input)
```

**Key Properties:**
- Bidirectional LSTM: looks at past AND future frames (no real-time capability)
- Multiplicative masking: `output = mask × input_magnitude`
- Sigmoid activation ensures mask stays in [0, 1] range
- Per-frequency normalization for robustness

### Audio Processing Pipeline

#### 1. Audio → STFT → Magnitude Spectrogram

```python
audio (stereo): (2, 44100)                    # 1 second @ 44.1 kHz
        ↓
STFT(n_fft=4096, hop_length=1024)
        ↓
Complex spectrogram: (2, 2049, 44)           # 2049 freq bins, 44 frames
        ↓
Magnitude: |X| = √(real² + imag²)
Phase: ∠X = atan2(imag, real)
        ↓
magnitude: (2, 2049, 44)                     # Stereo channels
phase: (2, 2049, 44)                         # Preserve for reconstruction
```

**STFT Parameters:**
- `n_fft = 4096`: Frequency resolution (44100/4096 ≈ 10.77 Hz per bin)
- `hop_length = 1024`: 75% overlap (4096/4 = 1024)
- Window: Hann (smooth tapering)
- Frequency bins: n_fft/2 + 1 = 2049

#### 2. Normalization (Per-Frequency Statistics)

```python
# Computed from training data
mean: (2049,)  # Average magnitude per frequency bin
std: (2049,)   # Std dev per frequency bin

# Applied during training/inference
magnitude_normalized = (magnitude - mean) / (std + ε)
```

**Why per-frequency?**
- Different frequencies have different amplitude ranges
- Low frequencies typically much louder than high frequencies
- Normalization helps model learn meaningful patterns at all frequencies

#### 3. Phase-Aware Reconstruction

```python
separated_magnitude = mask × mixture_magnitude    # From model output
separated_phase = mixture_phase                   # Preserve original

# Reconstruct complex spectrogram
complex_stft = separated_magnitude × e^(i × separated_phase)

# Inverse STFT back to time domain
audio = ISTFT(complex_stft)
```

**Critical Detail:** We use the **original mixture's phase**, not the target's phase. This is the Wiener filtering approach:
- Separates magnitude information (what the model learns)
- Preserves phase relationships from original mixture
- Results in high-quality reconstruction with minimal artifacts

## Training Algorithm

### Loss Function: Mean Squared Error (Magnitude Domain)

```
L = (1/N) Σ (target_magnitude - predicted_mask × mixture_magnitude)²
```

Why MSE?
- Simple, differentiable regression loss
- Works well for magnitude prediction
- Equal weight to all frequency bins (can be modified for perceptual loss)

### Optimization

**Optimizer: Adam**
- Learning rate: 0.001 (initial)
- β₁ = 0.9, β₂ = 0.999 (default)
- Weight decay (L2): 0.00001

**Learning Rate Scheduling: Plateau-based decay**
```
If val_loss doesn't improve for 80 epochs:
    lr ← lr × 0.3
```

**Early Stopping:**
```
If val_loss doesn't improve for 140 epochs:
    Stop training
```

### Training Loop Structure

```python
for epoch in range(1000):
    # Training
    for batch in train_loader:
        mixture, target = batch  # Normalized magnitude spectrograms
        
        # Forward pass
        mask = model(mixture)
        loss = MSE(mask × mixture, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss = evaluate(val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
    else if patience_counter > 140:
        break
```

### Data Loading & Augmentation

**MUSDB18 Dataset:**
- 100 training tracks, 50 test tracks
- Each track has isolated stems: mixture, vocals, drums, bass, other
- Stereo audio at 44.1 kHz

**Balanced Sampling:**
```
per_epoch: Select each track once randomly
per_track: Extract random 6-second chunks
per_chunk: Apply augmentation
```

**Augmentation (during training):**
1. **Random gain**: 0.25-1.25× (handles gain variation)
2. **Channel swap**: 50% probability (data augmentation)

**Training Batch:**
```
Batch size: 4
Chunks per batch: 4 × 6 seconds each
Per epoch: 100 tracks × 64 samples/track = 6400 total chunks
```

**Normalization:**
```
mean, std = compute_from_training_data()  # Per-frequency statistics
model.set_normalization(mean, std)        # Fixed during training
```

## File Structure & Usage

### Files

```
model.py          - Model architecture + audio processing
dataset.py        - MUSDB18 dataset loader with augmentation
train.py          - Training script
inference.py      - Inference and evaluation
```

### Quick Start

**1. Install Dependencies**
```bash
pip install torch torchaudio librosa musdb numpy soundfile tqdm
# Optional: museval for evaluation
pip install museval
```

**2. Prepare MUSDB18** (first time only)
```bash
python -c "import musdb; musdb.DB(download=True)"
```

**3. Train Model**
```bash
python train.py
# Saves checkpoints to ./checkpoints/
# Training logs to console
```

**4. Separate Audio**
```bash
python inference.py --audio input.wav --output separated_vocals.wav --target vocals
```

**5. Evaluate on MUSDB18**
```bash
python inference.py --evaluate
# Outputs SDR/SIR/SAR metrics to ./eval_results/
```

## Key Hyperparameters & Tuning

### Model
- `hidden_size`: 512 (LSTM units) - increase for larger models
- `num_layers`: 3 - more layers = deeper but slower
- `num_channels`: 2 (stereo) - set to 1 for mono

### Training
- `batch_size`: 4 (reduce if OOM)
- `learning_rate`: 0.001 (0.0005-0.002 typical)
- `weight_decay`: 0.00001 (L2 regularization)

### Data
- `chunk_duration`: 6.0 seconds (balance between GPU memory and temporal context)
- `samples_per_track`: 64 (chunks per track per epoch)

### Augmentation
- `gain_range`: [0.25, 1.25] - reduce if overfitting
- `channel_swap`: 50% probability

## Expected Performance

On MUSDB18 test set (after full training):
- **Vocals**: ~6-7 dB SDR
- **Drums**: ~5-6 dB SDR  
- **Bass**: ~4-5 dB SDR
- **Other**: ~3-4 dB SDR

(Actual values depend on training duration, data augmentation, and hyperparameters)

## Advanced: Multi-Source Training

For best results, train **separate models per source**:

```python
# Train vocals model
dataset = MUSDB18Dataset(target='vocals')
trainer = Trainer(...)
trainer.fit()

# Train drums model
dataset = MUSDB18Dataset(target='drums')
trainer = Trainer(...)
trainer.fit()

# etc for bass, other
```

This allows:
- Custom augmentation per source
- Per-source hyperparameter tuning
- Better performance but 4× training time

## Troubleshooting

**OOM (Out of Memory):**
- Reduce `batch_size` (4 → 2)
- Reduce `chunk_duration` (6 → 4 seconds)
- Use CPU: `python train.py --device cpu`

**Model not learning:**
- Check normalization statistics are correct
- Try higher learning rate (0.001 → 0.002)
- Verify data loading works: `python dataset.py`

**Poor separation quality:**
- Train longer (200+ epochs minimum)
- Use all 4 sources (not just vocals)
- Increase data augmentation

## References

- **Open-Unmix Paper**: https://arxiv.org/abs/1810.12947
- **MUSDB18**: https://github.com/sigsep/sigsep-mus-db
- **Norbert (Phase recovery)**: https://github.com/sigsep/norbert
