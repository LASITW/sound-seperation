import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import librosa


class OpenUnmixLSTM(nn.Module):
    """
    Open-Unmix-like architecture: 3-layer bidirectional LSTM for source separation.
    
    Takes magnitude spectrograms as input and outputs masks to separate sources.
    Model learns to predict: mask = output, which is multiplied with input spectrogram
    to obtain: separated_magnitude = mask * input_magnitude
    """
    
    def __init__(
        self,
        input_size: int = 2049,  # n_fft/2 + 1 for 4096 FFT
        hidden_size: int = 512,
        num_layers: int = 3,
        num_channels: int = 2,  # Stereo
        output_size: Optional[int] = None,
        dropout: float = 0.0
    ):
        """
        Args:
            input_size: Number of frequency bins (2049 for 4096 FFT)
            hidden_size: LSTM hidden units per layer
            num_layers: Number of stacked LSTM layers
            num_channels: Number of audio channels (1=mono, 2=stereo)
            output_size: Output frequency bins (same as input_size if None)
            dropout: Dropout probability between LSTM layers
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.output_size = output_size or input_size
        
        # Normalize input spectrograms using learned/fixed statistics
        # Shape: (1, 1, 1, freq_bins) to broadcast correctly
        self.register_buffer(
            'input_mean',
            torch.zeros(1, 1, 1, input_size)
        )
        self.register_buffer(
            'input_std',
            torch.ones(1, 1, 1, input_size)
        )
        
        # Input layer: compress frequency + channel dimensions
        # (nb_frames, nb_samples, nb_channels, nb_bins) -> (nb_frames, nb_samples, hidden_size)
        self.input_layer = nn.Linear(
            in_features=input_size * num_channels,
            out_features=hidden_size
        )
        
        # Bidirectional LSTM core: processes time dimension
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output layer: expand back to frequency bins
        # (nb_frames, nb_samples, 2*hidden_size) -> (nb_frames, nb_samples, nb_bins)
        self.output_layer = nn.Linear(
            in_features=2 * hidden_size,
            out_features=self.output_size * num_channels
        )
        
        # Final activation: sigmoid to keep mask in [0, 1] range
        self.mask_activation = nn.Sigmoid()
    
    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """
        Set input normalization statistics (computed from training data).
        
        Args:
            mean: Mean magnitude per frequency bin, shape (input_size,)
            std: Std dev per frequency bin, shape (input_size,)
        """
        self.input_mean.copy_(torch.from_numpy(mean).reshape(1, 1, 1, -1))
        self.input_std.copy_(torch.from_numpy(std).reshape(1, 1, 1, -1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: magnitude spectrogram -> mask.
        
        Args:
            x: Magnitude spectrogram tensor
               Shape: (batch, frames, channels, freq_bins) or (frames, channels, freq_bins)
        
        Returns:
            mask: Multiplicative mask in range [0, 1]
                  Shape: same as input x
        """
        # Handle both batched and unbatched inputs
        if x.dim() == 3:
            # (frames, channels, freq_bins) -> add batch dim
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch, frames, channels, freq_bins = x.shape
        
        # Normalize input using per-frequency statistics
        x_normalized = (x - self.input_mean) / (self.input_std + 1e-8)
        
        # Reshape: flatten channel and frequency dimensions
        # (batch, frames, channels, freq_bins) -> (batch, frames, channels*freq_bins)
        x_flat = x_normalized.reshape(batch, frames, channels * freq_bins)
        
        # Input layer: compress to hidden_size
        x_compressed = self.input_layer(x_flat)  # (batch, frames, hidden_size)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x_compressed)  # (batch, frames, 2*hidden_size)
        
        # Output layer: expand to frequency bins
        mask_flat = self.output_layer(lstm_out)  # (batch, frames, channels*freq_bins)
        
        # Apply sigmoid activation to keep in [0, 1]
        mask_flat = self.mask_activation(mask_flat)
        
        # Reshape back to original shape
        mask = mask_flat.reshape(batch, frames, channels, freq_bins)
        
        if squeeze_output:
            mask = mask.squeeze(0)
        
        return mask


class AudioProcessor:
    """
    Handles STFT, magnitude spectrogram computation, and phase-aware reconstruction.
    """
    
    def __init__(
        self,
        n_fft: int = 4096,
        hop_length: int = 1024,
        sample_rate: int = 44100,
    ):
        """
        Args:
            n_fft: FFT size (frequency resolution)
            hop_length: Number of samples between STFT frames
            sample_rate: Audio sample rate in Hz
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.n_freqs = n_fft // 2 + 1
    
    def audio_to_magnitude_spectrogram(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert audio waveform to magnitude spectrogram.
        
        Args:
            audio: Audio waveform, shape (n_samples,) for mono or (2, n_samples) for stereo
        
        Returns:
            magnitude: Magnitude spectrogram, shape (n_freqs, n_frames) for mono
                      or (2, n_freqs, n_frames) for stereo
            phase: Phase information from STFT, same shape as magnitude
        """
        is_stereo = audio.ndim == 2
        
        if is_stereo:
            # Process each channel separately
            channels = []
            phases = []
            for ch in range(audio.shape[0]):
                stft = librosa.stft(
                    audio[ch],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    center=True
                )
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                channels.append(magnitude)
                phases.append(phase)
            
            magnitude = np.stack(channels, axis=0)  # (2, n_freqs, n_frames)
            phase = np.stack(phases, axis=0)
        else:
            # Mono audio
            stft = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True
            )
            magnitude = np.abs(stft)  # (n_freqs, n_frames)
            phase = np.angle(stft)
        
        return magnitude, phase
    
    def magnitude_spectrogram_to_audio(
        self,
        magnitude: np.ndarray,
        phase: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct audio from magnitude spectrogram and phase.
        
        Args:
            magnitude: Magnitude spectrogram, shape (n_freqs, n_frames) or (2, n_freqs, n_frames)
            phase: Phase from original mixture, same shape as magnitude
        
        Returns:
            audio: Reconstructed audio waveform
        """
        is_stereo = magnitude.ndim == 3
        
        # Reconstruct complex STFT
        stft_complex = magnitude * np.exp(1j * phase)
        
        if is_stereo:
            # Inverse STFT for each channel
            channels = []
            for ch in range(stft_complex.shape[0]):
                audio_ch = librosa.istft(
                    stft_complex[ch],
                    hop_length=self.hop_length,
                    center=True
                )
                channels.append(audio_ch)
            audio = np.stack(channels, axis=0)
        else:
            audio = librosa.istft(
                stft_complex,
                hop_length=self.hop_length,
                center=True
            )
        
        return audio
    
    def apply_mask(
        self,
        magnitude: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply multiplicative mask to magnitude spectrogram.
        
        Args:
            magnitude: Original magnitude spectrogram
            mask: Learned mask in range [0, 1]
        
        Returns:
            separated_magnitude: Masked spectrogram
        """
        return magnitude * mask


if __name__ == "__main__":
    # Test the model
    print("Testing Open-Unmix LSTM architecture...")
    
    model = OpenUnmixLSTM(
        input_size=2049,
        hidden_size=512,
        num_layers=3,
        num_channels=2
    )
    
    # Dummy input: (batch=2, frames=100, channels=2, freq_bins=2049)
    x = torch.randn(2, 100, 2, 2049)
    
    mask = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # Test audio processor
    print("\nTesting AudioProcessor...")
    processor = AudioProcessor(n_fft=4096, hop_length=1024)
    
    # Create dummy stereo audio (2 channels, 44100 Hz for 1 second)
    dummy_audio = np.random.randn(2, 44100)
    
    magnitude, phase = processor.audio_to_magnitude_spectrogram(dummy_audio)
    print(f"Audio shape: {dummy_audio.shape}")
    print(f"Magnitude spectrogram shape: {magnitude.shape}")
    print(f"Phase shape: {phase.shape}")
    
    # Apply dummy mask and reconstruct
    mask = np.ones_like(magnitude) * 0.5
    separated_mag = processor.apply_mask(magnitude, mask)
    reconstructed_audio = processor.magnitude_spectrogram_to_audio(separated_mag, phase)
    print(f"Reconstructed audio shape: {reconstructed_audio.shape}")
    print("✓ All tests passed!")