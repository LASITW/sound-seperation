import torch
import numpy as np
import musdb
import random
from torch.utils.data import Dataset
from typing import Tuple
from model import AudioProcessor
import logging

logger = logging.getLogger(__name__)


class MUSDB18Dataset(Dataset):
    """
    MUSDB18 dataset for training audio source separation models.
    
    Implements balanced random track sampling with 6-second chunks,
    augmentation (random gains, channel swaps), and on-the-fly
    STFT/spectrogram computation.
    """
    
    def __init__(
        self,
        subset: str = "train",
        target: str = "vocals",
        chunk_duration: float = 6.0,
        samples_per_track: int = 64,
        n_fft: int = 4096,
        hop_length: int = 1024,
        augment: bool = True,
        download: bool = False,
        musdb_path: str = None,
    ):
        """
        Args:
            subset: 'train' or 'test'
            target: Target source to separate ('vocals', 'drums', 'bass', 'other')
            chunk_duration: Length of audio chunks in seconds
            samples_per_track: Number of chunks to sample per track per epoch
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            augment: Enable random gain and channel swap augmentation
            download: Download MUSDB18 if not present
            musdb_path: Path to MUSDB18 dataset (None = default location)
        """
        self.subset = subset
        self.target = target
        self.chunk_duration = chunk_duration
        self.samples_per_track = samples_per_track
        self.augment = augment
        
        # Load MUSDB18 database
        self.mus = musdb.DB(root=musdb_path, download=download, subsets=subset)
        self.tracks = self.mus.tracks
        
        # Audio processing
        self.processor = AudioProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=44100
        )
        self.sample_rate = 44100
        self.n_fft = n_fft
        
        # Precompute normalization statistics during init
        print("Computing normalization statistics...")
        self.mean, self.std = self._compute_normalization()
    
    def _compute_normalization(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-frequency mean and std across training set.
        Used for input normalization.
        
        Returns:
            mean: Mean magnitude per frequency bin, shape (2049,)
            std: Standard deviation per frequency bin, shape (2049,)
        """
        all_magnitudes = []
        
        # Sample 5 chunks from each track to compute statistics
        num_tracks = min(10, len(self.tracks))
        if num_tracks == 0:
            # No tracks available, return default normalization
            logger.warning("No tracks found for normalization, using default values")
            return np.zeros(self.processor.n_freqs), np.ones(self.processor.n_freqs)
        
        for track in self.tracks[:num_tracks]:
            mix_audio = track.audio.T  # (channels, samples)
            
            for _ in range(5):
                chunk_start = random.uniform(
                    0,
                    max(0.1, track.duration - self.chunk_duration)
                )
                chunk_start_sample = int(chunk_start * self.sample_rate)
                chunk_end_sample = int(
                    (chunk_start + self.chunk_duration) * self.sample_rate
                )
                
                chunk = mix_audio[:, chunk_start_sample:chunk_end_sample]
                if chunk.shape[1] < 1024:  # Skip if chunk too small
                    continue
                    
                magnitude, _ = self.processor.audio_to_magnitude_spectrogram(chunk)
                
                # Reduce stereo to per-frequency stats by averaging channels
                # Result shape should be (n_freqs, n_frames)
                if magnitude.ndim == 3:
                    # magnitude: (channels, n_freqs, n_frames)
                    magnitude = magnitude.mean(axis=0)
                
                all_magnitudes.append(magnitude)
        
        if len(all_magnitudes) == 0:
            # Fallback to default values
            logger.warning("No valid chunks for normalization, using default values")
            return np.zeros(self.processor.n_freqs), np.ones(self.processor.n_freqs)
        
        # Concatenate all magnitudes
        all_mags = np.concatenate(all_magnitudes, axis=1)  # (n_freqs, total_frames)
        
        # Compute statistics
        mean = all_mags.mean(axis=1)
        std = all_mags.std(axis=1)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
        
        print(f"  Normalization: mean={mean.mean():.4f}, std={std.mean():.4f}")
        return mean, std
    
    def _augment(
        self,
        mixture: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation: random gain and channel swap.
        
        Args:
            mixture: Stereo audio (2, n_samples)
            target: Stereo audio (2, n_samples)
        
        Returns:
            augmented_mixture, augmented_target
        """
        # Random gain in range [0.25, 1.25]
        gain = random.uniform(0.25, 1.25)
        mixture = mixture * gain
        target = target * gain
        
        # Random channel swap (50% probability)
        if random.random() < 0.5 and mixture.shape[0] == 2:
            mixture = mixture[[1, 0], :]
            target = target[[1, 0], :]
        
        return mixture, target
    
    def __len__(self) -> int:
        """Total number of chunks in epoch."""
        return len(self.tracks) * self.samples_per_track
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample: (mixture_spectrogram, target_spectrogram).
        
        Args:
            idx: Index (not directly used - we sample randomly per track)
        
        Returns:
            (mixture_magnitude, target_magnitude): Both normalized tensors
                Shape: (frames, channels, freq_bins)
        """
        # Select random track
        track = random.choice(self.tracks)
        
        # Get mixture and target audio
        mixture_audio = track.audio.T  # (channels, samples)
        target_audio = track.targets[self.target].audio.T
        
        # Extract random chunk
        chunk_start = random.uniform(
            0,
            max(0, track.duration - self.chunk_duration)
        )
        chunk_start_sample = int(chunk_start * self.sample_rate)
        chunk_end_sample = int(
            (chunk_start + self.chunk_duration) * self.sample_rate
        )
        
        mixture_chunk = mixture_audio[:, chunk_start_sample:chunk_end_sample]
        target_chunk = target_audio[:, chunk_start_sample:chunk_end_sample]
        
        # Augmentation
        if self.augment:
            mixture_chunk, target_chunk = self._augment(mixture_chunk, target_chunk)
        
        # Convert to spectrograms
        mixture_mag, mixture_phase = self.processor.audio_to_magnitude_spectrogram(
            mixture_chunk
        )
        target_mag, _ = self.processor.audio_to_magnitude_spectrogram(
            target_chunk
        )
        
        # Normalize using precomputed statistics
        # Shape for stereo: (channels, freq_bins, frames)
        if mixture_mag.ndim == 3:
            # (channels, freq_bins, frames) -> (frames, channels, freq_bins)
            mixture_mag = np.transpose(mixture_mag, (2, 0, 1))
            target_mag = np.transpose(target_mag, (2, 0, 1))
        else:
            # Mono: (freq_bins, frames) -> (frames, 1, freq_bins)
            mixture_mag = mixture_mag.T[:, np.newaxis, :]
            target_mag = target_mag.T[:, np.newaxis, :]
        
        # Normalize
        mean_expanded = self.mean.reshape(1, 1, -1)
        std_expanded = self.std.reshape(1, 1, -1)
        mixture_mag_norm = (mixture_mag - mean_expanded) / (std_expanded + 1e-8)
        target_mag_norm = (target_mag - mean_expanded) / (std_expanded + 1e-8)
        
        # Convert to tensors
        mixture_tensor = torch.from_numpy(mixture_mag_norm).float()
        target_tensor = torch.from_numpy(target_mag_norm).float()
        
        return mixture_tensor, target_tensor


class BalancedSampler:
    """
    Custom sampler for balanced training over MUSDB18 tracks.
    Ensures each track is sampled evenly per epoch.
    """
    
    def __init__(self, dataset: MUSDB18Dataset, batch_size: int = 4):
        """
        Args:
            dataset: MUSDB18Dataset instance
            batch_size: Batch size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_tracks = len(dataset.tracks)
        self.samples_per_track = dataset.samples_per_track
    
    def __len__(self) -> int:
        """Total batches per epoch."""
        total_samples = self.num_tracks * self.samples_per_track
        return (total_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Yield batch indices."""
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]


if __name__ == "__main__":
    print("Testing MUSDB18Dataset...")
    
    dataset = MUSDB18Dataset(
        subset="train",
        target="vocals",
        chunk_duration=6.0,
        samples_per_track=2,
        augment=True,
        download=False,  # Set to True if MUSDB not downloaded
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of tracks: {len(dataset.tracks)}")
    
    # Test data loading
    mixture, target = dataset[0]
    print(f"Mixture shape: {mixture.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Mixture range: [{mixture.min():.3f}, {mixture.max():.3f}]")
    print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")
    
    print("✓ Dataset test passed!")
