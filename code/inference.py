import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import argparse
import logging

from model import OpenUnmixLSTM, AudioProcessor
from dataset import MUSDB18Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Separator:
    """
    Audio source separator using trained Open-Unmix model.
    Handles inference, reconstruction, and file I/O.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        target: str = 'vocals',
        n_fft: int = 4096,
        hop_length: int = 1024,
        device: str = 'cpu',
    ):
        """
        Args:
            checkpoint_path: Path to saved model checkpoint
            target: Target source to separate
            n_fft: FFT size
            hop_length: Hop length for STFT
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.target = target
        
        # Load model
        self.model = OpenUnmixLSTM(
            input_size=1 + n_fft // 2,
            hidden_size=512,
            num_layers=3,
            num_channels=2,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Loaded model from {checkpoint_path}")
        
        # Audio processor
        self.processor = AudioProcessor(
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=44100,
        )
    
    @torch.no_grad()
    def separate(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Separate source from audio file.
        
        Args:
            audio_path: Path to input audio file
        
        Returns:
            separated_audio: Separated source audio (stereo)
            sample_rate: Sample rate (44100 Hz)
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Resample if needed
        if sr != 44100:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        else:
            audio = audio.T  # (samples, channels) -> (channels, samples)
        
        logger.info(f"Audio shape: {audio.shape}")
        
        # Convert to spectrogram
        mixture_mag, mixture_phase = self.processor.audio_to_magnitude_spectrogram(audio)
        
        # Normalize and convert to tensor
        # (channels, freq_bins, frames) -> (frames, channels, freq_bins)
        mixture_mag_t = np.transpose(mixture_mag, (2, 0, 1))
        
        # Load normalization from model
        mean = self.model.input_mean.cpu().numpy().squeeze()
        std = self.model.input_std.cpu().numpy().squeeze()
        
        mixture_mag_norm = (mixture_mag_t - mean) / (std + 1e-8)
        mixture_tensor = torch.from_numpy(mixture_mag_norm).float().unsqueeze(0).to(self.device)
        
        logger.info(f"Mixture spectrogram shape: {mixture_tensor.shape}")
        
        # Predict mask
        mask = self.model(mixture_tensor)
        mask = mask.squeeze(0).cpu().numpy()  # (frames, channels, freq_bins)
        
        # Apply mask to get separated magnitude
        separated_mag = mixture_mag * mask.T[np.newaxis, :, :]  # Transpose back
        
        # Reconstruct audio using original phase
        separated_audio = self.processor.magnitude_spectrogram_to_audio(
            separated_mag,
            mixture_phase
        )
        
        logger.info(f"Separated audio shape: {separated_audio.shape}")
        
        return separated_audio, 44100
    
    def separate_and_save(
        self,
        audio_path: str,
        output_path: str,
    ):
        """
        Separate source and save to file.
        
        Args:
            audio_path: Input audio file
            output_path: Output audio file path
        """
        separated_audio, sr = self.separate(audio_path)
        
        # Ensure output is in correct format (channels, samples) -> (samples, channels)
        if separated_audio.ndim == 2:
            separated_audio = separated_audio.T
        
        # Save
        sf.write(output_path, separated_audio, sr)
        logger.info(f"✓ Saved separated audio to {output_path}")


def evaluate_on_musdb():
    """
    Evaluate model on MUSDB18 test set using museval metrics.
    """
    import museval
    import musdb
    
    logger.info("Evaluating on MUSDB18 test set...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator(
        checkpoint_path='./checkpoints/best_model.pt',
        target='vocals',
        device=str(device),
    )
    
    # Load test set
    mus = musdb.DB(subsets='test')
    
    # Create output directory
    output_dir = Path('./eval_results')
    output_dir.mkdir(exist_ok=True)
    
    all_scores = []
    
    for track in mus:
        logger.info(f"Processing: {track.name}")
        
        # Separate all sources
        estimates = {}
        
        for source in ['vocals', 'drums', 'bass', 'other']:
            separator.target = source
            
            # Create temp audio file for this track's mixture
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, track.audio, 44100)
                temp_path = tmp.name
            
            try:
                separated_audio, _ = separator.separate(temp_path)
                estimates[source] = separated_audio
            finally:
                Path(temp_path).unlink()
        
        # Evaluate using museval
        scores = museval.eval_mus_track(
            track,
            estimates,
            output_dir=str(output_dir / track.name)
        )
        
        all_scores.append(scores)
        logger.info(f"  Vocals SDR: {scores[0]['vocal'][0]:.2f}")
    
    logger.info("✓ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Audio source separation inference')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to input audio file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output audio file')
    parser.add_argument('--target', type=str, default='vocals',
                       choices=['vocals', 'drums', 'bass', 'other'],
                       help='Target source to separate')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on MUSDB18 test set')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_on_musdb()
    else:
        separator = Separator(
            checkpoint_path=args.checkpoint,
            target=args.target,
            device=args.device,
        )
        separator.separate_and_save(args.audio, args.output)


if __name__ == "__main__":
    main()
