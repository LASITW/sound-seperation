import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import soundfile as sf
import argparse
import logging
import openunmix

from model import OpenUnmixLSTM, AudioProcessor
from dataset import MUSDB18Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Separator:
    """
    Audio source separator using pretrained Open-Unmix (umxl) model.
    Handles inference, reconstruction, and file I/O.
    """

    TARGETS = ['vocals', 'drums', 'bass', 'other']

    def __init__(
        self,
        checkpoint_path: str = None,  # kept for CLI/API compatibility; not used
        target: str = 'vocals',
        n_fft: int = 4096,
        hop_length: int = 1024,
        device: str = 'cpu',
    ):
        """
        Args:
            checkpoint_path: Unused; kept for backward compatibility
            target: Target source to return from separate()
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.target = target

        # Load pretrained Open-Unmix umxl weights for all 4 stems
        self.model = openunmix.umxl(
            targets=self.TARGETS,
            residual=False,
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("✓ Loaded pretrained Open-Unmix umxl model (vocals/drums/bass/other)")
    
    def _load_audio_stereo(self, audio_path: str) -> np.ndarray:
        """Load audio file and return (channels, samples) stereo float32 array at 44100 Hz."""
        audio, sr = sf.read(audio_path)
        if sr != 44100:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)  # (2, samples)
        else:
            audio = audio.T  # (samples, channels) -> (channels, samples)
        return audio.astype(np.float32)

    @torch.no_grad()
    def separate(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Separate self.target source from audio file.

        Returns:
            separated_audio: (channels, samples) float32 array
            sample_rate: 44100
        """
        audio = self._load_audio_stereo(audio_path)  # (channels, samples)
        logger.info(f"Audio shape: {audio.shape}")

        # (channels, samples) -> (1, channels, samples)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        # (1, n_targets, channels, samples)
        out = self.model(audio_tensor)

        target_idx = self.TARGETS.index(self.target)
        separated = out[0, target_idx].cpu().numpy()  # (channels, samples)

        logger.info(f"Separated audio shape: {separated.shape}")
        return separated, 44100

    @torch.no_grad()
    def separate_all_from_audio(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate all 4 stems in a single model pass.

        Args:
            audio: (channels, samples) float32 stereo array at 44100 Hz

        Returns:
            Dict mapping stem name -> (channels, samples) float32 array
        """
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)
        out = self.model(audio_tensor)  # (1, 4, channels, samples)
        return {
            stem: out[0, i].cpu().numpy()
            for i, stem in enumerate(self.TARGETS)
        }
    
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


def evaluate_on_musdb(musdb_path=None, output_dir='./eval_results', checkpoint='./checkpoints/best_model.pt'):
    """
    Evaluate model on MUSDB18 test set using museval metrics.
    """
    import museval
    import musdb
    
    logger.info("Evaluating on MUSDB18 test set...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator(
        checkpoint_path=checkpoint,
        target='vocals',
        device=str(device),
    )
    
    # Load test set
    mus_kwargs = {'subsets': 'test'}
    if musdb_path:
        mus_kwargs['root'] = musdb_path
    mus = musdb.DB(**mus_kwargs)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_scores = []
    
    for track in mus:
        logger.info(f"Processing: {track.name}")

        # track.audio is (samples, channels) — convert to (channels, samples) for the model
        mixture = track.audio.T.astype(np.float32)  # (channels, samples)

        # Single model pass for all 4 stems
        all_stems = separator.separate_all_from_audio(mixture)

        # museval expects (samples, channels)
        estimates = {stem: audio.T for stem, audio in all_stems.items()}
        
        # Evaluate using museval
        scores = museval.eval_mus_track(
            track,
            estimates,
            output_dir=str(output_dir / track.name)
        )
        
        all_scores.append(scores)
        # scores.scores["targets"] is a list of {name, frames} dicts
        for target_data in scores.scores['targets']:
            target = target_data['name']
            sdr_values = [f['metrics']['SDR'] for f in target_data['frames'] if f['metrics']['SDR'] is not None]
            median_sdr = float(np.nanmedian(sdr_values)) if sdr_values else float('nan')
            logger.info(f"  {target} SDR: {median_sdr:.2f} dB")
    
    logger.info("✓ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Audio source separation inference')
    parser.add_argument('--checkpoint', type=str, default='./code/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to input audio file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output audio file')
    parser.add_argument('--target', type=str, default='vocals',
                       choices=['vocals', 'drums', 'bass', 'other'],
                       help='Target source to separate')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on MUSDB18 test set')
    parser.add_argument('--musdb_path', type=str, default=None,
                       help='Path to MUSDB18 dataset root')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_on_musdb(musdb_path=args.musdb_path, output_dir=args.output_dir, checkpoint=args.checkpoint)
    else:
        if not args.audio or not args.output:
            parser.error('--audio and --output are required when not using --evaluate')
        separator = Separator(
            checkpoint_path=args.checkpoint,
            target=args.target,
            device=args.device,
        )
        separator.separate_and_save(args.audio, args.output)


if __name__ == "__main__":
    main()
