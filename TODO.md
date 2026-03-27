1. implement a training algorithm

2. convert audio signals to stft and then magnitude spectograms // whatever else I need
- look into open source

3. Generate new datasets from musdb 
- maybe seperate per time signature ie bpm and time signature (120 bpm in 4/4 ) would let us know how long to wait before splitting/rearranging and keep it musical, is there a better way 
***
while True:
    track = random.choice(mus.tracks)
    track.chunk_duration = 5.0
    track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
    x = track.audio.T
    y = track.targets['vocals'].audio.T
    yield x, y
****

4. - create evaluation metrics
****
import museval
# provide an estimate
estimates = {
    'vocals': np.random.random(track.audio.shape),
    'accompaniment': np.random.random(track.audio.shape)
}
# evaluates using BSSEval v4, and writes results to `./eval`
print(museval.eval_mus_track(track, estimates, output_dir="./eval")
*****



Sources:
MUSDB: https://github.com/sigsep/sigsep-mus-db


EXISTING ARCHITECTURES:
OPEN UNMIX: 

To perform separation into multiple sources, Open-unmix comprises multiple models that are trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source.

Each Open-Unmix source model is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target source, like vocals, from the magnitude spectrogram of a mixture input. Internally, the prediction is obtained by applying a mask on the input. The model is optimized in the magnitude domain using mean squared error.

