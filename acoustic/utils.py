import librosa
import torch
import torch.nn.functional as F
import matplotlib

import torchaudio.transforms as transforms

matplotlib.use("Agg")
import matplotlib.pylab as plt


class Metric:
    def __init__(self):
        self.steps = 0
        self.value = 0

    def update(self, value):
        self.steps += 1
        self.value += (value - self.value) / self.steps
        return self.value

    def reset(self):
        self.steps = 0
        self.value = 0


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.melspctrogram = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=False,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=128,
            mel_scale="slaney",
        )

    def forward(self, wav):
        padding = (1024 - 160) // 2
        wav = F.pad(wav, (padding, padding), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def save_checkpoint(
    checkpoint_dir,
    acoustic,
    optimizer,
    step,
    loss,
    best,
    logger,
):
    state = {
        "acoustic-model": acoustic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(
    load_path,
    acoustic,
    optimizer,
    rank,
    logger,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    acoustic.load_state_dict(checkpoint["acoustic-model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))
    return step, loss


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def extract_pitch(y, sr, hop_length, fmin, fmax):
    """
    Conduct pitch tracking. Return normalized pitch (0.0 ~ 1.0) and pitch mask where 1.0 corresponds to
    valid pitches and 0.0 corresponds to invalid pitches. 
    """
    y = y.numpy()
    pitches, _ = librosa.piptrack(y, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
    pitches[pitches == 0.0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitches = np.nanmin(pitches, axis=0)
    pitches[np.isnan(pitches)] = 0.0  # All-nan columns result in nan pitch. Use 0.0 for these values.
    pitch_mask = np.where(pitches > 0.0, 1.0, 0.0)
    normalized_pitches = np.clip((pitches - PITCH_FMIN) / (PITCH_FMAX - PITCH_FMIN), 0.0, 1.0)
    normalized_pitches = torch.from_numpy(normalized_pitches)
    return normalized_pitches