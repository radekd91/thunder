from .PreprocessorBase import PreprocessorBase
import numpy as np
import torch
import torchaudio
import soundfile as sf
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.signal import get_window
from librosa.util import pad_center, tiny
import librosa.util as librosa_util


class MelPreprocessor(PreprocessorBase):

    def __init__(self, 
                filter_length = 640 ,
                hop_length = 160,
                win_length = 640,
                n_mel_channels = 80,
                sampling_rate = 16000,
                mel_fmin = 0.0,
                mel_fmax = 8000.0 ,
                #  n_mels, n_fft, hop_length, win_length, sample_rate
                strict = False,
                input_key="raw_audio", 
                output_key="gt_mel",
                **kwargs,
                 ):
        super().__init__()
        self.filter_length = filter_length 
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        self.stft_fn = STFT(filter_length, hop_length, win_length)

        mel_basis = librosa_mel_fn(
                sr=self.sampling_rate, n_fft=self.filter_length, n_mels=self.n_mel_channels, fmin=self.mel_fmin, fmax=self.mel_fmax)
        self.mel_basis = torch.from_numpy(mel_basis).float()
        self.strict = strict 
        self.input_key = input_key
        self.output_key = output_key
        
    def window_hop_ratio(self):
        return self.win_length // self.hop_length

    def forward(self, batch, train=False, validation=False, 
                input_key=None, output_key=None, **kwargs):
        input_key = input_key or self.input_key
        output_key = output_key or self.output_key
        waveform = batch[input_key]
        if waveform.dtype != torch.float:
            waveform = waveform.float()
        B, T, F = waveform.shape[:3]
        waveform = waveform.view(B, T * F)
        mel_output = self._mel_spectrogram(waveform).transpose(1, 2) 
        T_mel = mel_output.shape[1]
        # if T_mel / self.window_hop_ratio() < T:
        T_mel_trim = (T_mel // self.window_hop_ratio()) * self.window_hop_ratio()
        mel_output = mel_output[:, :T_mel_trim, :]
        batch[output_key] = mel_output
        return batch

    def _mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        if self.strict:
            assert(torch.min(y.data) >= -1)
            assert(torch.max(y.data) <= 1)
        else:
            min_val = torch.min(y.data)
            max_val = torch.max(y.data)
            if min_val < -1 or max_val > 1:
                print(f"[WARNING]: Min value {min_val} or Max value {max_val} out of range [-1, 1]")
            y = torch.clamp(y, -1, 1)


        magnitudes, phases = self.stft_fn.transform(y)
        # magnitudes = magnitudes.data

        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = spectral_normalize(mel_output)
        return mel_output

    def to(self, device):
        self.mel_basis = self.mel_basis.to(device)
        self.stft_fn = self.stft_fn.to(device)
        return self

    @property
    def device(self):
        return self.mel_basis.device
    


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output


def mel_spectrogram(y, 
                    filter_length=640,
                    hop_length=160,
                    win_length=640,
                    sampling_rate=16000,
                    n_mel_channels=80,
                    mel_fmin=0.0,
                    mel_fmax=8000.0
                    ):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)
    stft_fn = STFT(filter_length, hop_length, win_length)
    stft_fn = stft_fn.cuda()
    
    magnitudes, phases = stft_fn.transform(y)
    magnitudes = magnitudes.data

    mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float().cuda()


    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_output = spectral_normalize(mel_output)
    return mel_output