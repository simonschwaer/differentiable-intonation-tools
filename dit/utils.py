"""
Description: utility functions (conversions, peak finding, synthesis)
Contributors: Simon Schwär, Sebastian Rosenzweig, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the Differentiable Intonation Tools
https://github.com/simonschwaer/differentiable-intonation-tools/
"""

import numpy as np

from librosa import stft as librosa_stft
from librosa import frames_to_time as librosa_frames_to_time
from librosa.decompose import hpss as librosa_hpss

from scipy.signal import find_peaks as scipy_find_peaks


def f2c(f, a_ref=440.):
    """Convert a frequency to cents w.r.t. a reference

    Parameters
    ----------
        f : float scalar or numpy array
            Frequency in Hz
        a_ref : float
            Reference frequency for MIDI pitch 69 in Hz (A4, default 440 Hz)

    Returns
    -------
        Cents difference of f to MIDI pitch 0 (C-1). The return value has the same dimension as f.
    """
    return 1200 * np.log2(f/a_ref) + 6900


def f2s(f, a_ref=440.):
    """Convert a frequency to string in MIDI-like format

    Parameters
    ----------
        f : float scalar
            Frequency in Hz
        a_ref : float
            Reference frequency for MIDI pitch 69 in Hz (A4, default 440 Hz)

    Returns
    -------
        String describing the given frequencies in terms of MIDI-like pitches (e.g. "Bb3 -7c")
    """
    pitch = f2c(f, a_ref)/100

    whole = int(np.round(pitch))
    cents = int(np.round(pitch - whole, 2) * 100)
    octave = int(np.floor(whole / 12.)) - 1

    detune_str = ""
    if cents > 0:
        detune_str = " +" + str(cents) + "c"
    elif cents < 0:
        detune_str = " " + str(cents) + "c"

    return _note_names[whole % 12] + str(octave) + detune_str


def s2f(s, a_ref=440., detune=0):
    """Convert a string in MIDI-like format (as given by 'f2s', but without the detuning in cents) to frequency

    Parameters
    ----------
        s : string
            MIDI-like string (e.g. "Bb3")
        a_ref : float
            Reference frequency for MIDI pitch 69 in Hz (A4, default 440 Hz)

    Returns
    -------
        12-tone equal temperament frequency in Hz
    """
    steps = _note_names.index(s[0].upper())
    if len(s) > 2 or (len(s) == 2 and not s[-1].isnumeric()): # we have a sharp or flat modifier
        for i in range(1, len(s)):
            if (s[i] == '#'):
                steps += 1
            elif (s[i] == 'b'):
                steps -= 1
    octave = int(s[-1]) if s[-1].isnumeric() else 4
    
    return a_ref * np.power(2, steps/12. + octave - 69./12 + 1) * np.power(2, detune / 1200.)


def find_peaks(x,
               fs=48000., 
               fft_size=4096,
               hop_size=2048,
               max_peaks=16,
               hpss_filter_len=10,
               freq_lim=4000.,
               **kwargs):
    """Identify spectral peaks in an audio signal

    Using 'scipy.signal.find_peaks', the function finds peaks in the (filtered) spectrogram of a signal and uses
    parabolic interpolation to refine the frequency resolution.

    Parameters
    ----------
        x : 1D float numpy array
            Input audio signal
        fs : float
            Sampling rate in Hz
        fft_size : int
            FFT size for each time frame in samples
        hop_size : int
            Hop size for each time frame in samples
        max_peaks : int
            Maximum number of peaks per time frame
        hpss_filter_len : int
            Length of the harmonic-percussive separation median filter (longer filter suppresses transients more strongly)
        freq_lim : float
            Frequency in Hz above which the spectrogram is multiplied with a small constant to suppress peaks
        kwargs
            Extra arguments for 'scipy.signal.find_peaks'

    Returns
    -------
        t : 1D float numpy array (dimensions: (T))
            Time of each frame in seconds. Dimension T depends on the signal length and the FFT hop size. 
        P : 3D float numpy array (dimensions: (T, max_peaks, 2))
            Peak frequencies and amplitudes for each time frame. Detected peaks are ordered by frequency first and
            amplitude second. If 'F < max_peaks' peaks are detected, the last 'max_peaks - F' frequencies and amplitudes
            are zero.
        H : 2D complex float array (dimensions: (T, ceil(fft_size/2)))
            Spectogram that was used to detect the peaks. It has been filtered with 'librosa.decompose.hpss' to remove
            transient components. Also, frequencies above 'freq_lim' are suppressed.
    """

    H_STFT = librosa_stft(x, n_fft=fft_size, hop_length=hop_size, center=False)
    t = librosa_frames_to_time(range(H_STFT.shape[1]), hop_length=hop_size, sr=fs)
    P = np.zeros((len(t), max_peaks, 2))

    # filter out percussive component and look for peaks only in harmonic part
    H_STFT, _ = librosa_hpss(H_STFT, kernel_size=[hpss_filter_len, 32], margin=1.0)

    # give lower weight to everything above given limit
    mi = int(np.round(freq_lim / fs * fft_size))
    H_STFT[mi:,:] *= 0.001 # - 60 dB

    for i in range(len(t)):
        peaks = _find_peaks_single(H_STFT[:,i], fft_size, fs, max_peaks, **kwargs)

        if (len(peaks) > 0):
            P[i, :len(peaks), :] = peaks

    return t, P, H_STFT


def synth(f0,
          duration=1.,
          fs=48000.,
          waveform='sawtooth',
          num_harmonics=8,
          vib_rate=0.,
          vib_depth=10,
          init_phase=[]):
    """Generate a tone with given harmonics as a time-domain signal

    Parameters
    ----------
        f0 : float
            Fundamental frequency in Hz
        duration : float
            Length of the output sequence in seconds
        fs : float
            Sampling rate  
        waveform : string or 1D float numpy array
            Either a waveform string (one of 'square', 'triangle', 'sawtooth') or a (Nx2) numpy array
            containing multipliers and magnitudes of harmonics
        num_harmonics : int
            number of harmonics if 'waveform' is a string
        vib_rate : float
            Rate of pitch change in Hz (<=0 for no vibrato)
        vib_depth : float
            Depth of the vibrato in cents (only if rate > 0)
        init_phase : list
            Initial phase of oscillators as returned by a previous call to this function
            (optional to allow continuous synthesis with different tones)
    
    Returns
    -------
        signal : 1D float numpy array
            The synthesized signal
        phase_carry : list
            Can be used as argument 'init_phase' for the next call to this function, so that there is no
            phase discontinuity between the two contiguous synthesized tones
            (only works when the harmonics don't change between calls)
    """

    if waveform == 'square':
        magnitudes = np.zeros((num_harmonics, 2))
        magnitudes[:,0] = np.arange(1, num_harmonics+1)
        magnitudes[2::2,1] = np.array([1. / (n+1) for n in np.arange(2, num_harmonics, 2)])
        magnitudes[0,1] = 1
        magnitudes[:,1] *= 0.5
    elif waveform == 'triangle':
        magnitudes = np.ones((num_harmonics, 2))
        magnitudes[:,0] = np.arange(1, num_harmonics+1)
        magnitudes[:,1] = np.array([8/(np.pi**2) * (-1)**int(n/2.) * n**(-2.) for n in np.arange(1, num_harmonics)])
        magnitudes[1::2,1] = 0
        magnitudes[:,1] *= 0.5
    elif waveform == 'sawtooth':
        magnitudes = np.ones((num_harmonics, 2))
        magnitudes[:,0] = np.arange(1, num_harmonics+1)
        magnitudes[1:,1] = np.array([2/np.pi * (-1)**n / n for n in np.arange(1, num_harmonics)])
        magnitudes[:,1] *= 0.5
    else:
        magnitudes = np.asarray(waveform)
        assert len(magnitudes.shape) == 2 and magnitudes.shape[1] == 2, "Custom waveform must be a Nx2 numpy array."

    t = np.arange(0, duration, step=1/fs)
    sig = np.zeros(t.shape)

    vib = np.ones(t.shape)
    if vib_rate > 0:
        vib = np.power(2, (vib_depth * np.sin(2 * np.pi * vib_rate * t))/1200)
    
    phase_carry = []
    i = 0
    for h in magnitudes:
        f = vib * f0 * h[0]
        delta_phase = 2 * np.pi * f * 1/fs
        p_start = 0 if len(init_phase) <= i else init_phase[i]
        phase = np.cumsum(delta_phase) + p_start
        sig += h[1] * np.sin(phase)
        phase_carry.append(phase[-1] % (2 * np.pi))
        i += 1

    return sig, phase_carry




_note_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

def _find_peaks_single(H, fft_size, fs, max_peaks, **kwargs):
    """Helper function to detect spectral peaks in a single FFT spectrum
    """

    # convert to log magnitude spectrum
    H_mag = np.abs(H)
    H_db = np.clip(20*np.log10(H_mag + 0.00001), -90, 1000) # adding -100dB const to avoid log(0)

    sig_rms = np.sqrt(np.mean(np.square(H_mag))/fft_size) # rms of harmonic part

    maxima, _ = scipy_find_peaks(H_db, **kwargs)

    peaks = []
    for i in maxima:
        # use parabolic interpolation to find true peak and save frequency val
        k = i + (H_db[i-1] - H_db[i+1]) / (2 * (H_db[i-1] - 2 * H_db[i] + H_db[i+1]))
        peaks.append((fs*k/fft_size, H_mag[i]/np.max(H_mag)))

    peaks.sort(key=lambda tup: tup[1], reverse=True) # sort by amplitude (highest first)
    peaks = peaks[:max_peaks] # truncate
    peaks.sort(key=lambda tup: tup[0]) # sort by frequency (lowest first)

    return peaks