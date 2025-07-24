"""
Description: utility functions (conversions, peak finding, synthesis)
Contributors: Simon Schwär, Sebastian Rosenzweig, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the Differentiable Intonation Tools
https://github.com/simonschwaer/differentiable-intonation-tools/
"""

import numpy as np

import librosa

import scipy.signal
from scipy.interpolate import PPoly, splev, splrep

import mir_eval


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

def c2f(c, a_ref=440.):
    """Convert cents difference to MIDI pitch 0 (C-1) to a frequency

    Parameters
    ----------
        c : float scalar or numpy array
            pitch in cents
        a_ref : float
            Reference frequency for MIDI pitch 69 in Hz (A4, default 440 Hz)

    Returns
    -------
        Cents difference of f to MIDI pitch 0 (C-1). The return value has the same dimension as f.
    """
    return a_ref * 2**((c - 6900) / 1200)


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


def find_peaks_harmonic(x, f0, fs, N=4096, H=2048, t_f0=None,
    max_harm=100, max_inharm=1.05, prominence_db=8, prominence_smoothing=1.2, abs_thrsh_db=-65,
    perform_hps=False, F_harm=20, F_perc=10):
    """Identify spectral peaks in an audio signal based on an initial F0 estimate.

    The function is searching for peaks near the integer multiples of F0
    using extremal values of interpolating spline of the spectral frame.

    Parameters
    ----------
        x : 1D np.ndarray
            Input audio signal
        f0 : 1D np.ndarray
            Frame-wise F0 estimates or annotations (see `t_f0` how this may be resampled to fit the analysis hop size
            of the peak finding)
        fs : float
            Sampling rate in Hz
        N : int
            Analysis window size in samples
        H : int
            Analysis hop size in samples
        t_f0 : None or 1D np.ndarray
            Time for each F0 estimate in `f0`. Has to have same length as `f0`.

            If this is given, the F0 trajectory is resampled to the exact analysis frames specified by
            the length of the signal `x` and the hop size `H`. Otherwise, the length of `f0` must correspond
            exactly to the number of analysis frames.
        max_harm : int
            Maximum number of harmonics to look for in the signal (also limited by Nyquist frequency)
        max_inharm : float
            Maximum allowed inharmonicity for each harmonic (defines the horizontal search range for the largest peak)
        prominence_db : float
            By how many dB a peak has to exceed the local threshold to be identified as a harmonic
        prominence_smoothing : float
            Factor to define the bandwidth of the local threshold smoothing (given as `factor * F0` for each frame)
        abs_thrsh_db : float
            Discard all harmonics that are below this absolute threshold in dB
        perform_hps: boolean
            Whether or not to pre-process the input signal with harmonic-percussive separation (HPS)
        F_harm : int
            Filter length for the harmonic median filter (only relevant when `perform_hps = True`)
        F_perc : int
            Filter length for the percussive median filter (only relevant when `perform_hps = True`)

    Returns
    -------
        t : 1D np.ndarray
            Time of each analysis frame in seconds
        P : 3D np.ndarray
            Peak frequencies and amplitudes for each time frame with dimensions (frames x max_harm x 2).
            Detected peaks are ordered by harmonic index and both frequency and amplitude are zero
            if no peak was detected.
        X : 2D np.ndarray
            Spectogram that was used to detect the peaks, possibly filtered with HPS

    """

    # calculate spectrogram
    win = scipy.signal.get_window("hann", N) # using flattop for optimal amplitude estimation
    X = np.abs(librosa.stft(x, n_fft=N, hop_length=H, center=False, window=win))
    X = X / np.sum(win) * 2 # normalize spectrum so that amplitudes correspond to actual sinusoid factors

    if perform_hps:
        # optionally use only harmonic part of the spectrogram
        X, _ = librosa.decompose.hpss(X, kernel_size=[hpss_filter_len, 32], margin=1.0)

    t = librosa.frames_to_time(range(X.shape[1]), hop_length=H, sr=fs)
    f_fft = np.fft.rfftfreq(N, 1/fs)

    # resample f0 trajectory
    if t_f0 is not None:
        voiced = (f0 > 0).astype(int)
        f0, _ = mir_eval.melody.resample_melody_series(t_f0, f0, voiced, t)

    L = len(f0)
    assert X.shape[1] == L, "Number of frames in F0 annotation does not match spectrogram size."

    P = np.zeros((L, max_harm, 2))

    for fr in range(L):
        if f0[fr] <= 0:
            continue # skip unvoiced frames

        # calculate a local threshold for harmonic prominence
        L_win = np.ceil(prominence_smoothing*f0[fr]/fs*N).astype(int)
        window = np.hanning(L_win + (1 - L_win % 2))
        window /= np.sum(window)
        thrsh = 20 * np.log10(
            np.convolve(np.pad(X[:,fr], (len(window)//2, len(window)//2)), window, mode="valid") + 1e-8
        )

        # find extremal values of interpolating spline of the spectral frame
        tck = splrep(f_fft, X[:,fr], k=3, s=0)
        ppoly = PPoly.from_spline(tck)
        X_fr_extrema = ppoly.derivative().roots(extrapolate=False)
        X_fr_extrema = np.append(X_fr_extrema, (f_fft[0], f_fft[-1]))

        for i in range(max_harm):
            f_test = (i + 1) * f0[fr]

            if f_test > fs/2: # we're above Nyquist
                continue

            f_min = (i + 1/max_inharm) * f0[fr]
            f_max = (i + max_inharm) * f0[fr]

            mask = np.where((X_fr_extrema >= f_min) & (X_fr_extrema <= f_max))
            if len(mask[0]) == 0: # no extrema in range, use integer multiple frequency
                P[fr,i,0] = f_test
                continue

            X_range = splev(X_fr_extrema[mask], tck)
            idx = np.argmax(X_range)

            P[fr,i,0] = X_fr_extrema[mask[0][idx]]

        mask = (P[fr, :, 0] > 0)
        P[fr,mask,1] = np.clip(splev(P[fr, mask, 0], tck), 0, np.inf)

        # remove harmonics that do not stand out enough
        nearest_bin = np.argmin(np.abs(P[fr,:,0,None] - f_fft[None,:]), axis=1)
        ampl_db = 20 * np.log10(P[fr,:,1] + 1e-8)
        mask = ((ampl_db - thrsh[nearest_bin]) < prominence_db) & (ampl_db < abs_thrsh_db)
        P[fr, mask,:] = 0

    return t, P, X


def find_peaks(x, fs=48000., N=4096, H=2048,
               max_peaks=16, hpss_filter_len=10, freq_lim=4000., **kwargs):
    """Identify spectral peaks in an audio signal

    Using 'scipy.signal.find_peaks', the function finds peaks in the (filtered) spectrogram of a signal and uses
    parabolic interpolation to refine the frequency resolution.

    Parameters
    ----------
        x : 1D np.ndarray
            Input audio signal
        fs : float
            Sampling rate in Hz
        N : int
            FFT size for each time frame in samples
        H : int
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
        t : 1D np.ndarray
            Time of each analysis frame in seconds
        P : 3D np.ndarray
            Peak frequencies and amplitudes for each time frame with dimensions (frames x max_harm x 2).
            Detected peaks are ordered by frequency first and amplitude second. If 'F < max_peaks' peaks are detected,
            the last 'max_peaks - F' frequencies and amplitudes are zero.
        X : 2D np.ndarray
            Spectogram that was used to detect the peaks, possibly filtered with HPS
    """

    X = librosa.stft(x, n_fft=N, hop_length=H, center=False)
    t = librosa.frames_to_time(range(X.shape[1]), hop_length=H, sr=fs)
    P = np.zeros((len(t), max_peaks, 2))

    # filter out percussive component and look for peaks only in harmonic part
    X, _ = librosa.decompose.hpss(X, kernel_size=[hpss_filter_len, 32], margin=1.0)

    # give lower weight to everything above given limit
    mi = int(np.round(freq_lim / fs * N))
    X[mi:,:] *= 0.001 # - 60 dB

    for i in range(len(t)):
        peaks = _find_peaks_single(X[:,i], N, fs, max_peaks, **kwargs)

        if (len(peaks) > 0):
            P[i, :len(peaks), :] = peaks

    return t, P, X


def synth(f0,
          duration=1.,
          fs=48000.,
          waveform='sawtooth',
          num_harmonics=16,
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
    if isinstance(waveform, str):
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
            magnitudes[1:,1] = np.array([2/np.pi * (-1)**n / n**1.5 for n in np.arange(1, num_harmonics)])
            magnitudes[:,1] *= 0.5
        elif waveform == 'flat':
            magnitudes = np.ones((num_harmonics, 2))
            magnitudes[:,0] = np.arange(1, num_harmonics+1)
            magnitudes[:,1] *= 0.5
        else:
            raise ValueError("Unknown waveform shape.")
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
    H_scale = np.sum(scipy.signal.get_window("hann", fft_size)) / 2
    H_db = np.clip(20*np.log10(H_mag + 0.00001), -90, 1000) # adding -100dB const to avoid log(0)

    sig_rms = np.sqrt(np.mean(np.square(H_mag))/fft_size) # rms of harmonic part

    maxima, _ = scipy.signal.find_peaks(H_db, **kwargs)

    peaks = []
    for i in maxima:
        # use parabolic interpolation to find true peak and save frequency val
        k = i + (H_db[i-1] - H_db[i+1]) / (2 * (H_db[i-1] - 2 * H_db[i] + H_db[i+1]))
        peaks.append((fs*k/fft_size, H_mag[i]/H_scale))

    peaks.sort(key=lambda tup: tup[1], reverse=True) # sort by amplitude (highest first)
    peaks = peaks[:max_peaks] # truncate
    peaks.sort(key=lambda tup: tup[0]) # sort by frequency (lowest first)

    return peaks