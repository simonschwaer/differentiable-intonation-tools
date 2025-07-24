"""
Description: higher-level intonation analysis functions
Contributors: Simon Schwär, Sebastian Rosenzweig, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the Differentiable Intonation Tools
https://github.com/simonschwaer/differentiable-intonation-tools/
"""

import numpy as np

from .cost import tonal_for_frames, harmonic_for_frames

def adapt_multivoice(P, w=0.33, mu=50, reg=0, mom=0.9, stop_it=1, stop_mag=0.0, start_from_prev=True,
                     skip_voices=None, unvoiced_thrsh=-60, kwargs_tonal={}, kwargs_harmonic={}):
    """Calculate pitch-shifts for adapting multiple voices simultaneously

    Parameters
    ----------
        P : 4D np.ndarray or list of 3D np.ndarray
            (V, L, M, 2) numpy array with V voices, N time frames and M frequency/amplitude
            pairs (f_n, a_n), as for example returned by 'utils.find_peaks'.
        w : float
            relative weighting of tonal vs. harmonic cost
        mu : float
            Step size for gradient descent (default: 50)
        reg : float
            Regularization factor to avoid voices drifting apart from each other (default: 0.0)
        mom : float
            Parameter for momentum in the gradient calculation (default: 0.9)
        stop_it : int
            number of gradient descent iterations per frame and voice (default: 1)
        stop_mag : float
            minimum magnitude of the gradient at which the gradient iterations should stop (default: 0.0)
        start_from_prev : bool

        skip_voices : list of int or None
            Optional list of voice indices to skip during processing (e.g., to keep one voice fixed, default: None)
        unvoiced_thrsh : float
            Threshold in dB at which a frame is considered to be unvoiced and is skipped
            (considering the sum of amplitudes in P, default: -60 dB)
        kwargs_tonal : dict
            keyword arguments to be forwarded to `cost.tonal_for_frames` (will overwrite the defaults used)
        kwargs_harmonic : dict
            keyword arguments to be forwarded to `cost.harmonic_for_frames` (will overwrite the defaults used)

    Returns
    -------
        (V, N) array of resulting pitch shifts for N time frames and V voices

    """
    V = len(P) # number of voices
    N = P[0].shape[0] # number of frames

    p_shift = np.zeros((V, N))
    d_mom = np.zeros(V) # separate momentum store for each voice

    for n in range(N):
        for v in range(V):
            if skip_voices is not None and v in skip_voices:
                continue

            if start_from_prev:
                p_shift[v,n] = p_shift[v,n-1] # start with shift value from previous frame

            if ((20 * np.log10(np.sum(P[v][n,:,1]) + 1e-8)) < unvoiced_thrsh): # skip if this voice is unvoiced
                continue

            i = 0 # gradient descent iteration count
            dp = np.inf

            while (i < stop_it) and (np.abs(dp) > stop_mag):
                P_cur = P[v][[n]].copy()
                P_cur[...,0] *= np.power(2, p_shift[v,n] / 1200)
                P_acc = []
                for v2 in range(V):
                    if v2 == v:  continue
                    P_a = P[v2][[n]].copy()
                    P_a[...,0] *= np.power(2, p_shift[v2,n-1] / 1200) # accompaniment based on previous frame shift
                    P_acc.append(P_a)
                P_acc = np.concatenate(P_acc, axis=1)

                t_kwargs_defaults = {
                    "K": 12,
                    "f_ref": 440.,
                    "fit_grid": False,
                }
                t_kwargs = {}
                t_kwargs.update(t_kwargs_defaults)
                t_kwargs.update(kwargs_tonal)

                h_kwargs_defaults = {
                    "log_mag_weights": True,
                    "log_mag_gamma": 1,
                    "norm": "full_sum",
                    "ampl_method": "min",
                    "berezovsky_erb": True,
                }
                h_kwargs = {}
                h_kwargs.update(h_kwargs_defaults)
                h_kwargs.update(kwargs_harmonic)

                dp_t = tonal_for_frames(P_cur[:,[0],:], P_acc[:,[0],:], gradient=True, **t_kwargs)
                dp_h = 30 * harmonic_for_frames(P_cur, P_acc, gradient=True, **h_kwargs)
                d_reg = p_shift[v,n] - np.mean(p_shift[:,n-1])

                dp = w * dp_t + (1 - w) * dp_h
                d_mom[v] = mom * d_mom[v] + (1 - mom) * dp

                p_shift[v, n] = p_shift[v, n] - mu * d_mom[v] - reg * d_reg

                i += 1

    return p_shift