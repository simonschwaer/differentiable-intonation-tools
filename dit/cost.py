"""
Description: cost measurement functions
Contributors: Simon Schwär, Sebastian Rosenzweig, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the Differentiable Intonation Tools
https://github.com/simonschwaer/differentiable-intonation-tools/
"""

import numpy as np


def tonal(f, K=12, f_ref=440., gradient=False):
    """Tonal cost for frequencies deviating from an equal-temperament (ET) grid
    
    Parameters
    ----------
        f : float scalar or 1D numpy array 
            Frequencies of interest in Hz
        K : int
            Number of ET divisions per octave (default: 12)
        f_ref : float
            Reference frequency for alignment of the the ET grid in Hz (default: 440 Hz)
        gradient : bool
            Whether or not to return the cost value (default) or the gradient at f
    
    Returns
    -------
        The return value has the same dimension as f, always as a numpy array, even when f is a scalar.
        if gradient is False:
            Cost between 0 and 1 at f
            (0 := exactly on ET grid, 1 := exactly between two ET grid frequencies)
        if gradient is True:
            Cost gradient at f with unit 'cost change per cent shifted'
    """
    f_arr = np.atleast_1d(np.asarray(f, dtype=np.float32))

    if not gradient:
        return (1 - np.cos(2 * np.pi * K * np.log2(f_arr/f_ref))) / 2
    else:
        return np.pi * K * np.sin(2 * np.pi * K * np.log2(f_arr/f_ref)) / 1200


def harmonic(f1, f2, kernel="berezovsky", berezovsky_erb=False, max_at_erb=0.25, gradient=False, **kwargs):
    """Harmonic cost between frequencies based on Plomp/Levelt perceptual dissonance

    Parameters
    ----------
        f1 : np.ndarray
            Frequencies of first set in Hz (dimensions have to be broadcastable with `f2`)
        f2 : np.ndarray
            Frequencies of second set in Hz (dimensions have to be broadcastable with `f1`)
        kernel : str
            kernel type with options "berezovsky" (default), "bigand", "marjieh" (see notes for references)
        berezovsky_erb : bool
            Whether or not to use the original maximum parametrization from Berezovsky 2019 or a modified formula
            which allows to control the position of the maximum in terms of ERBs (see `max_at_erb`)
        max_at_erb : float
            If `kernel = "berezovsky"` and `berezovsky_erb = True`, set the position of maximum dissonance expressed
            in terms of the equivalent rectangular bandwidth (ERB)
        gradient : bool
            Whether or not to return the cost value (default) or the gradient at f

    Returns
    -------
        res : np.ndarray
            if gradient is False:
                The pairwise harmonic cost between f1 and f2 with arbitrary unit
                (maximum of 1 is reached when log2(f1/f2) = wc)
            if gradient is True:
                The pairwise harmonic cost gradient w.r.t f1 with unit 'cost change per cent shifted'
    Notes
    -----
        - J. Berezovsky, “The structure of musical harmony as an ordered phase of sound:
          A statistical mechanics approach to music theory,” Science Advances, vol. 5,
          p. eaav8490, May 2019.
        - E. Bigand, R. Parncutt, and F. Lerdahl, “Perception of musical tension in short chord sequences:
          The influence of harmonic function, sensory dissonance, horizontal motion, and musical training,”
          Perception & Psychophysics, vol. 58, no. 1, pp. 125–141, Jan. 1996, doi: 10.3758/BF03205482.
        - R. Marjieh, P. M. C. Harrison, H. Lee, F. Deligiannaki, and N. Jacoby, “Timbral effects on consonance
          disentangle psychoacoustic mechanisms and suggest perceptual origins for musical scales,”
          Nat Commun, vol. 15, no. 1, p. 1482, Feb. 2024, doi: 10.1038/s41467-024-45812-z.

    """

    f1 = np.atleast_1d(np.asarray(f1, dtype=np.float32))
    f2 = np.atleast_1d(np.asarray(f2, dtype=np.float32))

    f_m = np.add(f1, f2) / 2

    if kernel in ["berezovsky"]:
        with np.errstate(divide='ignore', invalid='ignore'): # supress warning of log of zero
            ratio = np.where(f2 > 0, f1 / f2, 0) # np.divide(f1, f2, where=(f2 != 0))
            d = np.where(ratio > 0, np.log2(ratio), 0)
    elif kernel in ["marjieh", "bigand"]:
        d = np.zeros(f_m.shape)
        mask_d = (f_m > 0)
        d[mask_d] = np.abs(np.subtract(f1, f2))[mask_d] / 1.72 * f_m[mask_d]**(-0.65)
    else:
        raise ValueError(f"Unknown kernel type '{kernel}'.")

    if kernel == "berezovsky":
        if berezovsky_erb:
            erb = np.zeros(f_m.shape)
            mask_erb = (f_m > 0)
            erb[mask_erb] = np.log2(1 + 24.7 * ((4.37 * f_m[mask_erb]) / 1000 + 1) / f_m[mask_erb])
            d_max = max_at_erb * erb
        else:
            f_min = np.minimum(f1, f2)
            d_max = np.zeros(f_min.shape)
            mask = f_min > 0
            d_max[mask] = 0.67 * f_min[mask]**(-0.68)
        res = _kernel_berezovsky(d, gradient, d_max, **kwargs)
    elif kernel == "marjieh":
        res = _kernel_marjieh(d, gradient, **kwargs)
    elif kernel == "bigand":
        res = _kernel_bigand(d, gradient, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type '{kernel}'.")

    # account for the gradient of the distance function via chain rule
    if gradient:
        if kernel in ["berezovsky"]:
            f_r = np.add(f1, np.zeros_like(f2)) # recast to the same shape as f_m
            mask = (f_r > 0)
            res[mask] *= 1 / (f_r[mask] * np.log(2)) # derivative of log2(f1/f2)
        else:
            f_d = np.subtract(f1, f2)
            mask = (f_m > 0) & (f_d != 0)
            res[mask] *= f_d[mask] / (1.72 * f_m[mask]**(0.65) * np.abs(f_d[mask]))\
                       - (0.65 * f_m[mask]**(-1.65) * np.abs(f_d[mask])) / 1.72

    return res




def tonal_for_frames(P1, P2, K=12, f_ref=440., fit_grid=True, gradient=False):
    """Calculate total and weighted tonal cost for sets of pure tones

    Parameters
    ----------
        P1 : 2D or 3D float numpy array
            (T, N, 2) numpy array with N frequency/amplitude pairs (f_n, a_n), as for example
            returned by 'utils.find_peaks'. 'T' is an optional time dimension and must be equal
            to 'P2' if given.
        P2 : 2D or 3D float numpy array
            (T, M, 2) numpy array with M frequency/amplitude pairs (f_m, a_m), as for example
            returned by 'utils.find_peaks'. 'T' is an optional time dimension and must be equal
            to 'P1' if given.
            Only used to find the best grid shift automatically (default is an empty list, then 'f_ref' is used)
        K : int
            Number of ET divisions per octave (default: 12)
        f_ref : float
            Reference frequency for alignment of the the ET grid in Hz (default: 440 Hz)
        fit_grid : bool
            Whether or not to find the best fitting reference frequency for P2 (default True)
        gradient : bool
            Whether to return the cost value (default) or the gradient at each f

    Returns
    -------
        cost : 1D float numpy array (dimensions: (T))
            Total tonal cost or tonal cost gradient for all frequencies in P1, weighted and
            normalized by amplitude. T is 1 if P1 and P2 have only two dimensions.
    """
    P_lead, P_backing = _ensure_dimensions_peak_sets(P1, P2)
    T = P_lead.shape[0]

    result = np.zeros((T))
    for t in range(T):
        if len(P_lead[t]) == 0 or np.sum(P_lead[t,:,1]) < 0.0001:
            # return zero cost if the set is empty or practically silent
            # (happens e.g. when a voice is quiet in the signal analyzed by 'utils.find_peaks')
            continue

        # find best reference frequency for et penalty
        # TODO: more elegant way to find minimum
        if fit_grid and len(P_backing[t]) > 0:
            ref_candidates = np.linspace(440 * np.power(2, -0.5/K), 440 * np.power(2, 0.5/K), 100)
            ref_results = np.zeros_like(ref_candidates)
            for i in range(len(ref_candidates)):
                for j in range(len(P_backing[t])):
                    if P_backing[t,j,0] == 0: continue
                    ref_results[i] += np.abs(P_backing[t,j,1]) * tonal(P_backing[t,j,0], K=K, f_ref=ref_candidates[i])[0]
            opt_ref = ref_candidates[np.argmin(ref_results)]
        else:
            opt_ref = f_ref

        # with found reference, calculate cost for lead voice, weighted and normalized by amplitude
        ampl = 0
        for i in range(len(P_lead[t])):
            if P_lead[t,i,0] == 0: continue
            result[t] += np.abs(P_lead[t,i,1]) * tonal(P_lead[t,i,0], K=K, f_ref=opt_ref, gradient=gradient)[0]
            ampl += np.abs(P_lead[t,i,1])

        result[t] /= ampl

    return result


def harmonic_for_frames(P1, P2, log_mag_weights=False, log_mag_gamma=1., ampl_exp=1.,
                        norm="lead_backing_sum", ampl_method="min", ignore_distant_pairs_thrsh=None, **kwargs):
    """Calculate harmonic cost between all pairs of pure tones in two sets

    Parameters
    ----------
        P1 : 2D or 3D float numpy array
            (T, N, 2) numpy array with N frequency/amplitude pairs (f_n, a_n), as for example
            returned by 'utils.find_peaks'. 'T' is an optional time dimension and must be equal
            to 'P2' if given.
        P2 : 2D or 3D float numpy array
            (T, M, 2) numpy array with M frequency/amplitude pairs (f_m, a_m), as for example
            returned by 'utils.find_peaks'. 'T' is an optional time dimension and must be equal
            to 'P1' if given.
        log_mag_weights : bool
            Whether or not to use log compression `log(1 + gamma * mag)` for the amplitude weighting of each pair
            (default: False)
        log_mag_gamma : float
            Compression strength for the log compression using the formula `log(1 + gamma * mag)` (default: 1.0)
        ampl_exp : float
            Optional exponential compression for the amplitude weighting, where a value < 1 compresses the amplitudes
            (default: 1.0)
        norm : string
            Which norm to apply to the result (one of "lead_sum, "lead_count", "lead_backing_count", "full_count",
            "lead_backing_sum", "full_sum", "none")
        ampl_method : string
            Which method to use to compare amplitudes (one of "min", "mult", "beating")
        ignore_distant_pairs_thrsh : float or None
            If a number is given, frequency pairs with a distance above this value in octaves are ignored
            in the calculation (default: None)

    Returns
    -------
        Total harmonic cost or harmonic cost gradient for all frequencies in P1 w.r.t. P2,
        weighted depending on the settings
    """
    P_lead, P_backing = _ensure_dimensions_peak_sets(P1, P2)

    if ampl_method == "min":
        A = np.minimum(np.abs(P_lead[:,None,:,1]), np.abs(P_backing[:,:,None,1]))
    elif ampl_method == "mult":
        A = np.multiply(np.abs(P_lead[:,None,:,1]), np.abs(P_backing[:,:,None,1]))
    elif ampl_method == "beating":
        A =  P_lead[:,None,:,1] * P_backing[:,:,None,1]
        A /= (P_lead[:,None,:,1] + P_backing[:,:,None,1] + 1e-8)
    else:
        raise ValueError(f"Unknown amplitude calculation method '{ampl_method}'.")

    if log_mag_weights:
        A = np.log(1 + log_mag_gamma * A)
    else:
        A = A ** ampl_exp

    D = harmonic(P_lead[:,None,:,0], P_backing[:,:,None,0], **kwargs)

    if ignore_distant_pairs_thrsh is not None:
        with np.errstate(divide='ignore', invalid='ignore'): # supress warning of log of zero
            ratio = np.where(
                (P_lead[:,None,:,0] > 0) & (P_backing[:,:,None,0] > 0),
                P_lead[:,None,:,0] / P_backing[:,:,None,0],
                0
            )
            dist_oct = np.where(ratio > 0, np.log2(ratio), 0)
        dist_mask = (dist_oct > ignore_distant_pairs_thrsh)
        A[dist_mask] = 0 # ignore pairs that have a distance larger than the given threshold

    result = np.sum(A*D, axis=(1, 2))

    if norm == "lead_sum":
        ampls = np.abs(P_lead[:,:,1])
        ampls = np.log(1 + log_mag_gamma * ampls) if log_mag_weights else ampls ** ampl_exp
        norm_val = np.sum(ampls, axis=1)
    elif norm == "lead_count":
        norm_val = P_lead.shape[1]
    elif norm == "lead_backing_count":
        norm_val = P_lead.shape[1] + P_backing.shape[1]
    elif norm == "full_count":
        norm_val = P_lead.shape[1] * P_backing.shape[1]
    elif norm == "lead_backing_sum":
        ampls = np.abs(np.concatenate([P_lead[:,:,1], P_backing[:,:,1]], axis=1))
        ampls = np.log(1 + log_mag_gamma * ampls) if log_mag_weights else ampls ** ampl_exp
        norm_val = np.sum(ampls, axis=1)
    elif norm == "full_sum":
        norm_val = np.sum(A, axis=(1, 2))
    elif norm == "none":
        norm_val = 1 - 1e-8
    else:
        raise ValueError(f"Unknown normalization type '{norm}'.")

    result /= (norm_val + 1e-8)

    return result



def _kernel_berezovsky(x, gradient, x_max, fixed_wc=None):
    result = np.zeros_like(x)
    mask = (x != 0) & (x_max != 0)
    if fixed_wc is not None:
        x_log = np.log(np.abs(x[mask] / fixed_wc))
    else:
        x_log = np.log(np.abs(x[mask] / x_max[mask]))

    if gradient:
        result[mask] = -2 * np.exp(-1 * x_log**2) * x_log / x[mask]
    else:
        result[mask] = np.exp(-1 * x_log**2)


    return result

def _kernel_bigand(x, gradient):
    if gradient:
        return 32 * x * (1 - 4 * x) * np.exp(2 - 8 * x)
    else:
        return (4 * x * np.exp(1 - 4 * x))**2

def _kernel_marjieh(x, gradient, p=0.096, q=1.632):
    result = np.zeros_like(x)
    mask = (x >= p)
    result[mask] = _kernel_bigand(x[mask], gradient)
    xp = x[~mask]/p

    if gradient:
        n = (1 - np.cos(2 * np.pi * xp))
        n_der = 2 * np.pi * np.sin(2 * np.pi * xp) / p
        result[~mask] = _kernel_bigand(x[~mask], gradient=False) / p + xp * _kernel_bigand(x[~mask], gradient=True) \
                        + q * n / p - q * (1 - xp) * n_der
    else:
        neg_part = (1 - np.cos(2 * np.pi * xp))
        result[~mask] = (xp) * _kernel_bigand(x[~mask], gradient) - q * (1 - xp) * neg_part
    return result


def _ensure_3d(P):
    if P.ndim == 2:
        return np.expand_dims(P, 0)
    else:
        return P


def _ensure_dimensions_peak_sets(P1, P2):
    assert len(P1.shape) == len(P2.shape), "P1 and P2 must have the same number of dimensions."
    assert len(P1.shape) == 2 or len(P1.shape) == 3, "P1 and P2 must have 2 or 3 dimensions."
    P1_e = _ensure_3d(P1)
    P2_e = _ensure_3d(P2)
    assert P1_e.shape[0] == P2_e.shape[0], "P1 and P2 must have the same first dimension."
    return P1_e, P2_e
