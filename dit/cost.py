"""
Description: cost measurement functions
Contributors: Simon Schwär, Sebastian Rosenzweig, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the Differentiable Intonation Tools
https://github.com/simonschwaer/differentiable-intonation-tools/
"""

import numpy as np
from numba import jit


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
            Whether to return the cost value (default) or the gradient at f
    
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
        return (1 - np.cos(2 * np.pi * K * np.log2(f_arr/f_ref))) /  2
    else:
        return np.pi * K * np.sin(2 * np.pi * K * np.log2(f_arr/f_ref)) / 1200

# @jit
def harmonic(f1, f2, fixed_wc=None, gradient=False):
    """Harmonic cost between frequencies based on Plomp/Levelt perceptual dissonance

    Parameters
    ----------
        f1 : float scalar or 1D numpy array
            Frequencies of first set in Hz (length N)
        f2  : float scalar or 1D numpy array
            Frequencies of second set in Hz (length M)
        fixed_wc : float
            Optional fixed dissonance curve width parameter
            (default "None": automatic width based on frequency)
        gradient : bool
            Whether to return the cost value (default) or the gradient at f

    Returns
    -------
        NxM 2D numpy array containing
        if gradient is False:
            The pairwise harmonic cost between f1 and f2 with arbitrary unit
            (maximum of 1 is reached when log2(f1/f2) = wc)
        if gradient is True:
            The pairwise harmonic cost gradient w.r.t f1 with unit 'cost change per cent shifted'
    Notes
    -----
        The function uses a parametrization of the harmonic cost from:
        J. Berezovsky, “The structure of musical harmony as an ordered phase of sound:
        A statistical mechanics approach to music theory,” Science Advances, vol. 5,
        p. eaav8490, May 2019.
    """

    f1_arr = np.atleast_1d(np.asarray(f1, dtype=np.float32))
    f2_arr = np.atleast_1d(np.asarray(f2, dtype=np.float32))

    assert len(f1_arr.shape) == 1, \
        "Input frequencies to 'dit.cost.harmonic' should be scalar or a 1D array."
    assert len(f2_arr.shape) == 1, \
        "Input frequencies to 'dit.cost.harmonic' should be scalar or a 1D array."
    assert np.all(f1_arr > 0), \
        "Input frequencies to 'dit.cost.harmonic' must be strictly positive."
    assert np.all(f2_arr > 0), \
        "Input frequencies to 'dit.cost.harmonic' must be strictly positive."

    N = f1_arr.shape[0]
    M = f2_arr.shape[0]
    result = np.zeros((N, M))

    # np.vectorize is not really faster than the nested loop, so we keep
    # them for readability. Also with numba, nested loops seem to be faster
    # than vectorization.
    for i in range(N):
        for j in range(M):
            if f1_arr[i] == f2_arr[j]:
                result[i,j] = 0.
                continue

            if fixed_wc is not None:
                wc = fixed_wc
                assert wc > 0, "fixed_wc input to 'dit.cost.harmonic' must be strictly positive."
            else:
                f_m = (f1_arr[i] + f2_arr[j]) / 2
                # wc = 0.67 * min(f1_arr[i], f2_arr[j])**(-0.68)
                wc = 0.25 * 24.7 * ((4.37 * f_m) / 1000 + 1) / f_m
                # wc = 0.03 * min(f1_arr[i], f2_arr[j])**(-0.35)

            ln_dfwc = np.log(np.abs(np.log2(f1_arr[i]/f2_arr[j]))/wc)
            if not gradient:
                result[i,j] = np.exp(-1 * ln_dfwc**2)
            else:
                result[i,j] = -1 * np.exp(-1 * ln_dfwc**2) * ln_dfwc / (600 * np.log2(f1_arr[i]/f2_arr[j]))

    return result


def kernel_berezovsky(x, x_max):
    result = np.zeros_like(x)
    mask = (x != 0)
    x_log = np.log(np.abs(x[mask] / x_max[mask]))
    result[mask] = np.exp(-1 * x_log**2)
    return result

def kernel_bigand(x):
    return (4 * x * np.exp(1 - 4 * x))**2

def kernel_marjieh(x, p=0.096, q=1.632):
    result = np.zeros_like(x)
    mask = (x >= p)
    result[mask] = kernel_bigand(x[mask])
    xp = x[~mask]/p
    result[~mask] = (xp) * kernel_bigand(x[~mask]) - q * (1 - xp) * (1 + np.sin(2 * np.pi * xp - np.pi / 2))
    return result

def kernel_schwaer(x, x_min, x_max, q=1.632):
    result = np.zeros_like(x)
    mask = (x != 0)
    x_log_max = np.log(np.abs(x[mask] / x_max[mask]))**2
    x_log_min = np.log(np.abs(x[mask] / x_min[mask]))**2
    x_log_minmax = np.log(np.abs(x_min[mask] / x_max[mask]))**2
    result[mask] = np.exp(-x_log_max) - (q + np.exp(-x_log_minmax)) * np.exp(-x_log_min * 4)
    return result

def harmonic_new(f1, f2, max_at_erb=0.25, min_at_erb=0.08, kernel="berezovsky",
                 individual_pairs=False, berezovsky_original_max=False, **kwargs):

    if individual_pairs:
        f_m = np.add(f1, f2) / 2
    else:
        f_m = np.add.outer(f1, f2) / 2

    erb = 24.7 * ((4.37 * f_m) / 1000 + 1) / f_m

    d_max = max_at_erb * erb
    d_min = min_at_erb * erb

    if kernel in ["berezovsky", "schwaer"]:
        if individual_pairs:
            d = np.log2(np.divide(f1, f2))
        else:
            d = np.log2(np.divide.outer(f1, f2))
    elif kernel in ["marjieh", "bigand"]:
        if individual_pairs:
            d = np.abs(np.subtract(f1, f2)) / 1.72 * f_m**(-0.65)
        else:
            d = np.abs(np.subtract.outer(f1, f2)) / 1.72 * f_m**(-0.65)
    else:
        raise ValueError(f"Unknown kernel type '{kernel}'.")

    if kernel == "berezovsky":
        if berezovsky_original_max:
            if individual_pairs:
                f_min = np.minimum(f1, f2)
            else:
                f_min = np.minimum.outer(f1, f2)
            d_max = 1.33 * f_min**(-0.68)
        res = kernel_berezovsky(d, d_max, **kwargs)
    elif kernel == "marjieh":
        res = kernel_marjieh(d, **kwargs)
    elif kernel == "bigand":
        res = kernel_bigand(d, **kwargs)
    elif kernel == "schwaer":
        res = kernel_schwaer(d, d_min, d_max, **kwargs)
    else:
        raise ValueError(f"Unknown kernel type '{kernel}'.")

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
        if len(P_lead[t]) == 0 or np.sum(P_lead[t,:,0]) < 0.0001:
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


def harmonic_for_frames(P1, P2, log_mag_weights=False, log_mag_gamma=1, ampl_exp=1, norm="lead_backing_sum", **kwargs):
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
        fixed_wc : float
            Optional fixed dissonance curve width parameter
            (default "None": automatic width based on frequency)
        gradient : bool
            Whether to return the cost value (default) or the gradient at f

    Returns
    -------
        Total harmonic cost or harmonic cost gradient for all frequencies in P1 w.r.t. P2,
        weighted by amplitude

        TODO: Amplitude weighting would not be elegant here, because the average cost
        of one frequency pair is very low. Currently, the sum of all weighted pairings
        is divided only by the summed amplitudes in P1. This follows from the assumption
        that on average, each frequency in P1 is close to one or two frequencies in P2,
        so that the division results in a cost range comparable to the tonal cost.
    """
    P_lead, P_backing = _ensure_dimensions_peak_sets(P1, P2)

    result = np.zeros((P_lead.shape[0]))
    # ampl_tot = np.zeros((P_lead.shape[0]))

    for i in range(P_lead.shape[1]): # loop over lead harmonics
        for j in range(P_backing.shape[1]): # loop over backing harmonics
            mask = ((P_backing[:,j,0] > 0) & (P_lead[:,i,0] > 0))

            ampl = np.minimum(np.abs(P_lead[mask,i,1]), np.abs(P_backing[mask,j,1])) # element-wise
            if log_mag_weights:
                ampl = np.log(1 + log_mag_gamma * ampl)
            else:
                ampl = ampl ** ampl_exp

            result[mask] += ampl * harmonic_new(P_lead[mask,i,0], P_backing[mask,j,0], individual_pairs=True, **kwargs)

            # ampl_mask = (np.abs(np.log2(P_lead[mask,i,0] / P_backing[mask,j,0])) > 2)
            # ampl[ampl_mask] = 0 # ignore distances over 1.2 octaves (see Plomp & Levelt, Hutchinson & Knopoff)
            # ampl_tot[mask] += ampl


    # result /= (ampl_tot + 1e-8)
    if norm == "lead_sum":
        ampls = np.abs(P_lead[:,:,1])
        if log_mag_weights:
            ampls = np.log(1 + log_mag_gamma * ampls)
        else:
            ampls = ampls ** ampl_exp

        norm_val = np.sum(ampls, axis=1)
    elif norm == "lead_count":
        norm_val = P_lead.shape[1]
    elif norm == "lead_backing_count":
        norm_val = P_lead.shape[1] + P_backing.shape[1]
    elif norm == "full_count":
        norm_val = P_lead.shape[1] * P_backing.shape[1]
    elif norm == "lead_backing_sum":
        ampls = np.abs(np.concatenate([P_lead[:,:,1], P_backing[:,:,1]], axis=1))
        if log_mag_weights:
            ampls = np.log(1 + log_mag_gamma * ampls)
        else:
            ampls = ampls ** ampl_exp

        norm_val = np.sum(ampls, axis=1)
    elif norm == "full_sum":
        a_l = np.abs(P_lead[:,:,1])
        a_b = np.abs(P_backing[:,:,1])
        ampls = np.minimum(a_l[:,:,None], a_b[:,None,:])
        if log_mag_weights:
            ampls = np.log(1 + log_mag_gamma * ampls)
        else:
            ampls = ampls ** ampl_exp

        norm_val = np.sum(ampls, axis=(1, 2))

    else:
        raise ValueError(f"Unknown normalization type '{norm}'.")

    result /= (norm_val + 1e-8)

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
