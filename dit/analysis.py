import numpy as np

from .cost import tonal_for_frames, harmonic_for_frames


def calc_shifts_multivoice(P, wt, wh, mu=200, lmbd=0.03, fit_grid=False, K=12, f_ref=440., fixed_wc=None):
    """Calculate "optimal" pitch-shifts for multiple voices with joint gradient descent

    Parameters
    ----------
        P : 4D float numpy array
        	(V, L, M, 2) numpy array with V voices (or audio tracks), L time frames and M frequency/amplitude
        	pairs (f_n, a_n), as for example returned by 'utils.find_peaks'.
        wt : float
        	Weight of tonal cost
        wh : float 
        	Weight of harmonic cost
        mu : float
        	Update or "learning" rate for the gradient descent (default: 200)
        lmbd : float
			Regularization factor to disencourage voices drifting apart from each other (default: 0.03)
        fit_grid : bool
            (Used in tonal cost) Whether or not to find the best fitting reference frequency for P2 (default True)
        K : int
            (Used in tonal cost) Number of ET divisions per octave (default: 12)
        f_ref : float
            (Used in tonal cost) Reference frequency for alignment of the the ET grid in Hz (default: 440 Hz)
        fixed_wc : float
            (Used in harmonic cost) Optional fixed dissonance curve width parameter
            (default "None": automatic width based on frequency)

    Returns
    -------
        (V, L) array of resulting pitch shifts for L time frames and V voices
    """
    V = P.shape[0]
    L = P.shape[1]
    M = P.shape[2]
    result = np.zeros((V, L))

    for l in range(0, L):
        for v in range(V):
            # virtually apply pitch-shift of previous frame to selected voice
            P_lead = P[v,l].copy()
            P_lead[:,0] *= np.power(2, result[v,l-1]/1200)

            # accumulate all other voices after virtually applying shift of previous frame
            P_backing = np.zeros((M*(V-1), 2))
            j = 0
            for i in range(V):
                if i == v: continue
                P_backing[j*M:(j+1)*M,:] = P[i,l,:,:]
                P_backing[j*M:(j+1)*M,0] *= np.power(2, result[i,l-1]/1200)
                j += 1

            # calculate cost gradient and regularization for this frame
            dCdp = wt * tonal_for_frames(P_lead, P_backing, fit_grid=fit_grid, K=K, f_ref=f_ref, gradient=True) + \
            	   wh * harmonic_for_frames(P_lead, P_backing, fixed_wc=fixed_wc, gradient=True)
            reg = result[v,l-1] - np.mean(result[:,l-1])
            result[v,l] = result[v,l-1] - mu * dCdp - lmbd * reg

    return result
