import numpy as np
import dit


def test_tonal_cost():
    f = 220
    c = dit.cost.tonal(f)
    dc = dit.cost.tonal(f, gradient=True)

    assert np.isclose(c, 0, atol=1e-4)
    assert np.isclose(dc, 0, atol=1e-4)

    f = 220 * np.power(2, 1./12)
    c = dit.cost.tonal(f)
    dc = dit.cost.tonal(f, gradient=True)

    assert np.isclose(c, 0, atol=1e-4)
    assert np.isclose(dc, 0, atol=1e-4)

    f = 220 * np.power(2, 1.5/12)
    c = dit.cost.tonal(f)
    dc = dit.cost.tonal(f, gradient=True)

    assert np.isclose(c, 1, atol=1e-4)
    assert np.isclose(dc, 0, atol=1e-4)

    f = 220 * np.power(2, 1.1/12)
    c = dit.cost.tonal(f)
    dc = dit.cost.tonal(f, gradient=True)

    assert np.isclose(c, 0.0954915, atol=1e-4)
    assert np.isclose(dc, 0.0184658, atol=1e-4)

    f = 220 * np.power(2, 0.9/12)
    dc = dit.cost.tonal(f, gradient=True)

    assert np.isclose(dc, -0.0184658, atol=1e-4)

    f = 220 * np.power(2, 1.5/10)
    c = dit.cost.tonal(f, K=10)
    dc = dit.cost.tonal(f, K=10, gradient=True)

    assert np.isclose(c, 1, atol=1e-4)
    assert np.isclose(dc, 0, atol=1e-4)

    f = 216.5 * np.power(2, np.arange(12)/10)
    c = dit.cost.tonal(f, K=10, f_ref=433)
    dc = dit.cost.tonal(f, K=10, f_ref=433, gradient=True)

    assert np.allclose(c, np.zeros_like(c), atol=1e-4)
    assert np.allclose(dc, np.zeros_like(dc), atol=1e-4)


def test_harmonic_cost():
    f1 = 220
    f2 = 220
    c = dit.cost.harmonic(f1, f2)
    dc = dit.cost.harmonic(f1, f2, gradient=True)

    assert np.isclose(c, 0)
    assert np.isclose(dc, 0)

    wc = 0.03
    f1 = 220
    f2 = 220 * np.power(2, np.arange(3)*wc)
    c = dit.cost.harmonic(f1, f2, fixed_wc=wc)
    dc = dit.cost.harmonic(f1, f2, fixed_wc=wc, gradient=True)

    assert np.allclose(c, np.array([0, 1, 0.618503]))
    assert np.allclose(dc, np.array([0, 0, 0.0119087]), atol=1e-07)

    f1 = 300 * np.power(2, np.arange(3)/12)
    f2 = 200 * np.arange(1, 3)
    c = dit.cost.harmonic(f1, f2)
    dc = dit.cost.harmonic(f1, f2, gradient=True)

    assert c.shape[0] == 3 and c.shape[1] == 2
    assert np.allclose(c, np.array([
        [0.2576495, 0.30011772], [0.18561374, 0.43510599], [0.1349443, 0.64500703]
    ]))
    assert np.allclose(dc, np.array([
        [-0.00085488, 0.00132218], [-0.00060072, 0.00199433], [-0.00042348, 0.00286612]
    ]), atol=1e-07)


def test_tonal_cumulated():
    fs = 16000.

    x1 = np.zeros(int(1*fs))
    sig, phase_carry = dit.utils.synth(440, 0.5, fs, waveform='sawtooth')
    x1[:int(0.5*fs)] = sig
    sig, _ = dit.utils.synth(440 * np.power(2, 4./12), 0.5, fs, waveform='sawtooth', init_phase=phase_carry)
    x1[int(0.5*fs):] = sig

    x2, _ = dit.utils.synth(220 * np.power(2, 0.5/12), 1., fs, waveform='sawtooth')
    sig, _ = dit.utils.synth(220 * np.power(2, 4.5/12), 1., fs, waveform='sawtooth')
    x2 += sig

    _, P_lead, _ = dit.utils.find_peaks(x1, fs=fs, fft_size=8000, hop_size=8000, max_peaks=6, height=0)
    _, P_backing, _ = dit.utils.find_peaks(x2, fs=fs, fft_size=8000, hop_size=8000, max_peaks=6, height=0)

    c_fitted = dit.cost.tonal_for_frames(P_lead, P_backing, fit_grid=True)
    c_440 = dit.cost.tonal_for_frames(P_lead, P_backing, fit_grid=False)
    dc_fitted = dit.cost.tonal_for_frames(P_lead, P_backing, fit_grid=True, gradient=True)
    dc_440 = dit.cost.tonal_for_frames(P_lead, P_backing, fit_grid=False, gradient=True)

    assert np.allclose(c_fitted, np.array([1,1]), atol=0.05)
    assert np.allclose(c_440, np.array([0,0]), atol=0.05)
    assert np.allclose(dc_fitted, np.array([0,0]), atol=0.05)
    assert np.allclose(dc_440, np.array([0,0]), atol=0.05)

    # test case with 2D array
    c = dit.cost.tonal_for_frames(P_lead[0], P_backing[0], fit_grid=False)
    assert np.allclose(c, np.array([0]), atol=0.05)


def test_harmonic_cumulated():
    fs = 16000.

    x1 = np.zeros(int(1*fs))
    sig, phase_carry = dit.utils.synth(440, 0.25, fs, waveform='sawtooth')
    x1[:int(0.25*fs)] = sig
    sig, phase_carry = dit.utils.synth(440 * np.power(2, 4/12), 0.25, fs, waveform='sawtooth', init_phase=phase_carry)
    x1[int(0.25*fs):int(0.5*fs)] = sig
    sig, phase_carry = dit.utils.synth(440 * 1.3333333, 0.25, fs, waveform='sawtooth', init_phase=phase_carry)
    x1[int(0.5*fs):int(0.75*fs)] = sig
    sig, _ = dit.utils.synth(440 * np.power(2, 7/12), 0.25, fs, waveform='sawtooth', init_phase=phase_carry)
    x1[int(0.75*fs):] = sig

    x2, _ = dit.utils.synth(220, 1., fs, waveform='sawtooth')
    sig, _ = dit.utils.synth(220 * 1.5, 1., fs, waveform='sawtooth')
    x2 += sig

    _, P_lead, _ = dit.utils.find_peaks(x1, fs=fs, fft_size=4000, hop_size=4000, max_peaks=12, height=0)
    _, P_backing, _ = dit.utils.find_peaks(x2, fs=fs, fft_size=4000, hop_size=4000, max_peaks=12, height=0)

    c = dit.cost.harmonic_for_frames(P_lead, P_backing)
    dc = 1000 * dit.cost.harmonic_for_frames(P_lead, P_backing, gradient=True)

    assert np.allclose(c, np.array([0.234, 0.353, 0.236, 0.028]), atol=0.01)
    assert np.allclose(dc, np.array([-0.531, 0.431, -0.177, -0.039]), atol=0.01)

    # test case with 2D array
    c = dit.cost.harmonic_for_frames(P_lead[0], P_backing[0])
    assert np.allclose(c, np.array([0.234]), atol=0.05)
