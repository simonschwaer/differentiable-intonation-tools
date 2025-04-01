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
    assert np.allclose(dc, np.array([0, 0, 0.093712]), atol=1e-05)

    f1 = np.array([300, 310, 1000])
    f2 = np.array([300, 310])
    c = dit.cost.harmonic(f1[:,None], f2[None,:])

    assert c.shape[0] == 3 and c.shape[1] == 2
    assert np.isclose(c[1,0], c[0,1])
    assert c[0,0] == 0 and c[1,1] == 0
    assert np.allclose(c[-1,:], np.zeros(2), atol=1e-5)


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

    _, P_lead, _ = dit.utils.find_peaks(x1, fs=fs, N=8000, H=8000, max_peaks=6, height=0)
    _, P_backing, _ = dit.utils.find_peaks(x2, fs=fs, N=8000, H=8000, max_peaks=6, height=0)

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
    P_lead = np.array([
        [(440, 1), (880, 1), (1320, 1)],
        [(440, 1), (880, 1), (1320, 1)],
        [(435, 1), (870, 1), (1305, 1)],
        [(445, 1), (890, 1), (1335, 1)],
    ])
    P_backing = np.array([
        [(440, 1), (880, 1), (1320, 1)],
        [(440, 1), (880, 1), (1320, 1)],
        [(440, 1), (880, 1), (1320, 1)],
        [(440, 1), (880, 1), (1320, 1)],
    ])


    c = dit.cost.harmonic_for_frames(P_lead, P_backing)
    dc = dit.cost.harmonic_for_frames(P_lead, P_backing, gradient=True)

    assert np.allclose(c,  np.array([0, 0, 0.2553904,  0.2566513]), atol=1e-5)
    assert np.allclose(dc, np.array([0, 0, 0.0439010, -0.0433353]), atol=1e-5)

    # test case with 2D array
    c = dit.cost.harmonic_for_frames(P_lead[0], P_backing[0])
    assert np.allclose(c, np.array([0]), atol=1e-5)
