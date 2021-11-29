import numpy as np
import dit

def test_find_peaks():
	fs = 16000.
	t = np.linspace(0, 1., int(fs))
	x = np.zeros(t.shape)

	x[:8000] += 0.25 * np.sin(2*np.pi*t[:8000]*200)
	x[:8000] += 0.25 * np.sin(2*np.pi*t[:8000]*401.5)
	x[:8000] += 0.25 * np.sin(2*np.pi*t[:8000]*598.1)
	x[:8000] += 0.25 * np.sin(2*np.pi*t[:8000]*800)

	x[8000:] += 0.25 * np.sin(2*np.pi*t[8000:]*3000)
	x[8000:] += 0.25 * np.sin(2*np.pi*t[8000:]*3500)
	x[8000:] += 0.25 * np.sin(2*np.pi*t[8000:]*4000)
	# this is above max_freq, so it should be below the threshold
	x[8000:] += 0.25 * np.sin(2*np.pi*t[8000:]*4500)
	
	t, P, H = dit.utils.find_peaks(x, fs=fs, fft_size=8000, hop_size=8000, max_peaks=4, height=0)

	assert P.shape[0] == 2 and P.shape[1] == 4 and P.shape[2] == 2
	assert np.allclose(P[0,:,0], np.array([200, 401.5, 598.1, 800]), atol=0.05)
	assert np.allclose(P[1,:,0], np.array([3000, 3500, 4000, 0]), atol=5) # tolerance is higher at higher freqs


def test_synth():
	fs = 16000.
	x = np.zeros(int(1*fs))

	sig, phase_carry = dit.utils.synth(200, 0.25, fs, waveform='sawtooth')
	x[:int(0.25*fs)] = sig
	sig, phase_carry = dit.utils.synth(400, 0.25, fs, waveform='sawtooth', init_phase=phase_carry)
	x[int(0.25*fs):int(0.5*fs)] = sig
	sig, phase_carry = dit.utils.synth(300, 0.25, fs, waveform='square', init_phase=phase_carry)
	x[int(0.5*fs):int(0.75*fs)] = sig
	sig, _ = dit.utils.synth(200, 0.25, fs, waveform='square', init_phase=phase_carry)
	x[int(0.75*fs):] = sig
	
	t, P, H = dit.utils.find_peaks(x, fs=fs, fft_size=4000, hop_size=8000, max_peaks=4, height=0)

	assert P.shape[0] == 2 and P.shape[1] == 4 and P.shape[2] == 2
	assert np.allclose(P[0,:,0], np.array([200, 400, 600, 800]), atol=0.1)
	assert np.allclose(P[1,:,0], np.array([300, 900, 1500, 2100]), atol=0.1) # tolerance is higher at higher freqs


def test_f2s():
	assert dit.utils.f2s(220, 440) == "A3"
	assert dit.utils.f2s(330, 440) == "E4 +2c"
	assert dit.utils.f2s(180, 433) == "F#3 -20c"


def test_f2s():
	assert np.isclose(dit.utils.s2f("A4", 440), 440, atol=0.1)
	assert np.isclose(dit.utils.s2f("A4", 433), 433, atol=0.1)
	assert np.isclose(dit.utils.s2f("A2", 440), 110, atol=0.1)
	assert np.isclose(dit.utils.s2f("Ab2", 440), 110 * np.power(2, -1./12), atol=0.1)
	assert np.isclose(dit.utils.s2f("C#4", 440), 220 * np.power(2, 4./12), atol=0.1)
	assert np.isclose(dit.utils.s2f("C#4", 440, detune=10), 220 * np.power(2, 4.1/12), atol=0.1)
	assert np.isclose(dit.utils.s2f("C#4", 440, detune=-10), 220 * np.power(2, 3.9/12), atol=0.1)
