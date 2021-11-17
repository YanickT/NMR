from numba import njit
import numpy as np

GAMMA = 2.6752218744e2 / (2 * np.pi)
ALPHA = 0.1


@njit(cache=True)
def measure(sample, b0, tfactor, phases, f_larmor):
    """
    Performe a measurement of sample. Simulates NMR.
    :param sample: np.array[x, y, [density, t1, t2]] = 3D array representing the sample
    :param b0: np.array[x, y] = magnetic field (Do not forget the noise ;) )
    :param tfactor: int = number of steps per ms
    :param phases: np.array[t, [Gradx, Grady, Gradz, Pulse]] = 2D Array specifying the measurement process
    :param f_larmor: float = Excitation frequency
    :return: np.array[signal] = 1D array carrying the measured signal.
    """
    t1s = sample[:, :, 1]
    amplitude_t1 = sample[:, :, 0].astype(np.float32)
    amplitude_t1 /= np.sum(amplitude_t1)
    unready = np.zeros(amplitude_t1.shape, dtype=np.float32)
    t1_factor = np.zeros(t1s.shape[:2])
    for x, y in zip(*np.nonzero(t1s)):
        t1_factor[x, y] = np.exp(- 1 / (t1s[x, y] * tfactor))

    t2s = sample[:, :, 2]
    amplitudes = np.zeros(sample.shape[:2])
    t2_factor = np.zeros(t2s.shape[:2])
    for x, y in zip(*np.nonzero(t2s)):
        t2_factor[x, y] = np.exp(- 1 / (t2s[x, y] * tfactor))

    signal = np.zeros(phases.shape[0], dtype=np.complex64)
    signal_phase = np.zeros(sample.shape[:2])

    for t in range(phases.shape[0]):
        # determine frequencies
        if t == 0 or np.any(phases[t, :] != phases[t - 1, :]):
            b0_ = np.copy(b0)

            # Gradient x
            if phases[t, 0] != 0:
                for x in range(b0_.shape[1]):
                    b0_[:, x] += phases[t, 0] * (x - b0_.shape[1] // 2)

            # Gradient y
            if phases[t, 1] != 0:
                for y in range(b0_.shape[0]):
                    b0_[y, :] += phases[t, 1] * (y + b0_.shape[0] // 2)

            omegas = GAMMA * b0_
            omegas_t = omegas / tfactor

            # check for pulses
            if phases[t, 3] != 0:
                # excitation (this must have a good equation but the exponent will work for now)
                fac = np.exp(- ALPHA * np.abs(omegas / f_larmor - 1))

                # rotate M_xy
                amplitudes *= np.abs(np.cos(phases[t, 3])) * fac
                signal_phase *= np.cos(phases[t, 3]) * fac

                # rotate M_z and add imag to M_xy
                amplitudes += np.sin(phases[t, 3]) * (amplitude_t1 - unready) * fac
                unready += min((1 - np.cos(phases[t, 3])), 1) * (amplitude_t1 - unready) * fac

        # update amplitude in z direction (T1)
        unready *= t1_factor

        # update amplitude in xy-plane (T2)
        amplitudes *= t2_factor

        # add to signal
        signal[t] = np.sum(amplitudes * np.exp(1j * signal_phase))

        # update phase
        signal_phase += omegas_t

    return signal
