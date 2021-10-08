from helper import view, view_phase
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


GAMMA = 2.6752218744
INT2REAL = 10.5 / 305


# FRAMES = []
# FIG, AX = plt.subplots()


def measure(sample, b0, tfactor, phases):
    """

    :param sample:
    :param b0:
    :param tfactor:
    :param phases:
    :return:
    """
    t1s = sample[:, :, 1]
    amplitude_t1 = sample[:, :, 0].astype(np.float32) * INT2REAL
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
        b0_ = np.copy(b0)
        if phases[t, 0] != 0:
            for x in range(b0_.shape[1]):
                b0_[:, x] += phases[t, 0] * x

        if phases[t, 1] != 0:
            for y in range(b0_.shape[0]):
                b0_[y, :] += phases[t, 1] * y

        omegas = np.ones(sample.shape[:2]) * GAMMA * b0_

        # update amplitude in z direction (T1)
        unready *= t1_factor

        # update amplitude in xy-plane (T2)
        amplitudes *= t2_factor

        # check for pulses
        if phases[t, 3] != 0:
            # rotate M_xy
            amplitudes *= np.abs(np.cos(phases[t, 3]))
            signal_phase = (2 * phases[t, 3] - signal_phase) % (2 * np.pi)
            # rotate M_z and add imag to M_xy
            diff = np.sin(phases[t, 3]) * (amplitude_t1 - unready)
            for x, y in zip(*np.nonzero(amplitudes)):
                signal_phase[x, y] /= (1 + diff[x, y] / amplitudes[x, y])
            amplitudes += diff
            unready += min((1 - np.cos(phases[t, 3])), 1) * (amplitude_t1 - unready)

        # add to signal
        signal[t] = np.sum(amplitudes * np.exp(1j * signal_phase))

        # update phase
        signal_phase += omegas / tfactor

        # funny animation
        """signals = amplitudes * np.exp(1j * signal_phase)
        temp = []
        non_zeros = np.nonzero(signals)
        temp += [AX.text(0.5, 1.05, f"Time {t / tfactor}", size=plt.rcParams["axes.titlesize"], ha="center", transform=AX.transAxes)]
        for x, y in zip(*non_zeros):
            temp += AX.plot([0, signals[x, y].real], [0, signals[x, y].imag])
        FRAMES.append(temp)
        print(t)"""

    return np.real(signal)


if __name__ == "__main__":
    sample = np.load("sample.npy")

    b0 = np.zeros(sample.shape[:2])
    b0.fill(0.43)
    b0 += np.random.normal(0, 0.01, b0.shape)

    signal, fig, frames = measure(b0, )
    ani = animation.ArtistAnimation(FIG, FRAMES, interval=10, blit=False, repeat_delay=1000)
    # ani.save("movie.gif", fps=10)
    plt.show()
