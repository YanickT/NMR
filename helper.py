import numpy as np
import skimage.draw as draw
import matplotlib.pyplot as plt


LAYERS = ["Density", "T1", "T2"]
PHASES = ["$Grad_x$", "$Grad_y$", "$Grad_z$", "Pulse", "Read"]


def create_example(size=100, radius=10, sep=0):
    half_size = size // 2
    image = np.zeros((size, size, 3), dtype=np.uint16)
    # first sample (oil)
    rs, cs = draw.disk((half_size, half_size - radius - sep), radius, shape=None)
    image[rs, cs, 0] = 1  # density layer
    image[rs, cs, 1] = 114  # t1 layer ms
    image[rs, cs, 2] = 112  # t2 layer ms

    # second sample (water)
    rs, cs = draw.disk((half_size, half_size + radius + sep), radius, shape=None)
    image[rs, cs, 0] = 1  # density layer
    image[rs, cs, 1] = 3269  # t1 layer ms
    image[rs, cs, 2] = 1214  # t2 layer ms

    return image


def view(np_array):
    f, axs = plt.subplots(1, 3)
    for i in range(3):
        axs[i].set_title(LAYERS[i])
        axs[i].imshow(np_array[:, :, i], cmap="hot")
    plt.show()


def view_phase(np_array, tfactor):
    f, axs = plt.subplots(4, 1, sharex=True)
    ts = np.arange(np_array.shape[0]) / tfactor
    for i in range(4):
        axs[i].set_title(PHASES[i])
        axs[i].plot(ts, np_array[:, i])
    plt.show()


if __name__ == "__main__":
    img = create_example(size=300, radius=30, sep=1)
    view(img)
    np.save("sample2_modi", img)

