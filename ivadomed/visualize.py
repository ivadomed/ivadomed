import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np


def overlap_im_seg(img, seg):
    """Overlap image (background, greyscale) and segmentation (foreground, jet)."""
    seg_zero, seg_nonzero = np.where(seg <= 0.1), np.where(seg > 0.1)
    seg_jet = plt.cm.jet(plt.Normalize(vmin=0, vmax=1.)(seg))
    seg_jet[seg_zero] = 0.0
    img_grey = plt.cm.binary_r(plt.Normalize(vmin=np.amin(img), vmax=np.amax(img))(img))
    img_out = np.copy(img_grey)
    img_out[seg_nonzero] = seg_jet[seg_nonzero]
    return img_out


class LoopingPillowWriter(anim.PillowWriter):
    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)


class AnimatedGif:
    """Generates GIF.

    Args:
        size (tuple): Size of frames.

    Attributes:
        fig (plt):
        size_x (int):
        size_y (int):
        images (list): List of frames.
    """

    def __init__(self, size):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 50, size[1] / 50)
        self.size_x = size[0]
        self.size_y = size[1]
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.images = []

    def add(self, image, label=''):
        plt_im = self.ax.imshow(image, cmap='Greys', vmin=0, vmax=1, animated=True)
        plt_txt = self.ax.text(self.size_x * 3 // 4, self.size_y - 10, label, color='red', animated=True)
        self.images.append([plt_im, plt_txt])

    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images, interval=50, blit=True,
                                         repeat_delay=500)
        animation.save(filename, writer=LoopingPillowWriter(fps=1))
