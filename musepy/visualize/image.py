import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colors
from matplotlib.patches import Rectangle


def plot_colortable(color_list=None, *, sort_colors=False):
    cell_width = 22
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(color_list, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(color_list)

    n = len(names)
    nrows = 1

    width = cell_width * int(n//nrows) + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * int(n//nrows))
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        #ax.text(text_pos_x, y, name, fontsize=14,
        #        horizontalalignment='left',
        #        verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors.to_rgba(name), edgecolor='0.7')
        )

    return fig


def pixelate_and_quantize(image, palette, output_shape=None, to_rgb=False):

    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(image)

        # Load the input image
        image = cv2.imread(image)

    # Compute the pixelated version of the image
    if output_shape is not None:
        image = cv2.resize(image, output_shape, interpolation=cv2.INTER_NEAREST)

    # Quantize the colors in the image using the specified palette
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    quantized_img = image.reshape(-1, image.shape[-1])[:, None, :] - np.array(palette)[None, :, :]
    quantized_img = np.argmin(np.linalg.norm(quantized_img, axis=-1), axis=-1)

    if to_rgb:
        return np.array(palette)[quantized_img].reshape(*image.shape).astype(image.dtype)
    return quantized_img.reshape(*image.shape[:2])


def make_smiley():
    palette = [
        [0, 0, 0],        # Black
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255],    # Magenta
        [255, 255, 255],  # White
    ]
    output_shape = (63, 47)
    to_rgb = True
    # quantized_img = pixelate_and_quantize("data/img/animals/axolotl.png", palette, output_shape=output_shape)
    quantized_img = pixelate_and_quantize("data/img/emojis/smiley.jpg", palette,
                                          output_shape=output_shape, to_rgb=to_rgb)

    import matplotlib.pyplot as plt
    if to_rgb:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(np.array(palette) / 255)
        plt.imshow(quantized_img, cmap=cmap)
    else:
        plt.imshow(quantized_img)
    plt.show()


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([make_smiley,
                            ]
                           )
