import numpy as np
import torch


def crop(image):
    """

    Removes empty lines and rows at the extremities of an image,
    and convert it to a pytorch tensor.

    """
    image = np.array(image) / 255.
    m, n = image.shape
    mask = image > 0
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return torch.tensor(image[row_start:row_end, col_start:col_end])


def overlap(box1, box2):
    """

    Returns True if box1 and box2 overlap. Boxes are represented by a list of 4 element (x1, y1, x2, y2)
    where (x1, y1) is the top left corner and (x2, y2) the bottom right corner

    """
    (left1_x, left1_y, right1_x, right1_y) = box1
    (left2_x, left2_y, right2_x, right2_y) = box2
    return not (left1_x > right2_x or left2_x > right1_x or left1_y < right2_y or left2_y < right1_y)


def plot_image(image, classes, boxes):
    """ Plot the image, as well as boxes and class labels. """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.imshow(1-image, cmap='Greys')
    ax = plt.gca()
    for (cls, (bx1, by1, bx2, by2)) in zip(classes, boxes):
        rect = patches.Rectangle((by1-1, bx1-1), by2-by1, bx2-bx1,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        plt.text(by2-6, bx1+6, str(cls.numpy()), bbox=dict(color='red'))
        ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
