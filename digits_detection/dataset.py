import random
from time import time

import torchvision
import torch
from torch.utils.data import Dataset

from digits_detection.utils import overlap, crop


class CroppedMnistDataset(Dataset):
    """

    MNIST Dataset with cropped images, to remove empty borders.

    """
    def __init__(self, root, train=True):

        # get the mnist dataset from torchvision
        try:
            dataset = torchvision.datasets.MNIST(root, train=train, transform=crop)
        except RuntimeError:
            dataset = torchvision.datasets.MNIST(root, train=train, transform=crop, download=True)

        start = time()
        print("Cropping Mnist images...")

        # crop all images
        self.data = []
        for idx in range(len(dataset)):
            self.data.append(dataset[idx])
            print(f"\r{idx}/{len(dataset)}", end="")
        print(f"  (Elapsed time: {time() - start:.3} sec.)")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DigitsDetectionDataset(Dataset):
    """

    A dataset for Digits detection. Images are made off several MNIST digits, and outputs are classes
    and digit boxes (x1, y1, x2, y2).

    Parameters
    ----------

        mnist_root: str
            Path to download/load the mnist dataset.

        dataset_length: int
            Number of images to generate.

        train: bool
            If True, uses digits from the train MNIST dataset.

        image_resolution: tuple of int
            Resolution of the images (pixels).

        max_digits: int
            The number of digits in an image is sampled uniformly on [1, max_digits].

        min_size: int
            Minimum resolution of digits in the image (pixels).

        max_size: int
            Maximum resolution of digits in the image (pixels).

        margin: int
            Number of margin pixels for the boxes; i.e. how far are the boxes from the digits they contain.

    """
    def __init__(self, mnist_root, dataset_length, train=True, image_resolution=(140, 140),
                 max_digits=5, min_size=10, max_size=40, margin=3):
        self.params = {
            "image_resolution": image_resolution,
            "max_digits": max_digits,
            "min_size": min_size,
            "max_size": max_size,
            "margin": margin
        }
        self.data = {"image": [], "classes": [], "boxes": []}
        self.mnist = CroppedMnistDataset(root=mnist_root, train=train)
        self.build_dataset(dataset_length)

    def build_dataset(self, dataset_length):
        print("Building Mnist Detection dataset...")
        start = time()
        for idx in range(dataset_length):
            image, boxes, classes = self.generate_sample()
            self.data["image"].append(image)
            self.data["boxes"].append(boxes)
            self.data["classes"].append(classes)
            print(f"\r{idx}/{dataset_length}", end="")
        print(f"  (Elapsed time: {time() - start:.3} sec.)")

    def generate_sample(self):
        """

        Generate a single image containing 1 or several digits, as well as the corresponding labels and boxes.

        """
        # Number of digits in the image
        nb_digits = random.randint(1, self.params["max_digits"])
        image = torch.zeros(self.params["image_resolution"])
        classes = []
        boxes = []

        for _ in range(nb_digits):

            # will retry if boxes overlap
            nb_retry = 0
            max_retry = 100
            is_collision = True
            object_idx = random.randint(0, len(self.mnist) - 1)
            img, cls = self.mnist[object_idx]
            img_ratio = img.shape[1] / img.shape[0]

            while nb_retry < max_retry and is_collision:
                nb_retry += 1

                # make sure that the image is smaller than max_size
                if img_ratio < 1:
                    size_x = random.randint(self.params["min_size"], self.params["max_size"])
                    size_y = int(img_ratio * size_x)
                else:
                    size_y = random.randint(self.params["min_size"], self.params["max_size"])
                    size_x = int(size_y / img_ratio)

                position_x = random.randint(self.params["margin"],
                                            self.params["image_resolution"][0] - size_x - self.params["margin"])
                position_y = random.randint(self.params["margin"],
                                            self.params["image_resolution"][1] - size_y - self.params["margin"])

                box = (position_x - self.params["margin"],
                       position_y + size_y + self.params["margin"],
                       position_x + size_x + self.params["margin"],
                       position_y - self.params["margin"])

                is_collision = any(overlap(box, other) for other in boxes)

                if not is_collision:
                    resized = torch.nn.functional.interpolate(img[None, None, :, :], (size_x, size_y))[0, 0]
                    image[position_x:position_x + size_x, position_y:position_y + size_y] = resized
                    classes.append(torch.tensor(cls))
                    boxes.append(box)

        boxes = [torch.tensor(box) for box in boxes]
        return image, torch.stack(boxes), torch.stack(classes)

    def __getitem__(self, index):
        """

        Returns
        -------

            image: torch.tensor, shape = (self.img_size, self.img_size)
                The image.

            classes: tensor of int, shape=(nb_instance,)
                Classes of every instances present in the image.

            boxes: tensor of float, shape=(nb_instances, 4)
                Box localization (x1, x2, y1, y2) for each instance.

        """
        image = self.data["image"][index]
        classes = self.data["classes"][index]
        boxes = self.data["boxes"][index]
        return image, (classes, boxes)

    def __len__(self):
        return len(self.data["image"])


if __name__ == '__main__':

    detection_dataset = DigitsDetectionDataset(mnist_root="../data/",
                                               dataset_length=40000,
                                               image_resolution=(128, 128),
                                               max_digits=5,
                                               min_size=20,
                                               max_size=20*2,
                                               margin=6)

    from digits_detection.utils import plot_image
    i = random.randint(0, len(detection_dataset))
    image_, (classes_, boxes_) = detection_dataset[i]
    plot_image(image_, classes_, boxes_)
