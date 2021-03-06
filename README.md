## MNIST Detection Dataset

A Pytorch dataset containing non-overlaping MNIST digits of various shapes and sizes, with their corresponding labels and box positions. 

<p align="center">
  <img src="https://github.com/lzanini/digits-detection-dataset/blob/master/img/figure.png">
</p>

Images are generated lazily (when `__getitem__` is called) to avoid filling the RAM. Several parameters are available, such as the number of images to generate, the resolution, the size of the digits, the maximum number of digits per image, and the margin of boxes.

When the program runs for the first time, the original mnist dataset will be downloaded and saved in the folder `mnist_root`.

## Usage

```python
from digits_detection.dataset import DigitsDetectionDataset

dataset = DigitsDetectionDataset(mnist_root="../data/",
                                 train=True,
                                 dataset_length=100000,
                                 image_resolution=(256, 256),
                                 max_digits=5,
                                 min_size=28,
                                 max_size=28*3,
                                 margin=6)

image, (classes, boxes) = dataset[0]
```
