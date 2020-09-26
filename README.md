## Digits detection dataset

<p align="center">
  <img src="https://github.com/lzanini/digits-detection-dataset/blob/master/img/figure.png">
</p>

I implemented this Pytorch Dataset to experiment with object detection in a simple setting. It consists of images containing one or several MNIST digits of various shapes and sizes, and their corresponding labels and box positions. Images are create lazily (when `__getitem__` is called), to avoid filling the RAM. Several parameters are available, such as the number of images to generate, the resolution, the size of the digits, the maximum number of digits per image, or the margin of boxes. It can also be generated from either the train or the test MNIST partition.

The mnist dataset will be downloaded and saved in the folder <mnist_root> when the program is run for the first time; then is will loaded from the disk.

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

```
Cropping Mnist images...
59999/60000  (Elapsed time: 4.84 sec.)
Building Digits Detection dataset...
99999/100000  (Elapsed time: 10.3 sec.)
```
