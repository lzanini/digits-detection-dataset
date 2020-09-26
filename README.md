## Digits detection dataset

<p align="center">
  <img src="https://github.com/lzanini/digits-detection-dataset/blob/master/img/figure.png">
</p>

I implemented this Pytorch Dataset to experiment with object detection in a simple setting. It consists of images containing one or several MNIST digits of various shapes and sizes, and their corresponding labels and box positions. Images are generated ahead of time, when initializing the Dataset. Several parameters are available, such as the number of images to generate, the resolution, the size of the digits, the maximum number of digits per image, or the margin of boxes. 

The mnist dataset will be downloaded and saved in the folder <mnist_root> the first time the program is run; then is will loaded from the disk.

## Usage

```python
from digits_detection.dataset import DigitsDetectionDataset

dataset = DigitsDetectionDataset(mnist_root="../data/",
                                 dataset_length=40000,
                                 image_resolution=(128, 128),
                                 max_digits=5,
                                 min_size=20,
                                 max_size=20*3,
                                 margin=6)

image, (classes, boxes) = dataset[0]
```

```
Cropping Mnist images...
59999/60000  (Elapsed time: 4.84 sec.)
Building Mnist Detection dataset...
39999/40000  (Elapsed time: 11.6 sec.)
```

