# Semantic Segmentation
### Introduction
In this project, the pixels of a road in images are labeled using a Fully Convolutional Network (FCN), using transfer learning.

The hyper parameters that gave best results are:

|EPOCHS|BATCH_SIZE|KEEP_PROB|REG|LEARNING_RATE|EPSILON|CLIP_NORM|
|------|----------|---------|---|-------------|-------|---------|
|100|4 |.5|5.-4|1.-5|1.-8|0|

The results for some images are:

![](./runs/1508899791.9367473/um_000032.png)
![](./runs/1508899791.9367473/uu_000002.png)
![](./runs/1508899791.9367473/uu_000049.png)

### Setup
##### Frameworks and Packages
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```
