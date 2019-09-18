# Transfer Learning with Convolutional Neural Networks

## Material Curated By: Chris Havenstein


---

## Overview:

This GitHub repository is intended to give you some code to play with to try out transfer learning with convolutional neural networks also called CNNs). The data set here will be for a classification machine learning problem where the goal is to learn from labeled image data with three classes: "cat", "dog", and "horse". Then, we see how well our image classifier does on an validation image data set.

An overview of each of the files and directories follows. 


---

## Files and Folders In The Root Directory
In the root directory, there are four files: "Data Augmentation Generator.py", "dependencies list.txt", "transfer-learning.py", and "transfer-learning_w_data_augmentation.py" 

1. **"dependencies list.txt"** - This file will give you the python version I used along with all the libraries I downloaded and included in an anaconda environment. Also, I've included the version and build of each. My hope is this should help you to replicate it the environment I used. I also would like to specifically call out one of these libraries, tensorflow, that I did use the version of tensorflow that is configured to used a GPU (tensorflow-gpu). It is extremely useful if you have access to a compatible GPU with tensorflow-GPU, because it performs the linear algebra and matrix production calculations typically ~10x (or perhaps more than 10x times) faster than a CPU. I talk about setting up tensorflow-GPU in the section "tensorflow-gpu Set Up" further below.

2. **"transfer-learning.py"** - This file will use images that are in the "train" and "validation" directories to create a CNN model with transfer learning. The code is very well commented and I hope it will be easy to follow. An easy way to get this program working is to copy the contents of the "train" directory into the "validation" directory. However, copying the train images into the validation image directory comes with a disclaimer. Typically, you want your validation dataset/images to be different images that your CNN model has never learned from to build the model (i.e., not your training images) but are still images of the same classes that your model can classify into (cat, dog, and horse). A better process would be to find different cat, dog, and horse images to put in your validation subdirectories for cat, dog, and horse. 

3. **"Data Augmentation Generator.py"** - The data augmentation generator does something in my opinion which is very cool. What it will do is take the original training images that are in the "train" subdirectories for cat, dog, and horse images and it will create images that will be rotated, shifted, sheared, zoomed, or flipped horizontally within set ranges. Another word for data augmentation is synthetic data. We are creating different examples from our original images that simulate the original images in different conditions. The data generator will create different versions of permuted images for the "train2" and "validation2" directories and their subdirectories. 

4. **"transfer-learning_w_data_augmentation.py"** - This file will used the new synthetic images in the "train2" and "validation2" directories to perform the same process as "transfer-learning.py".

5. **"checkpoints"** - this directory is just created to save results during the CNN model fitting process to plot.


---

## TensorFlow-gpu Set Up

Installing tensorflow-gpu requires more work, but boy is it worth it in the time savings you get with a compatible GPU.

* Try in Anaconda (Windows) to create a tensorflow-gpu environment:

```
conda create –name tf_gpu tensorflow-gpu
activate tf_gpu
conda install tensorflow-gpu
conda install spyder
conda install pydot
spyder activate tf_gpu
```

* Also For TensorFlow-GPU: The following NVIDIA® software must be installed on your system:

1. [NVIDIA® GPU drivers](https://www.nvidia.com/drivers) —CUDA 10.0 requires 410.x or higher. 

2.  [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) — TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0) 

3. [cuDNN SDK](https://developer.nvidia.com/cudnn) (>= 7.4.1)

* You may also refer to **"keras-intro.docx"** for some additional information regarding Keras if you are interested.

* [This link](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781) will also likely help with  the tensorflow-gpu set up process by providing a number of steps with plenty of screenshots.

* If you have a discrete GPU and you want to know if it is compatible with CUDA for tensorflow-gpu, try starting [here.] (https://developer.nvidia.com/cuda-gpus)


---

## TensorFlow-GPU and Keras Installation Test

* If you want to test that your TensorFlow-GPU is configured correctly, you can try playing with the code below.:


```python
# -*- coding: utf-8 -*-
"""
TensorFlow and Keras installation test
"""

import tensorflow as tf
import keras.backend as K

#create message
message = tf.constant('Hello world!')

with tf.Session() as session:
    #Print 'Hello world!'
    session.run(message)
    print(message.eval())
    #List devices tensorflow sees
    devices = session.list_devices()
    for d in devices:
        print('\n\n', d.name)
        
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        print('\n\n', a.shape, b.shape)
        c = tf.matmul(a, b)

        # with tf.Session() as sess:
        print('\n\n', session.run(c))

#verify Keras is working   
print('\n\nEpsilon:', K.epsilon())
```


---

## Sources

I have referred to a number of sources in this repo when creating the code. I would recommend looking at the links that I have included as comments in this repos' .py files. If any of those links were useful or this repo was useful, please give credit to the relevent authors.


---

## Citation

If you use this implementation in your work, please cite the following:

```
@misc{chrishavenstein2019transferlearningCNNs,
  author = {Chris Havenstein},
  title = {Transfer Learning with Convolutional Neural Networks},
  year = {2019},
  howpublished = {\url{https://github.com/chavenstein/Transfer-Learning-With-CNNs}},
  note = {commit xxxxxxx}
}
```
