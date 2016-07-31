# autoencoder_demo
Autoencoder Demo for Fresh Machine Learning #5

Overview
============
This code demos a simple autoencoder that is able to generate handwritten digit images after training against the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. This the code for 'Build an Autoencoder in 5 Min' on [Youtube](https://youtu.be/GWn7vD2Ud3M)

Dependencies
============
* Python 2.7+ (https://www.python.org/download/releases/2.7/)
* Tensorflow (https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation)
* numpy (http://www.numpy.org/)


Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========

Once you've downloaded the repository simply type the following into terminal

`python main.py`

The autoencoder will automatically start downloading the MNIST dataset, then training on it. When finished, it'll be able to generate it's own images. See [this site](https://cs.stanford.edu/people/karpathy/convnetjs/demo/autoencoder.html) for a live autoencoder demo
in the browser 

Credits
===========
Credit for the vast majority of code here goes to the Tensorflow team. I've merely created a wrapper around all of the important functions to get people started.
