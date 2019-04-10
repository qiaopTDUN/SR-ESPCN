# SR-ESPCN
Tensorflow implementation of ‘Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network’
We have not tested the trained model to verify its performance with the reported performance in [1].

Windows7 / Ubuntu 14.04 + CUDA8.0 + CUDNN 5.1 + Tensorflow 1.4

run main.py

Before run, please prepare the training dataset. We have tried ImageNet and DIV2K.
Then, modify the Line 13 to 28 in main.py to the parameters you want to try.

# Train
set is_train to True
set batch_size to fit your GPU memory, larger is better
set image_size to fit your GPU memory, larger is better, but harder to train

It may take some time before the training is done.

# Test
set is_train to False

[1] W. Shi, J. Caballero, F. Huszar, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1874–1883, 2016.
