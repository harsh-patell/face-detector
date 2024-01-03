# Find My Face

A facial recognition application developed in Python with TensorFlow and OpenCV. This application leverages an object detection architecture and can
easily be expanded to identify other items. **Notable details:**
* Captured, annotated (labelled with [labelme](http://labelme2.csail.mit.edu/Release3.0/index.php)) and augmented (using [Albumentations](https://albumentations.ai/)) personal image data
* Leveraged the VGG16 convolutional neural network to create a detection model with classification and regression layers
* Utilized the Adam optimizer and binary cross entropy loss function to reduce the classification loss
