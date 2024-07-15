# Find My Face

A facial recognition application developed in Python with TensorFlow and OpenCV. This application leverages an object detection architecture and can
easily be expanded to identify other items.
* Captured, annotated (labelled with [labelme](http://labelme2.csail.mit.edu/Release3.0/index.php)) and augmented (using [Albumentations](https://albumentations.ai/)) personal image data
* Leveraged the VGG16 convolutional neural network to create a detection model with classification and regression layers
* Utilized the Adam optimizer and binary cross entropy loss function to reduce the classification loss

## Directory Structure
`src/data_augmentation.py` - captures, labels and augments data into training and test sets  
`src/face_tracker.py` -  model inherited from Keras Model class  
`src/model.py` -  creates and trains the model with classification and regression layers from the augmented data  
`src/detection.py` -  detection with OpenCV  
`main.ipynb` - jupyter notebook file with the entire code

## How To Run
1. Setup project directory as outlined in `src/data_augmentation.py`
2. Run the cells in `main.ipynb`
