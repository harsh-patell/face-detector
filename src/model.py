
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import json
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from keras.applications import VGG16
from .face_tracker import FaceTracker

def load_image(file):
    encoded = tf.io.read_file(file)
    img = tf.io.decode_jpeg(encoded)
    return img

def load_label(file):
    with open(file.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']
    
def create_model():
    input_layer = Input(shape=(120,120,3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    class0 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(class0)
    class2 = Dense(1, activaion='sigmoid')(class1)

    # Regression (Bounding Box) Model
    reg0 = GlobalMaxPooling2D()(vgg)
    reg1 = Dense(2048, activation='relu')(reg0)
    reg2 = Dense(4, activation='sigmoid')(reg1)

    facetracker = Model(inputs=input_layer, outputs=[class2, reg2])
    return facetracker

def localization_loss(act, pred):
    delta_coord = tf.reduce_sum(tf.square(act[:,:2] - pred[:,:2]))

    h_act = act[:,3] - act[:,1]
    w_act = act[:,2] - act[:,0]
    h_pred = pred[:,3] - pred[:,1]
    w_pred = pred[:,2] - pred[:,0]
    delta_size = tf.reduce_sum(tf.square(w_act - w_pred) + tf.square(h_act - h_pred))

    return delta_coord + delta_size

# Loading augmented data into tensorflow dataset
data_types = ['train', 'test', 'val']
aug_labels = {}
aug_images = {}
aug_data = {}

for type in data_types:
    type_images = tf.data.Dataset.list_files(f'aug_data/{type}/images/*.jpg', shuffle=False)
    type_images = type_images.map(load_image)
    type_images = type_images.map(lambda x: tf.image.resize(x, (120,120)))
    type_images = type_images.map(lambda x: x/255)
    aug_images[type] = type_images

for type in data_types:
    type_labels = tf.data.Dataset.list_files(f'aug_data/{type}/labels/*.json', shuffle=False)
    type_labels = type_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16])) 
    aug_labels[type] = type_labels

for type in data_types:
    data = tf.data.Dataset.zip((aug_images[type], aug_labels[type]))
    if (type == 'train'):
        data = data.shuffle(5000)
    elif (type == 'test'):
        data = data.shuffle(1000)
    else:
        data = data.shuffle(1000)
    data = data.batch(8)
    data = data.prefetch(4) # improve latency and throughput
    aug_data[type] = data

# Losses (classification and regression) and optimizers
batches_per_epoch = len(aug_data['train'])
decay = (1./0.75 -1)/batches_per_epoch
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=decay)

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# Logging
logdir='logs'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Training model
facetracker = create_model()
model = FaceTracker(facetracker)

model.compile(opt, classloss, regressloss)
model.fit(aug_data['train'].take(100), epochs=10, validation_data=aug_data['val'], callbacks=[tensorboard_cb])
