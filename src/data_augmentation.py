import cv2
import os
import time
import uuid
import json
import albumentations as alb
import numpy as np

# Initial Step
"""
1. Created a 'data' folder in the current directory with the following folders: 'images', 'labels', 'train', 'test', 'val'
2. Added an 'images' and 'labels' directory in each of 'train', 'test' and 'val'
3. Created an 'aug_data' folder in the current directory with the following folders: 'train', 'test', 'val'
4. Repeated step 2. on the 'aug_data' (augmented data) folder
"""
    
# Gathering Data (pictures of myself)
images_path = os.path.join('data', 'images')
num_pics = 100

camera = cv2.VideoCapture(0)
for i in range(num_pics):
    ret_val, image = camera.read()
    imgname = os.path.join(images_path, str(uuid.uuid1()) + '.jpg')
    cv2.imwrite(imgname, image)
    cv2.imshow('image', image)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

# Used labelme to create labels - annotated images with a bounding box
# Partitioned unaugmented imaged (done manually - 70 train, 15 test, 15 validation)

# Script to move corresponding labels 
data_types = ['train', 'test', 'val']

for folder in data_types:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        
        label_file = file.split('.')[0]+'.json'
        existing_path = os.path.join('data','labels', label_file)
        if os.path.exists(existing_path): 
            new_path = os.path.join('data',folder,'labels',label_file)
            os.replace(existing_path, new_path) 

# Augmentation pipeline
transform = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5),
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

# Create Augmented Data
for type in data_types:
    for image in os.listdir(os.path.join('data', type, 'images')):
        
        img = cv2.imread(os.path.join('data', type, 'images', image))
        h = img.shape[0]
        w = img.shape[1]
        coords = [0,0,0.00001,0.00001] # deafult coords (near 0) if label does not exist
        label_path = os.path.join('data', type, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as label_file:
                label = json.load(label_file)

        coords[0] = label['shapes'][0]['points'][0][0] # x1
        coords[1] = label['shapes'][0]['points'][0][1] # y1
        coords[2] = label['shapes'][0]['points'][1][0] # x2
        coords[3] = label['shapes'][0]['points'][1][1] # y2
        
        coords = list(np.divide(coords, [w, h, w, h])) # put coords in albumentations format

        try: # augmenting 60 images from one
            for i in range(60):
                augmented = transform(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', type, 'images', f'{image.split(".")[0]}.{i}.jpg'), augmented['image'])

                aug_label_data = {}
                aug_label_data['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: # no bounding box in the augmented image
                        aug_label_data['bbox'] = [0, 0, 0, 0]
                        aug_label_data['class'] = 0
                    else:
                        aug_label_data['bbox'] = augmented['bboxes'][0]
                        aug_label_data['class'] = 1
                else:
                    aug_label_data['bbox'] = [0, 0, 0, 0]
                    aug_label_data['class'] = 0

                with open(os.path.join('aug_data', type, 'labels', f'{image.split(".")[0]}.{i}.json'), 'w') as aug_label_file:
                    json.dump(aug_label_data, aug_label_file)
                    
        except Exception as e:
            print(e)
