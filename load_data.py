import os
import sys
import cv2
import csv
import random
import numpy as np

def load_data(load_type, image_dir, img_shape, keep_prob=1., grayscale=False):
    assert(load_type in ['test', 'dev', 'train'])

    x = []
    y = []

    #TODO modify this to load our data
    # load from image_dir/train/train_examples.csv
    csv_path = os.path.join(image_dir, load_type, load_type + "_examples.csv")
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        im_means = []
        for line in reader:
            path, label = line
            assert(os.path.exists(path))

            if random.random() < keep_prob:
                img = cv2.resize(cv2.imread(path), img_shape) / 256.
                if grayscale:
                    img = np.expand_dims(img[:,:,0], 3)
                im_means.append(np.mean(img))

                x.append(img)
                y.append(float('cancer' in label))

    # Less memory intensive way to subtract mean from images
    x = np.array(x)
    mean = np.mean(im_means)
    x -= mean
    y = np.array(y).reshape(len(y), 1)

    print("Data for {} set with {} examples. {}% have positive labels and {}% have negative labels.".format(load_type, x.shape[0], np.mean(y), 1 - np.mean(y)))
    return x, y
  
