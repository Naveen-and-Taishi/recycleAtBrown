import random
from PIL import Image
import numpy as np
from os import listdir

def get_data(data_path):
    
    images = []

    dirs = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    # 0 - cardboard
    # 1 - glass
    # 2 - metal
    # 3 - paper
    # 4 - plastic
    # 5 - trash
    labels = []

    label_idx = 0

    # convert images into numpy arrays and store them in the images list
    for dir in dirs:
        print(dir)
        for f in listdir(data_path + "/" + dir):
            new_img = Image.open(data_path + "/" + dir + "/" + f)
            new_img_data = np.asarray(new_img)
            print(np.shape(new_img_data))
            images.append(new_img_data)
            labels.append(label_idx)
        label_idx += 1
    
    # shuffle images and labels
    shuffle_order = np.arange(len(images))
    np.random.shuffle(shuffle_order)

    print(shuffle_order)

    images = np.array(images)
    labels = np.array(labels)

    images = images[shuffle_order.astype(int)]
    labels = labels[shuffle_order.astype(int)]

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    # separate into training and testing sets
    for i in range(len(images)):
        n = random.randint(0, 3)
        if n == 0:
            # add to testing data
            testing_images.append(images[i])
            testing_labels.append(labels[i])
        else:
            # add to training data
            training_images.append(images[i])
            training_labels.append(labels[i])
    
    return training_images, training_labels, testing_images, testing_labels

