import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocessing import get_data

def main():
    training_images, training_labels, testing_images, testing_labels = get_data("../data")
    print(training_images)
    print(training_labels)

if __name__ == '__main__':
    main()
