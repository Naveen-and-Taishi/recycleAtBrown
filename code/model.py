import os
import tensorflow as tf
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from preprocessing import get_data

class Model(tf.keras.Model):
    def __init__(self):
        """
        """
        super(Model, self).__init__()

        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = [] 


        # TODO: Initialize all hyperparameters
        self.learning_rate = 0.0025
        self.drop_rate = 0.3

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
       

    def loss(self, logits, labels):
        """
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

def train(model, train_inputs, train_labels):
    '''
    '''
    
    
def test(model, test_inputs, test_labels):
    """
    """
    
def main():
    training_images, training_labels, testing_images, testing_labels = get_data("../data")
    print(training_images)
    print(training_labels)

if __name__ == '__main__':
    main()