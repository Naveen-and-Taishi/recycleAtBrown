import os
import tensorflow as tf
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from preprocessing import get_data
from tensorflow.keras import Sequential
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, MaxPooling2D, Dropout, Conv2D, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

class Referee(tf.keras.Model):
    def __init__(self) -> None:
        super(Referee, self).__init__()

        self.num_classes = 6
        self.learning_rate = 0.00001
        self.decay_rate = 0.00000025
        self.dropout_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = 100
        self.hidden_size = 500
        self.conv_size = 128 

        self.ResNet50 = ResNet50V2(include_top=False, 
            weights="imagenet",
            input_shape=(384, 512, 3))

        self.ResNet50.trainable = False

        self.referee = Sequential([
            Flatten(),
            Dense(self.hidden_size, activation="relu"),
            Dropout(self.dropout_rate),
            Dense(self.hidden_size, activation="relu"),
            Dropout(self.dropout_rate),
            Dense(6, activation="relu")
        ])

        # self.referee = Sequential([
        #     Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(self.conv_size, kernel_size=(3, 3), activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Flatten(),
        #     Dense(self.hidden_size, activation='relu'),
        #     Dense(self.hidden_size, activation='relu'),
        #     Dense(6, activation='softmax')])


    def call(self, inputs):
        # outputs a 1 by 1 by 20
        resnet_output = tf.stop_gradient(self.ResNet50(inputs))
        referee_output = self.referee(resnet_output)

        # this returns probability of input being one of the 20 images
        return referee_output
    
    def loss(self, probs, labels): 
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=True)
        return tf.reduce_sum(loss)
        
    def accuracy(self, logits, labels):
        """
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        :return: the accuracy of the model as a Tensor

        labels should be one-hot vector
        """
        print("logits", logits)
        # print("labels Shape", labels.shape)
        labelsOneHot = tf.one_hot(labels, self.num_classes)
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labelsOneHot, 1))
        #print(correct_predictions) # correct predictions should be a one-dimensional list of labels
        # print("dimension1: ", tf.argmax(logits, 1))
        # print("dimension1: ", tf.argmax(labelsOneHot, 1))
        # print("dimension2", tf.argmax(logits, 0))
        # print("dimension2", tf.argmax(labelsOneHot, 0))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    for epoch in range(1):
        print("EPOCH: " + str(epoch))
        for b in range(0, len(train_inputs), model.batch_size):
            print("batch: " + str(int(b / model.batch_size)))
            batch_inputs = tf.convert_to_tensor(train_inputs[b : b + model.batch_size], dtype=tf.int32)
            batch_labels = tf.convert_to_tensor(train_labels[b : b + model.batch_size], dtype=tf.int32)
            # batch_inputs = tf.convert_to_tensor(train_inputs[b : b + model.batch_size])
            # batch_labels = tf.convert_to_tensor(train_labels[b : b + model.batch_size])
            with tf.GradientTape() as tape:
                logits = model.call(batch_inputs)
                loss = model.loss(logits, batch_labels)
                accuracy = model.accuracy(logits, batch_labels)
                print("ACCURACY: " + str(accuracy))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
    
def test(model, test_inputs, test_labels):
    """
    """
    accuracy = 0 
    n = 0 
    num_examples = model.batch_size 
    for i in range(0, len(test_inputs), num_examples): 
        # print(i)
        images = tf.convert_to_tensor(test_inputs[i:i + num_examples], dtype=tf.int32)
        labels = tf.convert_to_tensor(test_labels[i:i + num_examples], dtype=tf.int32)
        # images = tf.convert_to_tensor(test_inputs[i:i + num_examples])
        # labels = tf.convert_to_tensor(test_labels[i:i + num_examples])
        # print(test_inputs[i].shape)
        predictions = model.call(images)
        loss = model.loss(predictions, labels)
        test_acc = model.accuracy(predictions, labels)
        n += 1 
        accuracy += test_acc 
    overallAcc = accuracy / n
    return overallAcc

def main():
    training_images, training_labels, testing_images, testing_labels = get_data("../data")

    referee = Referee()

    print("Starting training...")
    train(referee, training_images, training_labels)
    print("Starting testing...")
    final_acc = test(referee, testing_images, testing_labels)

    print("FINAL TESTING ACCURACY: " + str(final_acc))

    tf.saved_model.save(referee, "../models")
if __name__ == '__main__':
    main()