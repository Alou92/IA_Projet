import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.datasets import cifar10


class DataManager(object):

    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.eval_data = []
        self.eval_labels = []
        self.loadData()

    def loadData(self):		#chargement des données
        (train_data, train_labels), (eval_data, eval_labels) = cifar10.load_data()
        self.train_data = train_data/255.0
        self.train_labels = train_labels
        self.eval_data = eval_data/255.0
        self.eval_labels = eval_labels


    def preprocessData(self):	#Generateur des parametres de données

        self.datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)

        self.datagen.fit(self.train_data)

        # https://keras.io/preprocessing/image/ data augmentation.
        """Load the data from cifar-10-batches. 
           See http://www.cs.toronto.edu/~kriz/cifar.html for instructions on 
           how to do so.
        """
        