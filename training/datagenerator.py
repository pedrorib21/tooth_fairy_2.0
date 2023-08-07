import json
import numpy as np
import tensorflow as tf
from vtk.util import numpy_support

from dataloading import loader
import random


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        objs_filenames,
        labels,
        num_classes,
        batch_size=1,
        shuffle=True,
        sample_points=10000,
    ):
        self.objs_filenames = objs_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.num_sample_points = sample_points
        self.num_classes = num_classes

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.objs_filenames) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        objs_filenames_temp = [self.objs_filenames[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(objs_filenames_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.objs_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, objs_filenames_temp, labels_temp):
        # Generate data
        for i, (obj_file, labels) in enumerate(
            zip(objs_filenames_temp, labels_temp)
        ):
            # Store sample
            polydata = loader.read_obj(obj_file)
            polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
            sampled_indices = random.sample(
                list(range(len(polydata_points))), self.num_sample_points
            )

            polydata_points = polydata_points[sampled_indices]

            X = np.array(polydata_points)[np.newaxis, ...]

            labels = np.array(labels)[sampled_indices]
            for i, l in enumerate(np.unique(labels)):
                labels[labels==l] = i

            one_hot_labels = self._class_labels_to_one_hot(labels)

            y = one_hot_labels[np.newaxis, ...]


        return X, y

    # Convert class labels to one-hot encoded arrays
    def _class_labels_to_one_hot(self, labels):
        one_hot_labels = np.zeros((self.num_sample_points, self.num_classes+1))
        one_hot_labels[np.arange(self.num_sample_points), labels] = 1

        return one_hot_labels
