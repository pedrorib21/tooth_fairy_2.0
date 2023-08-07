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
        labels_filenames,
        batch_size=1,
        unique_labels=[],
        shuffle=True,
    ):
        "Initialization"
        self.objs_filenames = objs_filenames
        self.labels_filenames = labels_filenames
        self.batch_size = batch_size
        self.unique_labels = unique_labels
        self.label_idx_translator = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.n_classes = len(unique_labels)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.NUM_SAMPLE_POINTS = 10000

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.objs_filenames) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        objs_filenames_temp = [self.objs_filenames[k] for k in indexes]
        labels_filenames_temp = [self.labels_filenames[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(objs_filenames_temp, labels_filenames_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.objs_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, objs_filenames_temp, labels_filenames_temp):
        # Generate data
        for i, (obj_file, label_file) in enumerate(
            zip(objs_filenames_temp, labels_filenames_temp)
        ):
            # Store sample
            polydata = loader.read_obj(obj_file)
            polydata_points = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
            sampled_indices = random.sample(list(range(len(polydata_points))), self.NUM_SAMPLE_POINTS)

            polydata_points = polydata_points[sampled_indices]

            X = np.array(polydata_points)[np.newaxis, ...]

            with open(label_file, "r") as file:
                labels = json.load(file)
            # Store class
            labels = np.array(labels["labels"])

            labels = labels[sampled_indices]

            one_hot_labels = self._class_labels_to_one_hot(
                labels, len(self.unique_labels)
            )

            y = one_hot_labels[np.newaxis,...]

        return X, y

    # Convert class labels to one-hot encoded arrays
    def _class_labels_to_one_hot(self, class_labels, num_classes):
        one_hot_labels = np.zeros((len(class_labels), num_classes))
        translated_labels = [self.label_idx_translator[label] for label in class_labels]
        one_hot_labels[np.arange(len(class_labels)), translated_labels] = 1

        return one_hot_labels
