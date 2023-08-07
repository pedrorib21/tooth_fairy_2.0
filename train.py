import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from vedo import Points, settings, show, Plotter, Mesh
import pyvista as pv
import vtk

from models import pointnet_seg
from training.callbacks import model_checkpoint, tensorboard_callback
from training.datagenerator import DataGenerator
from training.metrics import f1_metric

parser = argparse.ArgumentParser(description="")

# Add the arguments
parser.add_argument(
    "--data_dir", "-d", type=str, help="data directory", default="data/lower"
)
parser.add_argument("--epochs", type=int, help="epochs to train the model", default=10)
parser.add_argument(
    "--num_points", "-np", type=int, help="num points to sample", default=10000
)
parser.add_argument("--teeth_to_identify", nargs="+", type=str, help="teeth to segment")
args = parser.parse_args()

LOWER_PATH = args.data_dir
EPOCHS = args.epochs
NUM_POINTS = args.num_points
LABELS_TO_IDENTIFY = args.teeth_to_identify

with open("labels.json", "r") as f:
    teeth_name_label_dict = json.load(f)

LABELS_TO_IDENTIFY = [teeth_name_label_dict[name] for name in args.teeth_to_identify]

## Open train examples
with open("data/Teeth3DS_train_test_split/training_lower.txt", "r") as f:
    train_ex = f.read().splitlines()

folders = os.listdir(LOWER_PATH)
obj_files = [
    os.path.join(LOWER_PATH, f, f"{f}_lower.obj")
    for f in folders
    if f"{f}_lower" in train_ex
]
label_files = [
    os.path.join(LOWER_PATH, f, f"{f}_lower.json")
    for f in folders
    if f"{f}_lower" in train_ex
]
assert len(obj_files) == len(label_files)

df = pd.DataFrame({"obj_file": obj_files, "label_file": label_files})
read_and_parse_json = lambda file_path: json.load(open(file_path, "r"))
df["labels"] = df["label_file"].apply(
    lambda file_path: read_and_parse_json(file_path)["labels"]
)
df["has_selected_teeth"] = df["labels"].apply(
    lambda labels: set(LABELS_TO_IDENTIFY).issubset(labels)
)
df_to_train = df[df["has_selected_teeth"]]
df_to_train["train_labels"] = df_to_train["labels"].apply(
    lambda labels: [label if label in LABELS_TO_IDENTIFY else 0 for label in labels]
)
# Split data to train
train_df, val_df = train_test_split(
    df_to_train,
    test_size=0.2,
    random_state=42,
)

training_generator = DataGenerator(
    train_df["obj_file"].values,
    train_df["train_labels"].values,
    num_classes=len(LABELS_TO_IDENTIFY),
)
validation_generator = DataGenerator(
    val_df["obj_file"].values,
    val_df["train_labels"].values,
    num_classes=len(LABELS_TO_IDENTIFY),
)

# Train
num_classes = len(LABELS_TO_IDENTIFY) + 1
model = pointnet_seg.get_shape_segmentation_model(NUM_POINTS, num_classes=num_classes)
print(model.summary())
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1_metric]
)

current_time = time.strftime("%H_%M_%S_%d_%m_%Y")
model_path = f"data/tmp/models/{current_time}/"
log_path = f"data/tmp/logs/{current_time}/"
model.fit(
    training_generator,
    validation_data=validation_generator,
    verbose=1,
    epochs=EPOCHS,
    callbacks=[model_checkpoint(model_path), tensorboard_callback(log_path)],
)
print(f"Model saved in {model_path}")
