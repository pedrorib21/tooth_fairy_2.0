import os
import argparse
from training.datagenerator import DataGenerator
import tensorflow as tf
from models import pointnet_seg
import json
import numpy as np
import pandas as pd
from vedo import Points, settings, show, Plotter, Mesh
from training.processer import Processer
from training.metrics import f1_metric

parser = argparse.ArgumentParser(description="")

# Add the arguments
parser.add_argument(
    "--data_dir", "-d", type=str, help="data directory", default="data/lower"
)
parser.add_argument(
    "--model_path", "-mp", type=str, help="model to evaluate directory", required=True
)
parser.add_argument(
    "--num_points", "-np", type=int, help="num points to sample", default=10000
)
parser.add_argument("--teeth_to_identify", nargs="+", type=str, help="teeth to segment")

args = parser.parse_args()

LOWER_PATH = args.data_dir
MODEL_PATH = args.model_path
NUM_POINTS = args.num_points
LABELS_TO_IDENTIFY = args.teeth_to_identify
with open("data/labels.json", "r") as f:
    teeth_name_label_dict = json.load(f)

LABELS_TO_IDENTIFY = [teeth_name_label_dict[name] for name in args.teeth_to_identify]

with open("data/Teeth3DS_train_test_split/testing_lower.txt", "r") as f:
    test_ex = f.read().splitlines()

folders = os.listdir(LOWER_PATH)

test_obj_files = [
    os.path.join(LOWER_PATH, f, f"{f}_lower.obj")
    for f in folders
    if f"{f}_lower" in test_ex
]
test_label_files = [
    os.path.join(LOWER_PATH, f, f"{f}_lower.json")
    for f in folders
    if f"{f}_lower" in test_ex
]
assert len(test_obj_files) == len(test_label_files)

# Process Data
df = pd.DataFrame({"obj_file": test_obj_files, "label_file": test_label_files})
processer = Processer(LABELS_TO_IDENTIFY)
processed_df = processer.preprocessing(df)

test_generator = DataGenerator(
    processed_df["obj_file"].values,
    processed_df["labels_of_interest"].values,
    num_classes=len(LABELS_TO_IDENTIFY),
)
num_classes = len(LABELS_TO_IDENTIFY) + 1
model = pointnet_seg.get_shape_segmentation_model(NUM_POINTS, num_classes=num_classes)
model.load_weights(MODEL_PATH).expect_partial()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1_metric])

prediction = model.evaluate(test_generator)