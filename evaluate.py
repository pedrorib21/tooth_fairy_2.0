import os
import argparse
from training.datagenerator import DataGenerator
import tensorflow as tf
from models import pointnet_seg
import json
import numpy as np
import pandas as pd
from vedo import Points, settings, show, Plotter, Mesh

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
with open("labels.json", "r") as f:
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

df = pd.DataFrame({"obj_file": test_obj_files, "label_file": test_label_files})
read_and_parse_json = lambda file_path: json.load(open(file_path, "r"))
df["labels"] = df["label_file"].apply(
    lambda file_path: read_and_parse_json(file_path)["labels"]
)
df["has_selected_teeth"] = df["labels"].apply(
    lambda labels: set(LABELS_TO_IDENTIFY).issubset(labels)
)
df_to_test = df[df["has_selected_teeth"]]
df_to_test["test_labels"] = df_to_test["labels"].apply(
    lambda labels: [label if label in LABELS_TO_IDENTIFY else 0 for label in labels]
)

test_generator = DataGenerator(
    df_to_test["obj_file"].values,
    df_to_test["test_labels"].values,
    num_classes=len(LABELS_TO_IDENTIFY),
)
num_classes = len(LABELS_TO_IDENTIFY) + 1
model = pointnet_seg.get_shape_segmentation_model(NUM_POINTS, num_classes=num_classes)
model.load_weights(MODEL_PATH).expect_partial()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


for example in test_generator:

    prediction = model.predict(example[0])

    pred = np.argmax(prediction, axis=-1)[0]

    print(np.unique(pred, return_counts=True))

    point_cloud = Points(example[0][0])
    point_cloud.cmap("jet", prediction[0, :, 0])
    # Show the plot
    Plotter()
    show(point_cloud)

    # # Print the evaluation results
    # print("Loss:", evaluation_results[0])
    # print("Accuracy:", evaluation_results[1])

