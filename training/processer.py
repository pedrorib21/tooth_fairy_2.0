import pandas as pd
import json


class Processer:
    def __init__(self, labels_to_identify: list) -> None:
        self.labels_to_identify = labels_to_identify

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        df["labels"] = df["label_file"].apply(
            lambda file_path: self.read_and_parse_json(file_path)["labels"]
        )
        df["has_selected_teeth"] = df["labels"].apply(
            lambda labels: set(self.labels_to_identify).issubset(labels)
        )
        final_df = df[df["has_selected_teeth"]]
        final_df["labels_of_interest"] = final_df["labels"].apply(
            lambda labels: [
                label if label in self.labels_to_identify else 0 for label in labels
            ]
        )

        return final_df

    def read_and_parse_json(self, filepath: str) -> dict:
        with open(filepath, "r") as f:
            parsed_json = json.load(f)
        return parsed_json
