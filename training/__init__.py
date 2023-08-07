from typing import Optional

import pandas as pd
import tensorflow as tf


class CustomDataset:
    def __init__(
        self,
        preprocesser,
        seed: Optional[int] = 122164,
    ):
        self.preprocesser = preprocesser
        self.seed = seed

    def create(
        self,
        meta_df: pd.DataFrame,
    ) -> tf.data.Dataset:



        return ds