import tensorflow as tf


def model_checkpoint(path: str, monitor: str = "val_accuracy"):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path, monitor=monitor, save_best_only=True, save_weights_only=True
    )
    return model_checkpoint_callback


def tensorboard_callback(log_dir, freq=1):
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=freq)
