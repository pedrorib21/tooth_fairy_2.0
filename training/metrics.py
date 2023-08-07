import tensorflow as tf
from sklearn.metrics import f1_score


def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)  # Convert probabilities to binary predictions

    true_positives = tf.reduce_sum(y_true * y_pred, axis=0)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    macro_f1 = tf.reduce_mean(f1)

    return macro_f1
