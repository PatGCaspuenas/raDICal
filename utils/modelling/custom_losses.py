# PACKAGES
import tensorflow as tf

class MeanSquaredError(tf.keras.losses.MeanSquaredError):
    """Provides mean squared error metrics: loss / residuals.
    Use mean squared error for regression problems with one or more outputs.
    """

    def residuals(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return y_true - y_pred

def energy_loss(input_img, decoded):
    """
    Computes Cumulative Energetic Accuracy

    :param input_img: ground truth
    :param decoded: reconstructed
    :return: CEA
    """
    return tf.keras.backend.sum(tf.keras.backend.square(input_img - decoded)) / tf.keras.backend.sum(tf.keras.backend.square(input_img))

def null_loss(input_img, decoded):
    """
    Null loss

    :param input_img: ground truth
    :param decoded: reconstructed
    :return: 0
    """
    return 0