import tensorflow as tf
from keras import backend as K


class PrintBestIoUCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best_iou = -1  # Initialize with a value that's guaranteed to be lower than any valid IoU
        self.best_f1 = -1  # Initialize with a value that's guaranteed to be lower than any valid F1 score

    def on_epoch_end(self, epoch, logs=None):
        current_iou = logs.get('activation_24_one_hot_io_u',
                               -1)  # Get IoU from the logs, default to -1 if not found
        current_f1 = logs.get('activation_24_f1', -1)  # Get F1 from the logs, default to -1 if not found

        if current_iou > self.best_iou:
            self.best_iou = current_iou
            self.best_f1 = current_f1  # Record the F1 score when the best IoU is achieved
            print(f'Epoch {epoch + 1}: Best IoU improved to {current_iou:.4f}')


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    # Calculating metrics for each class
    num_classes = 3
    f1_scores = []
    for i in range(num_classes):
        class_true = y_true[..., i]
        class_pred = y_pred[..., i]
        class_precision = precision(class_true, class_pred)
        class_recall = recall(class_true, class_pred)
        class_f1 = 2 * ((class_precision * class_recall) / (class_precision + class_recall + K.epsilon()))
        f1_scores.append(class_f1)

    # Average F1 scores across all classes
    f1_scores_tensor = K.stack(f1_scores, axis=0)
    mean_f1 = K.mean(f1_scores_tensor)

    return mean_f1
