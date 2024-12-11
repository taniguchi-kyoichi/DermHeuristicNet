# evaluate.py
import numpy as np
from sklearn.metrics import (classification_report, precision_score, recall_score,
                             accuracy_score, roc_curve, auc, roc_auc_score)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def load_model_weights(model, weights_path):
    """Load trained model weights"""
    model.load_weights(weights_path)
    return model


def create_test_generator(test_dir, image_size, batch_size):
    """Create test data generator"""
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
    )

    test_batches = datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False
    )

    return test_batches


def evaluate_model(model, weights_path='IRV2+SA.hdf5', image_size=299, batch_size=16):
    """Evaluate model performance with various metrics"""
    # Load weights
    model = load_model_weights(model, weights_path)

    # Create test generator
    test_dir = 'HAM10000/test_dir'
    test_batches = create_test_generator(test_dir, image_size, batch_size)

    # Get predictions
    predictions = model.predict(test_batches, steps=len(test_batches), verbose=0)

    targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    y_true = test_batches.classes
    y_pred = np.argmax(predictions, axis=1)
    y_test = to_categorical(y_true)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=targetnames)
    print("\nClassification Report:")
    print(report)

    # Weighted metrics
    print("\nWeighted Metrics:")
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, predictions, multi_class='ovr', average='weighted'))

    # Macro metrics
    print("\nMacro Metrics:")
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, predictions, multi_class='ovr', average='macro'))

    # Micro metrics
    print("\nMicro Metrics:")
    print("Precision:", precision_score(y_true, y_pred, average='micro'))
    print("Recall:", recall_score(y_true, y_pred, average='micro'))
    print("Accuracy:", accuracy_score(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_test.ravel(), predictions.ravel())
    roc_auc = auc(fpr, tpr)
    print("ROC AUC Score:", roc_auc)


if __name__ == "__main__":
    from train import create_model

    model = create_model()
    evaluate_model(model)