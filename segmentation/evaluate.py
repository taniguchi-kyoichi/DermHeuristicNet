import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from model.triple_stream import TripleStreamModel
from model.data_generator import TripleStreamGenerator
from config import Config


def load_best_model():
    """Load the best trained model"""
    config = Config()

    # Build model
    model = TripleStreamModel(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    model_inst = model.build()
    model_inst = model.compile_model(model_inst)

    # Find best model weights
    model_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.h5')]
    if not model_files:
        raise FileNotFoundError(f"No model weights found in {config.MODEL_DIR}")

    # Get the model with highest validation accuracy
    best_model_file = max(model_files, key=lambda x: float(x.split('_')[-1].replace('.h5', '')))
    best_model_path = os.path.join(config.MODEL_DIR, best_model_file)

    print(f"Loading best model: {best_model_file}")
    model_inst.load_weights(best_model_path)

    return model_inst


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Save raw confusion matrix
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)


def plot_prediction_distribution(predictions, class_names, output_dir):
    """Plot prediction probability distribution by class"""
    plt.figure(figsize=(15, 8))
    for i, class_name in enumerate(class_names):
        class_probs = predictions[:, i]
        plt.hist(class_probs, bins=50, alpha=0.5, label=class_name)

    plt.title('Prediction Probability Distribution by Class')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'))
    plt.close()


def analyze_error_cases(X_test, y_true, y_pred, predictions, class_names, output_dir):
    """Analyze and save information about misclassified cases"""
    misclassified_indices = np.where(y_true != y_pred)[0]

    error_analysis = []
    for idx in misclassified_indices:
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        confidence = predictions[idx][y_pred[idx]]

        error_analysis.append({
            'index': idx,
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probabilities': predictions[idx].tolist()
        })

    # Save error analysis
    error_df = pd.DataFrame(error_analysis)
    error_df.to_csv(os.path.join(output_dir, 'error_analysis.csv'), index=False)

    # Create error distribution visualization
    plt.figure(figsize=(12, 6))
    error_counts = error_df.groupby(['true_class', 'predicted_class']).size().unstack(fill_value=0)
    sns.heatmap(error_counts, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Error Distribution Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()


def evaluate_model():
    """Evaluate the triple stream model"""
    config = Config()
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Create test data generator
    test_generator = TripleStreamGenerator(
        data_dir=config.TEST_DIR,
        seg_dir=config.TEST_SEG_DIR,
        color_dir=config.TEST_COLOR_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        is_training=False
    )

    # Load model
    model = load_best_model()

    # Evaluate model
    print("Evaluating model...")
    test_metrics = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Loss: {test_metrics[0]:.4f}")
    print(f"Test Accuracy: {test_metrics[1]:.4f}")

    # Generate predictions
    print("\nGenerating predictions...")
    all_predictions = []
    all_true_labels = []
    all_test_data = []

    for i in range(len(test_generator)):
        batch_data = test_generator[i]
        predictions = model.predict(batch_data[0], verbose=0)
        all_predictions.extend(predictions)
        all_true_labels.extend(batch_data[1])
        all_test_data.extend(batch_data[0])

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    all_test_data = np.array(all_test_data)

    # Get predicted classes
    y_pred = np.argmax(all_predictions, axis=1)
    y_true = np.argmax(all_true_labels, axis=1)

    # Generate and save visualizations
    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES, output_dir)
    plot_prediction_distribution(all_predictions, config.CLASS_NAMES, output_dir)
    analyze_error_cases(all_test_data, y_true, y_pred, all_predictions, config.CLASS_NAMES, output_dir)

    # Generate classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )

    # Save results
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write("Triple Stream Model Evaluation Results\n")
        f.write("===================================\n\n")
        f.write(f"Test Loss: {test_metrics[0]:.4f}\n")
        f.write(f"Test Accuracy: {test_metrics[1]:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Calculate and save per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(config.CLASS_NAMES):
        class_mask = y_true == i
        class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
        class_predictions = all_predictions[class_mask]
        class_confidence = np.mean(np.max(class_predictions, axis=1))

        class_metrics[class_name] = {
            'accuracy': class_accuracy,
            'average_confidence': class_confidence,
            'sample_count': np.sum(class_mask)
        }

    # Save class metrics
    class_metrics_df = pd.DataFrame(class_metrics).T
    class_metrics_df.to_csv(os.path.join(output_dir, 'class_metrics.csv'))

    return {
        'test_loss': test_metrics[0],
        'test_accuracy': test_metrics[1],
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions': all_predictions,
        'class_metrics': class_metrics
    }


if __name__ == "__main__":
    results = evaluate_model()