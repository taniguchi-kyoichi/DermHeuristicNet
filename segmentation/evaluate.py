import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from model.dual_stream import DualStreamModel
from model.data_generator import DualStreamGenerator
from config import Config


def load_best_model():
    """ベストモデルのロード"""
    config = Config()

    # モデルの構築
    model = DualStreamModel(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    model_inst = model.build()
    model_inst = model.compile_model(model_inst)

    # ベストウェイトのロード
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model_v1.h5')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model weights not found at {best_model_path}")

    model_inst.load_weights(best_model_path)
    return model_inst


def plot_confusion_matrix(y_true, y_pred, class_names):
    """混同行列のプロット"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # 保存先ディレクトリの作成
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig('evaluation_results/confusion_matrix.png')
    plt.close()


def evaluate_model():
    """モデルの評価"""
    config = Config()

    # テストデータジェネレータの作成
    test_generator = DualStreamGenerator(
        data_dir=config.TEST_DIR,
        seg_dir=config.TEST_SEG_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        is_training=False
    )

    # モデルのロード
    model = load_best_model()

    # 予測と評価
    print("Evaluating model...")
    test_metrics = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Loss: {test_metrics[0]:.4f}")
    print(f"Test Accuracy: {test_metrics[1]:.4f}")

    # クラスごとの予測を取得
    print("\nGenerating predictions...")
    all_predictions = []
    all_true_labels = []

    for i in range(len(test_generator)):
        batch_data = test_generator[i]
        predictions = model.predict(batch_data[0], verbose=0)
        all_predictions.extend(predictions)
        all_true_labels.extend(batch_data[1])

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # 予測クラスの取得
    y_pred = np.argmax(all_predictions, axis=1)
    y_true = np.argmax(all_true_labels, axis=1)

    # 混同行列の生成とプロット
    plot_confusion_matrix(y_true, y_pred, config.CLASS_NAMES)

    # クラスごとの詳細な評価メトリクス
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    # 結果をファイルに保存
    with open('evaluation_results/classification_report.txt', 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=======================\n\n")
        f.write(f"Test Loss: {test_metrics[0]:.4f}\n")
        f.write(f"Test Accuracy: {test_metrics[1]:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # クラスごとの予測確率分布
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(config.CLASS_NAMES):
        class_probs = all_predictions[:, i]
        plt.hist(class_probs, bins=50, alpha=0.5, label=class_name)

    plt.title('Prediction Probability Distribution by Class')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('evaluation_results/prediction_distribution.png')
    plt.close()

    return {
        'test_loss': test_metrics[0],
        'test_accuracy': test_metrics[1],
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions': all_predictions
    }


if __name__ == "__main__":
    results = evaluate_model()