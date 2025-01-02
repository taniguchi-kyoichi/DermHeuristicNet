import os
import shutil
import sys
import platform
import tensorflow as tf
import pandas as pd
from preprocess import process_dataset
from train import train_model
from evaluate import evaluate_model
from config import Config


def setup_m1_tensorflow():
    """M1 Mac向けのTensorFlow設定"""
    try:
        if platform.processor() == 'arm':
            print("M1 Mac detected. Configuring TensorFlow...")

            # デバイスの確認
            devices = tf.config.list_physical_devices()
            print("Available devices:", devices)

            # GPU (Metal) デバイスの確認
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print("Metal device found. Configuring for GPU execution...")

                # デバイス設定
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        # メモリ成長の設定に失敗しても続行
                        pass

                return True
            else:
                print("No Metal device found. Running on CPU...")
                return False
        else:
            print("Not an M1 Mac. Running on available hardware...")
            return False

    except Exception as e:
        print(f"Error during TensorFlow configuration: {str(e)}")
        print("Falling back to CPU execution...")
        return False


def check_directories(config):
    """必要なディレクトリの存在確認"""
    required_dirs = [
        config.TRAIN_DIR,
        config.TEST_DIR,
        config.TRAIN_SEG_DIR,
        config.TEST_SEG_DIR,
        config.TRAIN_COLOR_DIR,
        config.TEST_COLOR_DIR,
        config.MODEL_DIR,
        config.LOG_DIR
    ]

    missing_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing_dirs.append(directory)

    if missing_dirs:
        print("Following directories are missing:")
        for dir_path in missing_dirs:
            print(f"- {dir_path}")
        return False

    return True


def check_class_directories(config):
    """クラスディレクトリの存在確認"""
    parent_dirs = [
        config.TRAIN_DIR,
        config.TEST_DIR,
        config.TRAIN_SEG_DIR,
        config.TEST_SEG_DIR,
        config.TRAIN_COLOR_DIR,
        config.TEST_COLOR_DIR
    ]

    for parent_dir in parent_dirs:
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(parent_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Missing class directory: {class_dir}")
                return False

    return True


def setup_memory_config():
    """メモリ使用量の設定"""
    try:
        # M1 Mac向けのメモリ設定
        if platform.processor() == 'arm':
            print("Configuring memory settings for M1 Mac...")

            # バッチサイズとキャッシュの制限
            tf.config.experimental.set_lms_enabled(True)  # Long-term Memory Storage
            tf.data.experimental.enable_debug_mode()

            # GPU使用可能な場合はメモリ制限を設定
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        # より控えめなメモリ制限を設定
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
                        )
                    except:
                        # 設定に失敗しても続行
                        pass
    except Exception as e:
        print(f"Note: Using default memory settings: {str(e)}")


def main():
    """メイン実行関数"""
    try:
        config = Config()
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        raise

    # Configure paths
    zip_path = "datasets/HAM10000.zip"
    metadata_path = "datasets/HAM10000_metadata.csv"

    try:
        print("\nChecking system configuration...")
        print(f"Python version: {platform.python_version()}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Processor: {platform.processor()}")

        # M1 Mac向けの設定
        if platform.processor() == 'arm':
            metal_available = setup_m1_tensorflow()
            print(f"Metal acceleration {'enabled' if metal_available else 'not available'}")
            setup_memory_config()

        # Check if dataset exists
        if not check_directories(config) or not check_class_directories(config):
            print("\nInitiating dataset preprocessing...")
            data_pd, train_df, test_df = process_dataset(zip_path, metadata_path, config.IMAGE_SIZE)
            print("Preprocessing completed!")
        else:
            print("\nUsing existing preprocessed dataset")
            data_pd = pd.read_csv(metadata_path)
            train_df = data_pd[data_pd['train_test_split'] == 'train']
            test_df = data_pd[data_pd['train_test_split'] == 'test']

        # Verify preprocessing results
        if not check_directories(config) or not check_class_directories(config):
            raise Exception("Dataset preprocessing failed to create required directory structure!")

        # Enable mixed precision for better performance on M1
        if platform.processor() == 'arm':
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Enabled mixed precision training")

        # Train model
        print("\nStarting model training...")
        model, history = train_model(train_df, test_df)
        print("Training completed!")

        # Evaluate model
        print("\nStarting model evaluation...")
        evaluation_results = evaluate_model()
        print("Evaluation completed!")

        # Print summary results
        print("\nFinal Results:")
        print(f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}")
        print(f"Test Loss: {evaluation_results['test_loss']:.4f}")

        return model, history, evaluation_results

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        model, history, evaluation_results = main()
        print("\nExecution completed successfully!")
    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
        sys.exit(1)