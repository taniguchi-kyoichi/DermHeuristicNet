import os
from preprocess import process_dataset
from train import train_model
from config import Config


def main():
    config = Config()

    # Configure paths
    zip_path = "datasets/HAM10000.zip"  # Update with actual path
    metadata_path = "datasets/HAM10000_metadata.csv"  # Update with actual path

    # データの前処理
    print("Starting data preprocessing...")
    data_pd, train_df, test_df = process_dataset(zip_path, metadata_path, config.IMAGE_SIZE)
    print("Preprocessing completed!")

    # データの準備完了を確認
    if not all(os.path.exists(path) for path in [config.TRAIN_DIR, config.TEST_DIR,
                                                 config.TRAIN_SEG_DIR, config.TEST_SEG_DIR]):
        raise Exception("Required directories not found!")

    # モデルの学習
    print("\nStarting model training...")
    model, history = train_model(train_df, test_df)
    print("Training completed!")

    return model, history


if __name__ == "__main__":
    model, history = main()