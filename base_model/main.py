# main.py
import os
from preprocess import process_dataset
from train import train_model


def main():
    # Configure paths
    zip_path = "datasets/HAM10000.zip"  # Update with actual path
    metadata_path = "datasets/HAM10000_metadata.csv"  # Update with actual path
    image_size = 299

    # データの前処理
    print("Starting data preprocessing...")
    data_pd, train_df, test_df = process_dataset(zip_path, metadata_path, image_size)
    print("Preprocessing completed!")

    # データの準備完了を確認
    train_path = os.path.join('HAM10000', 'train_dir')
    test_path = os.path.join('HAM10000', 'test_dir')

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise Exception("Training and test directories not found!")

    # モデルの学習
    print("\nStarting model training...")
    model, history = train_model(train_df, test_df)
    print("Training completed!")

    return model, history


if __name__ == "__main__":
    model, history = main()