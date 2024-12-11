# preprocess.py
import os
import shutil
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import ImageFile, Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob


class DatasetPreprocessor:
    def __init__(self, zip_path, metadata_path, image_size=299):
        self.zip_path = zip_path
        self.metadata_path = metadata_path
        self.image_size = image_size
        self.temp_dir = 'temp_ham10000'
        self.output_dir = 'HAM10000'
        self.train_dir = os.path.join(self.output_dir, 'train_dir')
        self.test_dir = os.path.join(self.output_dir, 'test_dir')
        self.target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def find_image_files(self):
        image_files = glob.glob(os.path.join(self.temp_dir, '**', '*.jpg'), recursive=True)
        return {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}

    def extract_dataset(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

        with ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        print("Dataset extraction completed")

        image_files = self.find_image_files()
        if not image_files:
            raise Exception("No image files found in the extracted directory")
        print(f"Found {len(image_files)} images")

        return image_files

    def create_directory_structure(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.makedirs(self.train_dir)
        os.makedirs(self.test_dir)

        for target in self.target_names:
            os.makedirs(os.path.join(self.train_dir, target))
            os.makedirs(os.path.join(self.test_dir, target))

    def process_metadata(self):
        """Process the metadata CSV file and split into train/test sets"""
        # メタデータの読み込み
        data_pd = pd.read_csv(self.metadata_path)

        # 既存実装と同じ重複チェック処理
        df_count = data_pd.groupby('lesion_id').count()
        df_count = df_count[df_count['dx'] == 1]
        df_count.reset_index(inplace=True)

        # Identify unique lesions
        unique_lesions = set(df_count['lesion_id'])
        data_pd['is_duplicate'] = data_pd['lesion_id'].apply(
            lambda x: 'no' if x in unique_lesions else 'duplicates'
        )

        # Split into train and test sets
        train_df, test_df = train_test_split(
            data_pd,
            test_size=0.15,
            stratify=data_pd['dx'],
            random_state=42
        )

        # Mark images as train or test
        test_ids = set(test_df['image_id'])
        data_pd['train_test_split'] = data_pd['image_id'].apply(
            lambda x: 'test' if x in test_ids else 'train'
        )

        return data_pd, train_df, test_df

    def resize_and_save_image(self, source_path, target_path):
        try:
            with Image.open(source_path) as img:
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                img.save(target_path, quality=95)
        except Exception as e:
            print(f"Error processing image {source_path}: {str(e)}")
            raise

    def process_images(self, data_pd, train_df, test_df, image_files):
        # Process training images
        for idx, row in train_df.iterrows():
            if row['image_id'] in image_files:
                source = image_files[row['image_id']]
                target = os.path.join(self.train_dir, row['dx'], f"{row['image_id']}.jpg")
                self.resize_and_save_image(source, target)

        # Process test images
        for idx, row in test_df.iterrows():
            if row['image_id'] in image_files:
                source = image_files[row['image_id']]
                target = os.path.join(self.test_dir, row['dx'], f"{row['image_id']}.jpg")
                self.resize_and_save_image(source, target)

    def augment_training_data(self):
        """トレーニングデータの増強を行う"""
        print("Starting data augmentation...")

        datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        for img_class in self.target_names:
            print(f"Augmenting {img_class} class images...")

            # Create temporary directory for augmentation
            aug_dir = 'aug_dir'
            img_dir = os.path.join(aug_dir, 'img_dir')
            os.makedirs(img_dir, exist_ok=True)

            # Get original images
            class_dir = os.path.join(self.train_dir, img_class)
            img_list = os.listdir(class_dir)
            num_original = len(img_list)

            # Copy original images to temporary directory
            for file_name in img_list:
                source = os.path.join(class_dir, file_name)
                target = os.path.join(img_dir, file_name)
                shutil.copyfile(source, target)

            # Set up augmentation generator
            aug_datagen = datagen.flow_from_directory(
                aug_dir,
                target_size=(self.image_size, self.image_size),
                save_to_dir=class_dir,
                save_format='jpg',
                batch_size=1
            )

            # Generate augmented images
            target_total = 8000  # 目標とする総画像数
            num_to_generate = max(0, target_total - num_original)

            print(f"Generating {num_to_generate} new images for {img_class}")
            for _ in range(num_to_generate):
                aug_datagen.next()

            # Clean up
            shutil.rmtree(aug_dir)

            print(f"Completed augmentation for {img_class}. Total images: {len(os.listdir(class_dir))}")

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def process(self):
        try:
            print("Starting preprocessing pipeline...")

            image_files = self.extract_dataset()
            self.create_directory_structure()
            data_pd, train_df, test_df = self.process_metadata()
            self.process_images(data_pd, train_df, test_df, image_files)
            self.augment_training_data()
            self.cleanup()

            print("Preprocessing completed successfully!")
            return data_pd, train_df, test_df

        except Exception as e:
            print(f"An error occurred during preprocessing: {str(e)}")
            self.cleanup()
            raise


def process_dataset(zip_path, metadata_path, image_size=299):
    preprocessor = DatasetPreprocessor(zip_path, metadata_path, image_size)
    return preprocessor.process()


if __name__ == "__main__":
    zip_path = "datasets/HAM10000.zip"
    metadata_path = "datasets/HAM10000_metadata.csv"
    data_pd, train_df, test_df = process_dataset(zip_path, metadata_path)