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
import cv2
from tqdm import tqdm

from model.segmentation import SkinLesionSegmentation


class DatasetPreprocessor:
    def __init__(self, zip_path, metadata_path, image_size=299):
        self.zip_path = zip_path
        self.metadata_path = metadata_path
        self.image_size = image_size
        self.temp_dir = 'temp_ham10000'
        self.output_dir = 'HAM10000'
        self.train_dir = os.path.join(self.output_dir, 'train_dir')
        self.test_dir = os.path.join(self.output_dir, 'test_dir')
        self.train_seg_dir = os.path.join(self.output_dir, 'train_segmentation')
        self.test_seg_dir = os.path.join(self.output_dir, 'test_segmentation')
        self.target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.segmentation = SkinLesionSegmentation()

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
        # Remove existing directories
        for dir_path in [self.output_dir, self.train_seg_dir, self.test_seg_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        # Create main directories
        for dir_path in [self.train_dir, self.test_dir, self.train_seg_dir, self.test_seg_dir]:
            for target in self.target_names:
                os.makedirs(os.path.join(dir_path, target), exist_ok=True)

    def process_metadata(self):
        """Process the metadata CSV file and split into train/test sets"""
        # メタデータの読み込み
        data_pd = pd.read_csv(self.metadata_path)

        # 重複チェック
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

    def resize_and_save_image(self, source_path, target_path, target_seg_path):
        try:
            # Load and resize original image
            with Image.open(source_path) as img:
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                img.save(target_path, quality=95)

            # Generate and save segmentation mask
            image = cv2.imread(target_path)
            if image is not None:
                segmented = self.segmentation.segment_image(image)
                cv2.imwrite(target_seg_path, segmented)
            else:
                print(f"Failed to read image for segmentation: {target_path}")

        except Exception as e:
            print(f"Error processing image {source_path}: {str(e)}")
            raise

    def process_images(self, data_pd, train_df, test_df, image_files):
        # Process training images
        for idx, row in train_df.iterrows():
            if row['image_id'] in image_files:
                source = image_files[row['image_id']]
                target = os.path.join(self.train_dir, row['dx'], f"{row['image_id']}.jpg")
                target_seg = os.path.join(self.train_seg_dir, row['dx'], f"{row['image_id']}.jpg")
                self.resize_and_save_image(source, target, target_seg)

        # Process test images
        for idx, row in test_df.iterrows():
            if row['image_id'] in image_files:
                source = image_files[row['image_id']]
                target = os.path.join(self.test_dir, row['dx'], f"{row['image_id']}.jpg")
                target_seg = os.path.join(self.test_seg_dir, row['dx'], f"{row['image_id']}.jpg")
                self.resize_and_save_image(source, target, target_seg)

    def augment_training_data(self):
        """改善されたデータ増強処理"""
        print("Starting data augmentation...")

        for img_class in self.target_names:
            print(f"Processing {img_class} class images...")

            # Get original images
            class_dir = os.path.join(self.train_dir, img_class)
            img_list = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_original = len(img_list)

            # Calculate augmentation factor based on class
            if img_class in ['mel', 'bcc']:  # 重要なクラスは多めに増強
                target_total = 8000
            else:
                target_total = 6000

            num_to_generate = max(0, target_total - num_original)

            if num_to_generate > 0:
                print(f"Generating {num_to_generate} augmented pairs for {img_class}")

                # Process each original image
                for filename in tqdm(img_list):
                    # Load original image and its segmentation
                    orig_path = os.path.join(class_dir, filename)
                    seg_path = os.path.join(self.train_seg_dir, img_class, filename)

                    orig_img = cv2.imread(orig_path)
                    seg_img = cv2.imread(seg_path)

                    if orig_img is None or seg_img is None:
                        print(f"Warning: Could not read image pair {filename}")
                        continue

                    # Calculate number of augmentations per image
                    augs_per_image = max(1, num_to_generate // num_original)

                    for i in range(augs_per_image):
                        try:
                            # 回転角度などのパラメータを生成
                            angle = np.random.uniform(-10, 10)
                            tx = np.random.uniform(-0.1, 0.1)
                            ty = np.random.uniform(-0.1, 0.1)
                            scale = np.random.uniform(0.9, 1.1)
                            flip = np.random.choice([True, False])

                            # 変換行列を計算
                            center = (orig_img.shape[1] // 2, orig_img.shape[0] // 2)
                            M = cv2.getRotationMatrix2D(center, angle, scale)
                            M[0, 2] += tx * orig_img.shape[1]
                            M[1, 2] += ty * orig_img.shape[0]

                            # 同じ変換を両方の画像に適用
                            aug_orig = cv2.warpAffine(orig_img, M, (orig_img.shape[1], orig_img.shape[0]),
                                                      borderMode=cv2.BORDER_REFLECT)
                            aug_seg = cv2.warpAffine(seg_img, M, (seg_img.shape[1], seg_img.shape[0]),
                                                     borderMode=cv2.BORDER_REFLECT)

                            if flip:
                                aug_orig = cv2.flip(aug_orig, 1)
                                aug_seg = cv2.flip(aug_seg, 1)

                            # Save augmented image pair
                            aug_filename = f'aug_{i}_{filename}'
                            aug_orig_path = os.path.join(class_dir, aug_filename)
                            aug_seg_path = os.path.join(self.train_seg_dir, img_class, aug_filename)

                            cv2.imwrite(aug_orig_path, aug_orig)
                            cv2.imwrite(aug_seg_path, aug_seg)

                        except Exception as e:
                            print(f"Error augmenting {filename}: {str(e)}")
                            continue

                print(f"Completed augmentation for {img_class}")

        print("Data augmentation completed!")

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def process(self):
        """全体の前処理パイプラインを実行"""
        try:
            print("Starting preprocessing pipeline...")

            # データセットが既に存在するか確認
            if self._check_existing_dataset():
                print("Found existing processed dataset. Loading metadata...")
                data_pd = pd.read_csv(self.metadata_path)

                # 重複チェック
                df_count = data_pd.groupby('lesion_id').count()
                df_count = df_count[df_count['dx'] == 1]
                df_count.reset_index(inplace=True)

                # Identify unique lesions
                unique_lesions = set(df_count['lesion_id'])
                data_pd['is_duplicate'] = data_pd['lesion_id'].apply(
                    lambda x: 'no' if x in unique_lesions else 'duplicates'
                )

                # Split into train and test sets if not already split
                if 'train_test_split' not in data_pd.columns:
                    print("Splitting data into train and test sets...")
                    train_df, test_df = train_test_split(
                        data_pd,
                        test_size=0.15,
                        stratify=data_pd['dx'],
                        random_state=42
                    )

                    # Add train_test_split column
                    test_ids = set(test_df['image_id'])
                    data_pd['train_test_split'] = data_pd['image_id'].apply(
                        lambda x: 'test' if x in test_ids else 'train'
                    )

                    # Save updated metadata
                    data_pd.to_csv(self.metadata_path, index=False)
                else:
                    print("Using existing train/test split...")
                    train_df = data_pd[data_pd['train_test_split'] == 'train']
                    test_df = data_pd[data_pd['train_test_split'] == 'test']

                return data_pd, train_df, test_df

            # データセットが存在しない場合は新規作成
            print("Creating new dataset...")
            image_files = self.extract_dataset()
            self.create_directory_structure()
            data_pd, train_df, test_df = self.process_metadata()

            print("Processing and segmenting images...")
            self.process_images(data_pd, train_df, test_df, image_files)

            print("Starting data augmentation...")
            self.augment_training_data()

            self.cleanup()
            print("Preprocessing completed successfully!")
            return data_pd, train_df, test_df

        except Exception as e:
            print(f"An error occurred during preprocessing: {str(e)}")
            self.cleanup()
            raise

    def _check_existing_dataset(self):
        """既存のデータセットが存在するか確認"""
        required_dirs = [
            self.train_dir,
            self.test_dir,
            self.train_seg_dir,
            self.test_seg_dir
        ]

        # すべての必要なディレクトリが存在するか確認
        dirs_exist = all(os.path.exists(d) for d in required_dirs)

        # 各クラスディレクトリ内にファイルが存在するか確認
        if dirs_exist:
            for dir_path in required_dirs:
                for target in self.target_names:
                    class_dir = os.path.join(dir_path, target)
                    if not os.path.exists(class_dir) or not os.listdir(class_dir):
                        return False
            return True

        return False

    def cleanup(self):
        """一時ファイルとディレクトリの削除"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def process_dataset(zip_path, metadata_path, image_size=299):
    preprocessor = DatasetPreprocessor(zip_path, metadata_path, image_size)
    return preprocessor.process()


if __name__ == "__main__":
    zip_path = "datasets/HAM10000.zip"
    metadata_path = "datasets/HAM10000_metadata.csv"
    data_pd, train_df, test_df = process_dataset(zip_path, metadata_path)