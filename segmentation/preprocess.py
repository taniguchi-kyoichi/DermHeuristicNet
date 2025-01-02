import os
import shutil
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import ImageFile, Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob
import cv2
from tqdm import tqdm

from model.segmentation import SkinLesionSegmentation
from model.color_diversity import ColorDiversitySegmentation
from config import Config


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
        self.train_color_dir = os.path.join(self.output_dir, 'train_color_diversity')
        self.test_color_dir = os.path.join(self.output_dir, 'test_color_diversity')
        self.target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.segmentation = SkinLesionSegmentation()
        self.color_diversity = ColorDiversitySegmentation(n_clusters=Config.COLOR_CLUSTERS)

    def find_image_files(self):
        """Find all image files in the extracted directory"""
        image_files = glob.glob(os.path.join(self.temp_dir, '**', '*.jpg'), recursive=True)
        return {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}

    def extract_dataset(self):
        """Extract the dataset from zip file"""
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
        """Create necessary directories for the dataset"""
        # Remove existing directories
        for dir_path in [self.output_dir, self.train_seg_dir, self.test_seg_dir,
                         self.train_color_dir, self.test_color_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        # Create main directories
        for dir_path in [self.train_dir, self.test_dir, self.train_seg_dir,
                         self.test_seg_dir, self.train_color_dir, self.test_color_dir]:
            for target in self.target_names:
                os.makedirs(os.path.join(dir_path, target), exist_ok=True)

    def process_metadata(self):
        """Process the metadata CSV file and split into train/test sets"""
        # Load metadata
        data_pd = pd.read_csv(self.metadata_path)

        # Check for duplicates
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

    def resize_and_save_image(self, source_path, target_path, target_seg_path, target_color_path):
        """Process and save a single image with its segmentation and color diversity maps"""
        try:
            # Load and resize original image
            with Image.open(source_path) as img:
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                img.save(target_path, quality=95)

            # Read image for processing
            image = cv2.imread(target_path)
            if image is None:
                raise ValueError(f"Failed to read image: {target_path}")

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate and save segmentation mask
            segmented = self.segmentation.segment_image(image_rgb)
            cv2.imwrite(target_seg_path, segmented)

            # Generate and save color diversity map
            diversity_map = self.color_diversity.generate_diversity_map(image_rgb, segmented)
            cv2.imwrite(target_color_path, diversity_map)

        except Exception as e:
            print(f"Error processing image {source_path}: {str(e)}")
            raise

    def process_images(self, data_pd, train_df, test_df, image_files):
        """Process all images in the dataset"""
        # Process training images
        print("Processing training images...")
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
            if row['image_id'] in image_files:
                source = image_files[row['image_id']]
                target = os.path.join(self.train_dir, row['dx'], f"{row['image_id']}.jpg")
                target_seg = os.path.join(self.train_seg_dir, row['dx'], f"{row['image_id']}.jpg")
                target_color = os.path.join(self.train_color_dir, row['dx'], f"{row['image_id']}.jpg")
                self.resize_and_save_image(source, target, target_seg, target_color)

        # Process test images
        print("Processing test images...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            if row['image_id'] in image_files:
                source = image_files[row['image_id']]
                target = os.path.join(self.test_dir, row['dx'], f"{row['image_id']}.jpg")
                target_seg = os.path.join(self.test_seg_dir, row['dx'], f"{row['image_id']}.jpg")
                target_color = os.path.join(self.test_color_dir, row['dx'], f"{row['image_id']}.jpg")
                self.resize_and_save_image(source, target, target_seg, target_color)

    def augment_training_data(self):
        """Improved data augmentation process"""
        print("Starting data augmentation...")

        for img_class in self.target_names:
            print(f"Processing {img_class} class images...")

            # Get original images
            class_dir = os.path.join(self.train_dir, img_class)
            img_list = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            num_original = len(img_list)

            # Calculate augmentation factor based on class
            if img_class in ['mel', 'bcc']:  # More augmentation for important classes
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
                    color_path = os.path.join(self.train_color_dir, img_class, filename)

                    orig_img = cv2.imread(orig_path)
                    seg_img = cv2.imread(seg_path)
                    color_img = cv2.imread(color_path)

                    if orig_img is None or seg_img is None or color_img is None:
                        print(f"Warning: Could not read image set {filename}")
                        continue

                    # Calculate number of augmentations per image
                    augs_per_image = max(1, num_to_generate // num_original)

                    for i in range(augs_per_image):
                        try:
                            # Generate transformation parameters
                            angle = np.random.uniform(-10, 10)
                            tx = np.random.uniform(-0.1, 0.1)
                            ty = np.random.uniform(-0.1, 0.1)
                            scale = np.random.uniform(0.9, 1.1)
                            flip = np.random.choice([True, False])

                            # Calculate transformation matrix
                            center = (orig_img.shape[1] // 2, orig_img.shape[0] // 2)
                            M = cv2.getRotationMatrix2D(center, angle, scale)
                            M[0, 2] += tx * orig_img.shape[1]
                            M[1, 2] += ty * orig_img.shape[0]

                            # Apply same transformation to all images
                            aug_orig = cv2.warpAffine(orig_img, M, (orig_img.shape[1], orig_img.shape[0]),
                                                      borderMode=cv2.BORDER_REFLECT)
                            aug_seg = cv2.warpAffine(seg_img, M, (seg_img.shape[1], seg_img.shape[0]),
                                                     borderMode=cv2.BORDER_REFLECT)
                            aug_color = cv2.warpAffine(color_img, M, (color_img.shape[1], color_img.shape[0]),
                                                       borderMode=cv2.BORDER_REFLECT)

                            if flip:
                                aug_orig = cv2.flip(aug_orig, 1)
                                aug_seg = cv2.flip(aug_seg, 1)
                                aug_color = cv2.flip(aug_color, 1)

                            # Save augmented image set
                            aug_filename = f'aug_{i}_{filename}'
                            aug_orig_path = os.path.join(class_dir, aug_filename)
                            aug_seg_path = os.path.join(self.train_seg_dir, img_class, aug_filename)
                            aug_color_path = os.path.join(self.train_color_dir, img_class, aug_filename)

                            cv2.imwrite(aug_orig_path, aug_orig)
                            cv2.imwrite(aug_seg_path, aug_seg)
                            cv2.imwrite(aug_color_path, aug_color)

                        except Exception as e:
                            print(f"Error augmenting {filename}: {str(e)}")
                            continue

                print(f"Completed augmentation for {img_class}")

        print("Data augmentation completed!")

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def process(self):
        """Execute the complete preprocessing pipeline"""
        try:
            print("Starting preprocessing pipeline...")

            # Check for existing dataset
            if self._check_existing_dataset():
                print("Found existing processed dataset. Loading metadata...")
                data_pd = pd.read_csv(self.metadata_path)
                train_df = data_pd[data_pd['train_test_split'] == 'train']
                test_df = data_pd[data_pd['train_test_split'] == 'test']
                return data_pd, train_df, test_df

            # Create new dataset
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
        """Check if processed dataset already exists"""
        required_dirs = [
            self.train_dir,
            self.test_dir,
            self.train_seg_dir,
            self.test_seg_dir,
            self.train_color_dir,
            self.test_color_dir
        ]

        # Check if all required directories exist
        dirs_exist = all(os.path.exists(d) for d in required_dirs)

        # Check if all class directories contain files
        if dirs_exist:
            for dir_path in required_dirs:
                for target in self.target_names:
                    class_dir = os.path.join(dir_path, target)
                    if not os.path.exists(class_dir) or not os.listdir(class_dir):
                        return False
            return True

        return False


def process_dataset(zip_path, metadata_path, image_size=299):
    """Main function to process the dataset"""
    preprocessor = DatasetPreprocessor(zip_path, metadata_path, image_size)
    return preprocessor.process()


if __name__ == "__main__":
    zip_path = "datasets/HAM10000.zip"
    metadata_path = "datasets/HAM10000_metadata.csv"
    data_pd, train_df, test_df = process_dataset(zip_path, metadata_path)