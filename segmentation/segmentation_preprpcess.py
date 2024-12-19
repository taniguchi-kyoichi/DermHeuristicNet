import os
import shutil
import pandas as pd
from zipfile import ZipFile
from PIL import ImageFile, Image
from sklearn.model_selection import train_test_split
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
        """Extract image files from the temporary directory"""
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
        """Create the required directory structure"""
        # Remove existing directories
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        # Create main directories
        for dir_path in [self.train_dir, self.test_dir]:
            for target in self.target_names:
                os.makedirs(os.path.join(dir_path, target), exist_ok=True)

    def process_metadata(self):
        """Process the metadata CSV file and split into train/test sets"""
        data_pd = pd.read_csv(self.metadata_path)

        # Train-test split
        train_df, test_df = train_test_split(
            data_pd,
            test_size=0.15,
            stratify=data_pd['dx'],
            random_state=42
        )

        return train_df, test_df

    def resize_and_save_image(self, source_path, target_path):
        """Resize and save the image"""
        try:
            with Image.open(source_path) as img:
                img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
                img.save(target_path, quality=95)
        except Exception as e:
            print(f"Error processing image {source_path}: {str(e)}")
            raise

    def process_images(self, train_df, test_df, image_files):
        """Process and resize images for both train and test sets"""
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

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def process(self):
        """Execute the entire preprocessing pipeline"""
        try:
            print("Starting preprocessing pipeline...")
            image_files = self.extract_dataset()
            self.create_directory_structure()
            train_df, test_df = self.process_metadata()

            print("Processing and resizing images...")
            self.process_images(train_df, test_df, image_files)

            self.cleanup()
            print("Preprocessing completed successfully!")
            return train_df, test_df

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
    train_df, test_df = process_dataset(zip_path, metadata_path)