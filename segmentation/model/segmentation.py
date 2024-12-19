import cv2
import numpy as np
from skimage import morphology, measure, feature
import os
from tqdm import tqdm


class SkinLesionSegmentation:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(1, (9, 9))
        self.small_kernel = cv2.getStructuringElement(1, (5, 5))

    def preprocess_image(self, image):
        """画像の前処理"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0]

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        lab[:, :, 0] = l_chan

        return lab

    def create_initial_mask(self, lab_image):
        """初期マスクの生成"""
        # Split channels
        l_chan = lab_image[:, :, 0]
        a_chan = lab_image[:, :, 1]
        b_chan = lab_image[:, :, 2]

        # Create mask using a-channel and b-channel (color information)
        ab_mask = np.zeros_like(l_chan)

        # Looking for significant deviations in a and b channels
        a_mean, a_std = np.mean(a_chan), np.std(a_chan)
        b_mean, b_std = np.mean(b_chan), np.std(b_chan)

        # Mark pixels that deviate significantly from the mean
        color_mask = (
                (np.abs(a_chan - a_mean) > a_std * 1.5) |
                (np.abs(b_chan - b_mean) > b_std * 1.5)
        )

        # Create luminance mask using L-channel
        l_blur = cv2.GaussianBlur(l_chan, (5, 5), 0)
        _, l_mask = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Combine masks
        combined_mask = ((color_mask) | (l_mask > 0)).astype(np.uint8) * 255

        # Close small gaps
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return combined_mask

    def refine_mask(self, mask):
        """マスクの精製"""
        # Remove small objects
        binary = mask > 127
        cleaned = morphology.remove_small_objects(binary, min_size=150)

        # Fill holes
        filled = morphology.remove_small_holes(cleaned, area_threshold=500)

        # Get connected components
        labels = measure.label(filled)
        if labels.max() > 0:
            # Keep significant components
            areas = np.bincount(labels.flat)[1:]
            mask = np.zeros_like(labels, dtype=np.uint8)

            # Keep components that are at least 15% of the largest component
            largest_area = np.max(areas)
            for i, area in enumerate(areas, 1):
                if area >= largest_area * 0.15:
                    mask[labels == i] = 255
        else:
            mask = filled.astype(np.uint8) * 255

        # Final cleaning
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def segment_image(self, image):
        """セグメンテーションのメインパイプライン"""
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Preprocess
            lab_image = self.preprocess_image(image)

            # Create initial mask
            initial_mask = self.create_initial_mask(lab_image)

            # Refine mask
            refined_mask = self.refine_mask(initial_mask)

            # Convert to RGB
            mask_rgb = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2RGB)

            return mask_rgb

        except Exception as e:
            print(f"Error in segmentation: {str(e)}")
            return np.full_like(image, 255, dtype=np.uint8)

    def process_directory(self, input_dir, output_dir, min_lesion_size=1000):
        """ディレクトリ内の画像を処理"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = [f for f in os.listdir(input_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for filename in tqdm(image_files, desc=f"Processing {os.path.basename(input_dir)}"):
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                image = cv2.imread(input_path)
                if image is None:
                    print(f"Failed to read image: {input_path}")
                    continue

                mask = self.segment_image(image)

                # Adaptive minimum lesion size
                adaptive_min_size = min(min_lesion_size,
                                        int(image.shape[0] * image.shape[1] * 0.005))

                if cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)) > adaptive_min_size:
                    cv2.imwrite(output_path, mask)
                else:
                    print(f"Warning: Small lesion detected in {filename}, skipping...")

            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                continue