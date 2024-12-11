import cv2
import numpy as np
from skimage import filters, morphology, measure
import os
from tqdm import tqdm


class SkinLesionSegmentation:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(1, (9, 9))
        self.small_kernel = cv2.getStructuringElement(1, (5, 5))

    def preprocess_image(self, image):
        """画像の前処理"""
        img = image.astype(np.float32)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img

    def remove_hair(self, image):
        """毛髪除去の改善版"""
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply black hat filter
        blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, self.small_kernel)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        thresh = cv2.dilate(thresh, self.small_kernel, iterations=1)
        thresh = cv2.medianBlur(thresh, 3)

        return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

    def create_initial_mask(self, image):
        """ABCDルールを考慮した初期マスクの生成"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Enhance contrast for better border detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Get edges using Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Create initial mask using Otsu's method
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine edge information
        mask = cv2.bitwise_or(mask, edges)

        return mask

    def refine_mask(self, mask):
        """マスクの精製とABCDルールの強調"""
        # Remove small objects
        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=100)

        # Fill holes
        filled = morphology.remove_small_holes(cleaned, area_threshold=100)

        # Keep largest connected component
        labels = measure.label(filled)
        if labels.max() > 0:
            largest_component = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1))
            filled = largest_component

        # Convert to uint8
        mask = filled.astype(np.uint8) * 255

        # Enhance borders
        kernel = np.ones((3, 3), np.uint8)
        gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        mask = cv2.addWeighted(mask, 1, gradient, 0.5, 0)

        return mask

    def segment_image(self, image):
        """改善されたセグメンテーションパイプライン"""
        try:
            # Remove hair
            no_hair_image = self.remove_hair(image)

            # Create initial mask
            initial_mask = self.create_initial_mask(no_hair_image)

            # Refine the mask
            refined_mask = self.refine_mask(initial_mask)

            # Convert to RGB
            mask_rgb = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2RGB)

            return mask_rgb

        except Exception as e:
            print(f"Error in segmentation: {str(e)}")
            return np.full_like(image, 255)

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

                if cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)) > min_lesion_size:
                    cv2.imwrite(output_path, mask)
                else:
                    print(f"Warning: Small lesion detected in {filename}, skipping...")

            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                continue