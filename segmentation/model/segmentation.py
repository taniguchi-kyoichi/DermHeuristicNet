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
        """改善版の初期マスク生成 - より安定した閾値決定"""
        l_chan = lab_image[:, :, 0]
        a_chan = lab_image[:, :, 1]
        b_chan = lab_image[:, :, 2]

        # グローバルな統計量の計算
        l_mean, l_std = np.mean(l_chan), np.std(l_chan)
        a_mean, a_std = np.mean(a_chan), np.std(a_chan)
        b_mean, b_std = np.mean(b_chan), np.std(b_chan)

        # 大津の方法による基本閾値の取得
        _, l_otsu = cv2.threshold(l_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_threshold = np.mean(l_chan[l_otsu > 0])

        # 色空間での異常値検出
        color_deviation_a = np.abs(a_chan - a_mean)
        color_deviation_b = np.abs(b_chan - b_mean)

        # 適応的な閾値の計算
        k = 1.5  # 感度係数
        color_threshold_a = k * a_std
        color_threshold_b = k * b_std

        # マスクの生成
        color_mask = ((color_deviation_a > color_threshold_a) |
                      (color_deviation_b > color_threshold_b))

        # 輝度マスクの生成
        l_deviation = np.abs(l_chan - l_mean)
        l_threshold = k * l_std
        l_mask = l_deviation > l_threshold

        # マスクの組み合わせ
        combined_mask = ((color_mask) | (l_mask)).astype(np.uint8) * 255

        # ノイズ除去とマスクの改善
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # 連結成分分析によるノイズ除去
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask)

        # 最小サイズのフィルタリング
        min_size = 50  # 最小サイズ閾値
        clean_mask = np.zeros_like(combined_mask)

        for i in range(1, num_labels):  # 0はバックグラウンド
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                clean_mask[labels == i] = 255

        return clean_mask

    def refine_mask(self, mask):
        """マスクの精製処理を改善"""
        # Remove small objects
        binary = mask > 127
        cleaned = morphology.remove_small_objects(binary, min_size=50)

        # Fill holes
        filled = morphology.remove_small_holes(cleaned, area_threshold=100)

        # Connected component analysis
        labels = measure.label(filled)
        if labels.max() > 0:
            # Keep only significant components
            areas = np.bincount(labels.flat)[1:]
            mask = np.zeros_like(labels, dtype=np.uint8)

            # Keep components that are at least 10% of the largest component
            largest_area = np.max(areas)
            for i, area in enumerate(areas, 1):
                if area >= largest_area * 0.1:
                    mask[labels == i] = 255
        else:
            mask = filled.astype(np.uint8) * 255

        # Final morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def segment_image(self, image):
        """セグメンテーションのメインパイプライン"""
        try:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Preprocess
            lab_image = self.preprocess_image(image)

            # Create initial mask with adaptive thresholding
            initial_mask = self.create_initial_mask(lab_image)

            # Refine mask with improved morphological operations
            refined_mask = self.refine_mask(initial_mask)

            # Convert to RGB
            mask_rgb = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2RGB)

            return mask_rgb

        except Exception as e:
            print(f"Error in segmentation: {str(e)}")
            return np.full_like(image, 255, dtype=np.uint8)

    def process_directory(self, input_dir, output_dir, min_lesion_size=1000):
        """ディレクトリ処理"""
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

                adaptive_min_size = min(min_lesion_size,
                                        int(image.shape[0] * image.shape[1] * 0.005))

                if cv2.countNonZero(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)) > adaptive_min_size:
                    cv2.imwrite(output_path, mask)
                else:
                    print(f"Warning: Small lesion detected in {filename}, skipping...")

            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                continue