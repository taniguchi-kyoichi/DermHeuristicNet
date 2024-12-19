import os
import cv2
import numpy as np
from tqdm import tqdm


class DataAugmentor:
    def __init__(self, base_dir='HAM10000'):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'train_dir')
        self.train_seg_dir = os.path.join(base_dir, 'train_segmentation')
        self.target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

        # クラスごとの目標数を設定
        self.target_counts = {
            'mel': 8000,  # メラノーマは重要なので多めに
            'bcc': 8000,  # 基底細胞がんも重要
            'nv': 6000,  # 母斑は元々多いので控えめに
            'bkl': 6000,
            'akiec': 6000,
            'vasc': 6000,
            'df': 6000
        }

    def count_original_images(self, class_name):
        """指定クラスの元画像数をカウント"""
        class_dir = os.path.join(self.train_dir, class_name)
        return len([f for f in os.listdir(class_dir) if not f.startswith('aug_')])

    def apply_augmentation(self, image, mask, params):
        """画像とマスクに同じ変換を適用"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # 回転とスケーリングの変換行列を作成
        M = cv2.getRotationMatrix2D(
            center,
            params['rotation'],
            params['scale']
        )

        # 平行移動を追加
        M[0, 2] += params['tx'] * width
        M[1, 2] += params['ty'] * height

        # アフィン変換を適用
        aug_image = cv2.warpAffine(
            image, M, (width, height),
            borderMode=cv2.BORDER_REFLECT
        )
        aug_mask = cv2.warpAffine(
            mask, M, (width, height),
            borderMode=cv2.BORDER_REFLECT
        )

        # 左右反転
        if params['flip']:
            aug_image = cv2.flip(aug_image, 1)
            aug_mask = cv2.flip(aug_mask, 1)

        # 明るさ調整
        if params['brightness'] != 0:
            aug_image = cv2.add(
                aug_image,
                params['brightness'],
                dtype=cv2.CV_8U
            )

        # コントラスト調整
        if params['contrast'] != 1:
            aug_image = cv2.multiply(
                aug_image,
                params['contrast'],
                dtype=cv2.CV_8U
            )

        return aug_image, aug_mask

    def generate_augmentation_params(self):
        """ランダムな増強パラメータを生成"""
        return {
            'rotation': np.random.uniform(-20, 20),
            'scale': np.random.uniform(0.8, 1.2),
            'tx': np.random.uniform(-0.1, 0.1),
            'ty': np.random.uniform(-0.1, 0.1),
            'flip': np.random.choice([True, False]),
            'brightness': np.random.randint(-30, 30),
            'contrast': np.random.uniform(0.8, 1.2)
        }

    def augment_class(self, class_name):
        """特定のクラスに対してデータ増強を実行"""
        print(f"\nProcessing {class_name} class...")

        # 元画像のパスを取得
        class_dir = os.path.join(self.train_dir, class_name)
        seg_dir = os.path.join(self.train_seg_dir, class_name)

        # 元画像のファイル名リストを取得（aug_で始まるものは除外）
        original_files = [
            f for f in os.listdir(class_dir)
            if not f.startswith('aug_')
        ]

        original_count = len(original_files)
        target_count = self.target_counts[class_name]

        if original_count >= target_count:
            print(f"Skipping {class_name}: Already has enough samples")
            return

        augmentations_needed = target_count - original_count
        augs_per_image = augmentations_needed // original_count + 1

        print(f"Generating {augmentations_needed} augmented images...")

        augmented_count = 0
        for filename in tqdm(original_files):
            # 元画像とセグメンテーションマスクを読み込み
            image_path = os.path.join(class_dir, filename)
            mask_path = os.path.join(seg_dir, filename)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)

            if image is None or mask is None:
                print(f"Warning: Could not read image pair {filename}")
                continue

            # 画像ごとに必要な数の増強を生成
            for i in range(augs_per_image):
                if augmented_count >= augmentations_needed:
                    break

                try:
                    # 増強パラメータを生成して適用
                    params = self.generate_augmentation_params()
                    aug_image, aug_mask = self.apply_augmentation(
                        image, mask, params
                    )

                    # 増強画像を保存
                    base_name = os.path.splitext(filename)[0]
                    aug_filename = f'aug_{i}_{base_name}.jpg'

                    cv2.imwrite(
                        os.path.join(class_dir, aug_filename),
                        aug_image
                    )
                    cv2.imwrite(
                        os.path.join(seg_dir, aug_filename),
                        aug_mask
                    )

                    augmented_count += 1

                except Exception as e:
                    print(f"Error augmenting {filename}: {str(e)}")
                    continue

        print(f"Created {augmented_count} augmented images for {class_name}")
        final_count = len(os.listdir(class_dir))
        print(f"Final count for {class_name}: {final_count}")

    def augment_all(self):
        """全クラスに対してデータ増強を実行"""
        print("Starting data augmentation process...")

        # 各クラスの元の画像数を表示
        for class_name in self.target_names:
            count = self.count_original_images(class_name)
            target = self.target_counts[class_name]
            print(f"{class_name}: {count} original images, target: {target}")

        # 各クラスに対して増強を実行
        for class_name in self.target_names:
            self.augment_class(class_name)

        print("\nData augmentation completed!")


def augment_dataset(base_dir='HAM10000'):
    """データセット全体の増強を実行するメイン関数"""
    augmentor = DataAugmentor(base_dir)
    augmentor.augment_all()


if __name__ == "__main__":
    augment_dataset()