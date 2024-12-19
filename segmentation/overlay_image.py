import os
import cv2
import numpy as np
from tqdm import tqdm


def create_overlay_image(original_path, segmentation_path, alpha=0.4):
    """
    原画像とセグメンテーション結果を重ね合わせる

    Parameters:
    -----------
    original_path : str
        元画像のパス
    segmentation_path : str
        セグメンテーション結果画像のパス
    alpha : float
        セグメンテーション結果の透明度 (0.0 - 1.0)

    Returns:
    --------
    numpy.ndarray
        重ね合わせ画像
    """
    # 画像の読み込み
    original = cv2.imread(original_path)
    if original is None:
        raise ValueError(f"Could not read original image: {original_path}")

    segmentation = cv2.imread(segmentation_path)
    if segmentation is None:
        raise ValueError(f"Could not read segmentation image: {segmentation_path}")

    # サイズの一致を確認し、必要に応じてリサイズ
    if original.shape != segmentation.shape:
        segmentation = cv2.resize(segmentation, (original.shape[1], original.shape[0]))

    # セグメンテーション結果を緑色のマスクに変換
    mask = np.zeros_like(original)
    mask_indices = segmentation > 0
    mask[..., 0][mask_indices[..., 0]] = 0  # Blue channel
    mask[..., 1][mask_indices[..., 0]] = 255  # Green channel
    mask[..., 2][mask_indices[..., 0]] = 0  # Red channel

    # 重ね合わせ
    overlay = cv2.addWeighted(original, 1, mask, alpha, 0)

    return overlay


def process_dataset(base_dir, segmentation_dir, output_dir):
    """
    データセット全体の処理を行う

    Parameters:
    -----------
    base_dir : str
        元画像のベースディレクトリ
    segmentation_dir : str
        セグメンテーション結果のディレクトリ
    output_dir : str
        出力先ディレクトリ
    """
    # ディレクトリ内のすべてのカテゴリを処理
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 出力用のカテゴリディレクトリを作成
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)

        # カテゴリ内の各画像を処理
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Processing category: {category}")
        for image_file in tqdm(image_files):
            try:
                original_path = os.path.join(base_dir, category, image_file)
                segmentation_path = os.path.join(segmentation_dir, category, image_file)
                output_path = os.path.join(output_category_dir, f"overlay_{image_file}")

                # セグメンテーション結果が存在する場合のみ処理
                if os.path.exists(segmentation_path):
                    overlay = create_overlay_image(original_path, segmentation_path)
                    cv2.imwrite(output_path, overlay)
                else:
                    print(f"Segmentation not found for: {image_file}")

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")


def main():
    # パスの設定
    base_dir = "HAM10000/test_dir"  # 元画像のディレクトリ
    segmentation_dir = "HAM10000/test_segmentation"  # セグメンテーション結果のディレクトリ
    output_dir = "HAM10000/test_overlay"  # 重ね合わせ画像の出力先

    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # データセットの処理
    process_dataset(base_dir, segmentation_dir, output_dir)

    print("Processing completed!")


if __name__ == "__main__":
    main()