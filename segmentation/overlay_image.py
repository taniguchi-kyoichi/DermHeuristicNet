import os
import cv2
import numpy as np
from tqdm import tqdm


def create_triple_overlay_image(original_path, segmentation_path, color_path, alpha_seg=0.4, alpha_color=0.3):
    """
    原画像、セグメンテーション結果、色多様性マップを重ね合わせる

    Parameters:
    -----------
    original_path : str
        元画像のパス
    segmentation_path : str
        セグメンテーション結果画像のパス
    color_path : str
        色多様性マップのパス
    alpha_seg : float
        セグメンテーション結果の透明度 (0.0 - 1.0)
    alpha_color : float
        色多様性マップの透明度 (0.0 - 1.0)

    Returns:
    --------
    numpy.ndarray
        重ね合わせ画像
    numpy.ndarray
        並べて表示用の画像
    """
    # 画像の読み込み
    original = cv2.imread(original_path)
    if original is None:
        raise ValueError(f"Could not read original image: {original_path}")

    segmentation = cv2.imread(segmentation_path)
    if segmentation is None:
        raise ValueError(f"Could not read segmentation image: {segmentation_path}")

    color_map = cv2.imread(color_path)
    if color_map is None:
        raise ValueError(f"Could not read color diversity map: {color_path}")

    # サイズの一致を確認し、必要に応じてリサイズ
    target_size = original.shape[:2]
    if segmentation.shape[:2] != target_size:
        segmentation = cv2.resize(segmentation, (target_size[1], target_size[0]))
    if color_map.shape[:2] != target_size:
        color_map = cv2.resize(color_map, (target_size[1], target_size[0]))

    # セグメンテーションマスクを緑色に変換
    seg_mask = np.zeros_like(original)
    seg_indices = segmentation > 0
    seg_mask[..., 1][seg_indices[..., 0]] = 255  # Green channel

    # 色多様性マップをヒートマップに変換
    color_heatmap = cv2.applyColorMap(
        cv2.cvtColor(color_map, cv2.COLOR_BGR2GRAY),
        cv2.COLORMAP_JET
    )

    # 重ね合わせ画像の作成
    overlay = original.copy()
    # セグメンテーションの重ね合わせ
    overlay = cv2.addWeighted(overlay, 1, seg_mask, alpha_seg, 0)
    # 色多様性マップの重ね合わせ
    overlay = cv2.addWeighted(overlay, 1, color_heatmap, alpha_color, 0)

    # 4枚並べて表示用の画像を作成
    # 1枚のサイズを計算
    height, width = original.shape[:2]
    # 2x2のグリッドを作成
    grid = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)

    # 各画像を配置
    grid[:height, :width] = original  # 左上: オリジナル
    grid[:height, width:] = segmentation  # 右上: セグメンテーション
    grid[height:, :width] = color_heatmap  # 左下: 色多様性マップ
    grid[height:, width:] = overlay  # 右下: 重ね合わせ

    # ラベルを追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)

    # 各画像にラベルを追加
    labels = ['Original', 'Segmentation', 'Color Diversity', 'Overlay']
    positions = [(10, 30), (width + 10, 30), (10, height + 30), (width + 10, height + 30)]

    for label, pos in zip(labels, positions):
        cv2.putText(grid, label, pos, font, font_scale, font_color, font_thickness)

    return overlay, grid


def process_dataset(base_dir, segmentation_dir, color_dir, output_dir):
    """
    データセット全体の処理を行う

    Parameters:
    -----------
    base_dir : str
        元画像のベースディレクトリ
    segmentation_dir : str
        セグメンテーション結果のディレクトリ
    color_dir : str
        色多様性マップのディレクトリ
    output_dir : str
        出力先ディレクトリ
    """
    # 出力ディレクトリの構造を作成
    overlay_dir = os.path.join(output_dir, 'overlay')
    grid_dir = os.path.join(output_dir, 'grid')

    for dir_path in [overlay_dir, grid_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # ディレクトリ内のすべてのカテゴリを処理
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue

        # 出力用のカテゴリディレクトリを作成
        overlay_category_dir = os.path.join(overlay_dir, category)
        grid_category_dir = os.path.join(grid_dir, category)
        os.makedirs(overlay_category_dir, exist_ok=True)
        os.makedirs(grid_category_dir, exist_ok=True)

        # カテゴリ内の各画像を処理
        image_files = [f for f in os.listdir(category_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Processing category: {category}")
        for image_file in tqdm(image_files):
            try:
                # 入力パスの設定
                original_path = os.path.join(base_dir, category, image_file)
                segmentation_path = os.path.join(segmentation_dir, category, image_file)
                color_path = os.path.join(color_dir, category, image_file)

                # 出力パスの設定
                overlay_path = os.path.join(overlay_category_dir, f"overlay_{image_file}")
                grid_path = os.path.join(grid_category_dir, f"grid_{image_file}")

                # 全ての必要なファイルが存在する場合のみ処理
                if all(os.path.exists(p) for p in [original_path, segmentation_path, color_path]):
                    # オーバーレイと並べて表示用の画像を作成
                    overlay, grid = create_triple_overlay_image(
                        original_path,
                        segmentation_path,
                        color_path
                    )

                    # 画像の保存
                    cv2.imwrite(overlay_path, overlay)
                    cv2.imwrite(grid_path, grid)
                else:
                    missing_files = []
                    for path, name in [(original_path, 'Original'),
                                       (segmentation_path, 'Segmentation'),
                                       (color_path, 'Color map')]:
                        if not os.path.exists(path):
                            missing_files.append(name)
                    print(f"Missing files for {image_file}: {', '.join(missing_files)}")

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue


def main():
    """メイン実行関数"""
    # パスの設定
    base_dir = "HAM10000/test_dir"  # 元画像のディレクトリ
    segmentation_dir = "HAM10000/test_segmentation"  # セグメンテーション結果のディレクトリ
    color_dir = "HAM10000/test_color_diversity"  # 色多様性マップのディレクトリ
    output_dir = "HAM10000/test_visualization"  # 出力先

    # データセットの処理
    try:
        process_dataset(base_dir, segmentation_dir, color_dir, output_dir)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()