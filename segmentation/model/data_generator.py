import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DualStreamGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, seg_dir, image_size=299, batch_size=16, is_training=True):
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.is_training = is_training

        # 基本的な前処理のジェネレータ
        self.base_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
        )

        # 元画像のジェネレータ
        self.orig_gen = self.base_datagen.flow_from_directory(
            directory=self.data_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=self.is_training,
            class_mode='categorical'
        )

        # セグメンテーション画像のジェネレータ
        self.seg_gen = self.base_datagen.flow_from_directory(
            directory=self.seg_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=self.is_training,
            class_mode='categorical'
        )

        # データ増強用の設定
        if self.is_training:
            self.augmentor = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='reflect'
            )

        self.indices = np.arange(len(self.orig_gen))
        if self.is_training:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.orig_gen)

    def _apply_augmentation(self, orig_img, seg_img):
        """Apply the same augmentation to both images"""
        transform_params = self.augmentor.get_random_transform(orig_img.shape)

        aug_orig = self.augmentor.apply_transform(orig_img, transform_params)
        aug_seg = self.augmentor.apply_transform(seg_img, transform_params)

        return aug_orig, aug_seg

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Get original images and labels
        orig_batch = self.orig_gen[idx]
        X_orig, y = orig_batch[0], orig_batch[1]

        # Get corresponding segmentation masks
        X_seg = self.seg_gen[idx][0]

        if self.is_training:
            # Apply augmentation to the batch
            aug_orig_batch = np.zeros_like(X_orig)
            aug_seg_batch = np.zeros_like(X_seg)

            for i in range(len(X_orig)):
                aug_orig_batch[i], aug_seg_batch[i] = self._apply_augmentation(
                    X_orig[i], X_seg[i]
                )

            return [aug_orig_batch, aug_seg_batch], y

        return [X_orig, X_seg], y

    def on_epoch_end(self):
        """Reset the indices at the end of each epoch"""
        if self.is_training:
            np.random.shuffle(self.indices)
            self.orig_gen.on_epoch_end()
            self.seg_gen.on_epoch_end()