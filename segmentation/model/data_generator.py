import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class TripleStreamGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, seg_dir, color_dir, image_size=299, batch_size=16, is_training=True):
        """
        Initialize the triple stream data generator.

        Parameters:
        -----------
        data_dir : str
            Directory containing original images
        seg_dir : str
            Directory containing segmentation masks
        color_dir : str
            Directory containing color diversity maps
        image_size : int
            Size of the input images
        batch_size : int
            Number of samples per batch
        is_training : bool
            Whether this generator is for training or validation/testing
        """
        self.data_dir = data_dir
        self.seg_dir = seg_dir
        self.color_dir = color_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.is_training = is_training

        # Base preprocessing generator
        self.base_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
        )

        # Original image generator
        self.orig_gen = self.base_datagen.flow_from_directory(
            directory=self.data_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=self.is_training,
            class_mode='categorical'
        )

        # Segmentation mask generator
        self.seg_gen = self.base_datagen.flow_from_directory(
            directory=self.seg_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=self.is_training,
            class_mode='categorical'
        )

        # Color diversity map generator
        self.color_gen = self.base_datagen.flow_from_directory(
            directory=self.color_dir,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            shuffle=self.is_training,
            class_mode='categorical'
        )

        # Data augmentation configuration
        if self.is_training:
            self.augmentor = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='reflect'
            )

        # Initialize indices
        self.indices = np.arange(len(self.orig_gen))
        if self.is_training:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Return the number of batches per epoch"""
        return len(self.orig_gen)

    def _apply_augmentation(self, orig_img, seg_img, color_img):
        """
        Apply the same augmentation to all three images

        Parameters:
        -----------
        orig_img : ndarray
            Original image
        seg_img : ndarray
            Segmentation mask
        color_img : ndarray
            Color diversity map

        Returns:
        --------
        tuple
            Augmented versions of all three images
        """
        # Generate random transformation parameters
        transform_params = self.augmentor.get_random_transform(orig_img.shape)

        # Apply same transformation to all three images
        aug_orig = self.augmentor.apply_transform(orig_img, transform_params)
        aug_seg = self.augmentor.apply_transform(seg_img, transform_params)
        aug_color = self.augmentor.apply_transform(color_img, transform_params)

        return aug_orig, aug_seg, aug_color

    def on_epoch_end(self):
        """Reset indices at the end of each epoch"""
        if self.is_training:
            np.random.shuffle(self.indices)
            self.orig_gen.on_epoch_end()
            self.seg_gen.on_epoch_end()
            self.color_gen.on_epoch_end()

    def __getitem__(self, idx):
        """
        Get a batch of data

        Parameters:
        -----------
        idx : int
            Batch index

        Returns:
        --------
        tuple
            ([original_images, segmentation_masks, color_maps], labels)
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Get original images and labels
        orig_batch = self.orig_gen[idx]
        X_orig, y = orig_batch[0], orig_batch[1]

        # Get corresponding segmentation masks and color maps
        X_seg = self.seg_gen[idx][0]
        X_color = self.color_gen[idx][0]

        if self.is_training:
            # Apply augmentation to the batch
            aug_orig_batch = np.zeros_like(X_orig)
            aug_seg_batch = np.zeros_like(X_seg)
            aug_color_batch = np.zeros_like(X_color)

            for i in range(len(X_orig)):
                aug_orig_batch[i], aug_seg_batch[i], aug_color_batch[i] = self._apply_augmentation(
                    X_orig[i], X_seg[i], X_color[i]
                )

            return [aug_orig_batch, aug_seg_batch, aug_color_batch], y

        return [X_orig, X_seg, X_color], y

    def get_steps_per_epoch(self):
        """Return the number of steps per epoch"""
        return len(self)

    def get_class_weights(self):
        """Return the class weights from the original generator"""
        if hasattr(self.orig_gen, 'class_weights'):
            return self.orig_gen.class_weights
        return None

    def get_classes(self):
        """Return the classes from the original generator"""
        return self.orig_gen.classes

    def reset(self):
        """Reset all generators"""
        self.orig_gen.reset()
        self.seg_gen.reset()
        self.color_gen.reset()
        if self.is_training:
            np.random.shuffle(self.indices)