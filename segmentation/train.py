import os
import tensorflow as tf
from model.dual_stream import DualStreamModel
from model.data_generator import DualStreamGenerator
from config import Config


def setup_callbacks(config):
    """Set up training callbacks"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=config.EARLY_STOPPING_PATIENCE,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_PATIENCE,
            verbose=1,
            min_lr=config.MIN_LR
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    return callbacks


def train_model(train_df, test_df):
    """Train the dual stream model"""
    config = Config()
    print("Initializing training process...")

    # Create data generators
    train_generator = DualStreamGenerator(
        data_dir=config.TRAIN_DIR,
        seg_dir=config.TRAIN_SEG_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        is_training=True
    )

    val_generator = DualStreamGenerator(
        data_dir=config.TEST_DIR,
        seg_dir=config.TEST_SEG_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        is_training=False
    )

    # Create and compile model
    print("Building and compiling model...")
    model = DualStreamModel(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    model_inst = model.build()
    model_inst = model.compile_model(model_inst)

    # Print model summary
    model_inst.summary()

    # Get callbacks
    callbacks = setup_callbacks(config)

    print("Starting model training...")
    history = model_inst.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=config.CLASS_WEIGHTS,
        verbose=1
    )

    print("Training completed!")
    return model_inst, history