import os
import tensorflow as tf
from model.triple_stream import TripleStreamModel
from model.data_generator import TripleStreamGenerator
from config import Config


def setup_callbacks(config):
    """Set up training callbacks"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    model_checkpoint_path = os.path.join(
        config.MODEL_DIR,
        'triple_stream_model_best_{val_accuracy:.4f}.h5'
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
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
            log_dir=os.path.join(config.LOG_DIR, 'triple_stream'),
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        )
    ]
    return callbacks


def setup_mixed_precision():
    """Setup mixed precision training if available"""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled")
        return True
    except Exception as e:
        print(f"Mixed precision training not available: {str(e)}")
        return False


def train_model(train_df, test_df):
    """Train the triple stream model"""
    config = Config()
    print("Initializing training process...")

    # Enable mixed precision if available
    mixed_precision = setup_mixed_precision()

    # Create data generators
    train_generator = TripleStreamGenerator(
        data_dir=config.TRAIN_DIR,
        seg_dir=config.TRAIN_SEG_DIR,
        color_dir=config.TRAIN_COLOR_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        is_training=True
    )

    val_generator = TripleStreamGenerator(
        data_dir=config.TEST_DIR,
        seg_dir=config.TEST_SEG_DIR,
        color_dir=config.TEST_COLOR_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        is_training=False
    )

    # Create and compile model
    print("Building and compiling model...")
    model = TripleStreamModel(
        input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    model_inst = model.build()

    # Adjust learning rate for mixed precision if enabled
    initial_lr = config.INITIAL_LR * 2.0 if mixed_precision else config.INITIAL_LR
    model_inst = model.compile_model(model_inst, learning_rate=initial_lr)

    # Print model summary
    model_inst.summary()

    # Get callbacks
    callbacks = setup_callbacks(config)

    # Setup GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")

    print("Starting model training...")
    try:
        history = model_inst.fit(
            train_generator,
            validation_data=val_generator,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            class_weight=config.CLASS_WEIGHTS,
            workers=1,  # M1 Macではマルチプロセスを避ける
            max_queue_size=10,
            use_multiprocessing=False,  # ピクリングエラーを避けるためFalseに設定
            verbose=1
        )

        print("Training completed successfully!")

        # Save training history
        history_path = os.path.join(config.MODEL_DIR, 'training_history.npy')
        np.save(history_path, history.history)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    return model_inst, history


if __name__ == "__main__":
    import pandas as pd
    from preprocess import process_dataset

    # Configure paths
    zip_path = "datasets/HAM10000.zip"
    metadata_path = "datasets/HAM10000_metadata.csv"

    # Process dataset if needed
    if not os.path.exists(Config().TRAIN_DIR):
        print("Processing dataset...")
        data_pd, train_df, test_df = process_dataset(zip_path, metadata_path)
    else:
        print("Loading existing dataset...")
        data_pd = pd.read_csv(metadata_path)
        train_df = data_pd[data_pd['train_test_split'] == 'train']
        test_df = data_pd[data_pd['train_test_split'] == 'test']

    # Train model
    model, history = train_model(train_df, test_df)