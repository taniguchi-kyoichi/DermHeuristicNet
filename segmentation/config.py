class Config:
    # Dataset configuration
    IMAGE_SIZE = 299
    BATCH_SIZE = 16
    NUM_CLASSES = 7
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # Training configuration
    EPOCHS = 150
    INITIAL_LR = 0.0001
    MIN_LR = 1e-7
    LR_REDUCE_FACTOR = 0.5
    LR_PATIENCE = 5
    EARLY_STOPPING_PATIENCE = 15

    # Data augmentation configuration
    ROTATION_RANGE = 10
    SHIFT_RANGE = 0.1
    ZOOM_RANGE = 0.1

    # Model configuration
    FEATURE_FUSION_REDUCTION = 16
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1

    # Directory configuration
    DATASET_DIR = 'HAM10000'
    TRAIN_DIR = f'{DATASET_DIR}/train_dir'
    TEST_DIR = f'{DATASET_DIR}/test_dir'
    TRAIN_SEG_DIR = f'{DATASET_DIR}/train_segmentation'
    TEST_SEG_DIR = f'{DATASET_DIR}/test_segmentation'
    MODEL_DIR = 'models'
    LOG_DIR = 'logs'

    # Class weights
    CLASS_WEIGHTS = {
        0: 1.0,  # akiec
        1: 1.0,  # bcc
        2: 1.0,  # bkl
        3: 1.0,  # df
        4: 5.0,  # mel
        5: 1.0,  # nv
        6: 1.0,  # vasc
    }