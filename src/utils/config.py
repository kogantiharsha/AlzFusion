class Config:
    
    GENETIC_DATA_PATH = "Datasets-20251207T091459Z-1-001/Datasets/ALZ_Variant Datset/preprocessed_alz_data.npz"
    MRI_TRAIN_PATH = "Datasets-20251207T091459Z-1-001/Datasets/MRI Dataset/train.parquet"
    MRI_TEST_PATH = "Datasets-20251207T091459Z-1-001/Datasets/MRI Dataset/test.parquet"
    
    GENETIC_INPUT_DIM = 130
    GENETIC_HIDDEN_DIMS = [256, 128, 64]
    MRI_OUTPUT_DIM = 64
    FUSION_DIM = 128
    NUM_CLASSES = 9
    DROPOUT = 0.3
    USE_ATTENTION = True
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0
    
    MODEL_SAVE_DIR = "models"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
