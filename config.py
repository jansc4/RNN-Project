import torch

# paths
TRAIN_CSV = r"C:\Users\HARDPC\PycharmProjects\random\RoboticArm\train\attributes_normal_source_train.csv"
TEST_NORMAL_CSV = r"C:\Users\HARDPC\PycharmProjects\random\RoboticArm\test\attributes_normal_source_test.csv"
TEST_ANOMALY_CSV = r"C:\Users\HARDPC\PycharmProjects\random\RoboticArm\test\attributes_anomaly_source_test.csv"

# sensors
SENSOR_CONFIG = {
    'accelerometer': {
        'csv_column': 'ism330dhcx_acc',
        'parquet_columns': ["A_x [g]", "A_y [g]", "A_z [g]"],
        'enabled': True
    },
    'gyroscope': {
        'csv_column': 'ism330dhcx_gyro',
        'parquet_columns': ['G_x [mdps]', 'G_y [mdps]', 'G_z [mdps]'],
        'enabled': True
    },
    'microphone': {
        'csv_column': 'imp23absu_mic',
        'parquet_columns': ["MIC [Waveform]"],
        'enabled': True
    }
}


WINDOW_SIZE = 576
STEP_SIZE = 432

# for training
HIDDEN_DIM = 128
LATENT_DIM = 32
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 5
ENABLE_AUTOGRAD_DETECT_ANOMALY = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
