# train_lstm_autoencoder.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ---------------------------
# HYPERPARAMS 
# ---------------------------
WINDOW_SIZE = 200   # musi być zgodne z tym co przygotowałeś
N_FEATURES = 6      # ax,ay,az,gx,gy,gz
LATENT_DIM = 64     # rozmiar wewnętrznej reprezentacji
BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = "lstm_autoencoder.h5"

def build_lstm_autoencoder(window_size, n_features, latent_dim):
    """Zbuduj model LSTM Autoencoder"""
    inp = Input(shape=(window_size, n_features))
    
    # ENCODER
    x = LSTM(128, return_sequences=True)(inp)
    x = LSTM(latent_dim, return_sequences=False)(x)
    
    # DECODER
    x = RepeatVector(window_size)(x)
    x = LSTM(latent_dim, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)
    
    model = Model(inp, out)
    return model


def train_model(X_train, X_val, MODEL_PATH=MODEL_PATH):
    print("\n" + "="*60)
    print("TRENOWANIE LSTM AUTOENCODER")
    print("="*60)

    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ X_val:   {X_val.shape}")

    model = build_lstm_autoencoder(
        WINDOW_SIZE,
        N_FEATURES,
        LATENT_DIM
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, save_best_only=True)
        ],
        verbose=1
    )

    return model, history



if __name__ == "__main__":
    train_model()
