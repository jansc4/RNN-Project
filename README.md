🇵🇱 [Polska wersja](README.pl.md)

# Sensor Anomaly Detection with an LSTM Autoencoder

Detecting anomalies in accelerometer/gyroscope time-series data by training an LSTM autoencoder on normal motion patterns and flagging high reconstruction error as anomalous.

## Motivation

Sensor data (from a robotic arm, a motor, a wearable) is sequential — each reading depends on the ones before it. Anomalies (sensor faults, noise artifacts, mechanical issues, or a genuine deviation in the physical system) rarely show up as a single bad value; they show up as a break in the expected pattern over time. Recurrent networks, and LSTMs in particular, are built to model exactly that kind of temporal dependency.

## What's here

A full pipeline, not just a notebook experiment:

- **Data loading & preprocessing** — raw accelerometer/gyroscope CSV data → windowed sequences (`data_loader.py`, `prepare_sequences.py`, `prepare_lstm_data.py`)
- **Model training** — an LSTM autoencoder trained to reconstruct normal motion sequences (`train_lstm_autoencoder.py`)
- **Anomaly scoring** — a reconstruction-error threshold computed from the training distribution, then applied to flag anomalous sequences (`compute_threshold.py`, `detect_anomalies.py`)
- **Visualization** of training data and detected anomalies (`visualize_results.py`)
- An exploratory notebook (`RNN_EDA.ipynb`)
- Dockerized for reproducibility

## Data

Accelerometer/gyroscope recordings from two real hardware setups: a robotic arm and a brushless motor.

## Results

On the evaluated run: **6,140** sequences scored, **803** flagged anomalous (**~13.1%** detection rate) at a reconstruction-error threshold of **~0.0020**.

## Tech stack

Python · TensorFlow/Keras (LSTM autoencoder) · NumPy · pandas · scikit-learn · SciPy · Docker

## Status

Academic project from my studies. Not actively maintained — kept here as a portfolio reference.
