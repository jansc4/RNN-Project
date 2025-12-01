"""
data_plot.py

Wczytuje pliki CSV z katalogów `training_data_raw/` i `anomaly_data_raw/`,
stosuje filtr dolnoprzepustowy (reusing prepare_lstm_data.butter_lowpass_filter),
i tworzy wykres 5 przykładowych sekwencji treningowych oraz 5 anomalii.

Użycie:
	python data_plot.py

Zapisuje wykres do `examples_filtered_plot.png` w katalogu skryptu.
"""

import os
import glob
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse filtering utilities / ustawienia z istniejącego skryptu
from prepare_lstm_data import butter_lowpass_filter, USE_COLUMNS, CUT_OFF


def load_and_filter_csv(path: str, cutoff: float = CUT_OFF) -> pd.DataFrame:
	"""Wczytaj CSV, policz fs na podstawie kolumny `timestamp` i zastosuj filtr.

	Zwraca dataframe z przefiltrowanymi kolumnami z USE_COLUMNS.
	"""
	df = pd.read_csv(path)

	if "timestamp" not in df.columns:
		raise ValueError(f"Brak kolumny 'timestamp' w pliku: {path}")

	time = df["timestamp"].values
	if len(time) < 2:
		raise ValueError(f"Za mało próbek w pliku: {path}")

	dt = np.mean(np.diff(time))
	fs = 1.0 / dt

	df_filtered = df.copy()
	for col in USE_COLUMNS:
		if col not in df.columns:
			raise ValueError(f"Brak kolumny '{col}' w pliku: {path}")
		df_filtered[col] = butter_lowpass_filter(df[col].values, cutoff, fs)

	return df_filtered


def accel_magnitude(df: pd.DataFrame) -> np.ndarray:
	"""Szybko oblicza wektorową magnitudę przyspieszeń ax, ay, az."""
	return np.sqrt(df["ax"].values ** 2 + df["ay"].values ** 2 + df["az"].values ** 2)


def pick_files(directory: str, n: int = 5) -> List[str]:
	files = sorted(glob.glob(os.path.join(directory, "*.csv")))
	if not files:
		raise FileNotFoundError(f"Brak plików CSV w katalogu: {directory}")
	# jeśli jest mniej niż n — bierz wszystkie
	return files[:n]


def plot_examples(train_dir: str = "training_data_raw",
				  anomalies_dir: str = "anomaly_data_raw",
				  n: int = 5,
				  out_path: str = "examples_filtered_plot.png") -> None:
	train_files = pick_files(train_dir, n)
	anom_files = pick_files(anomalies_dir, n)

	ncols = n
	fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6), sharex=False)

	for col_idx, filepath in enumerate(train_files):
		df = load_and_filter_csv(filepath)
		mag = accel_magnitude(df)

		ax = axes[0, col_idx] if n > 1 else axes[0]
		ax.plot(df["timestamp"], mag, lw=0.8)
		ax.set_title(f"TRAIN: {os.path.basename(filepath)}", fontsize=9)
		ax.set_xlabel("timestamp")
		ax.set_ylabel("|a| (filtered)")

	for col_idx, filepath in enumerate(anom_files):
		df = load_and_filter_csv(filepath)
		mag = accel_magnitude(df)

		ax = axes[1, col_idx] if n > 1 else axes[1]
		ax.plot(df["timestamp"], mag, lw=0.8, color="tab:red")
		ax.set_title(f"ANOMALY: {os.path.basename(filepath)}", fontsize=9)
		ax.set_xlabel("timestamp")
		ax.set_ylabel("|a| (filtered)")

	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	print(f"✔ Zapisano wykres — {out_path}")
	plt.show()


if __name__ == "__main__":
	# lokalne katalogi projektu
	BASE = os.path.dirname(__file__)

	train_dir = os.path.join(BASE, "training_data_raw")
	anomalies_dir = os.path.join(BASE, "anomaly_data_raw")

	try:
		plot_examples(train_dir, anomalies_dir, n=5, out_path=os.path.join(BASE, "examples_filtered_plot.png"))
	except Exception as e:
		print("Błąd:", e)
		raise

