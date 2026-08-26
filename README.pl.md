🇬🇧 [English version](README.md)

# Wykrywanie anomalii sensorycznych za pomocą autoenkodera LSTM

Wykrywanie anomalii w danych czasowych z akcelerometru/żyroskopu poprzez trenowanie autoenkodera LSTM na normalnych wzorcach ruchu i oznaczanie wysokiego błędu rekonstrukcji jako anomalii.

## Motywacja

Dane sensoryczne (z ramienia robota, silnika, urządzenia noszonego) mają charakter sekwencyjny — każda próbka zależy od poprzednich. Anomalie (awarie czujnika, artefakty szumowe, problemy mechaniczne lub faktyczne odchylenie w mierzonym zjawisku) rzadko objawiają się jako pojedyncza zła wartość — zwykle są zerwaniem oczekiwanego wzorca w czasie. Sieci rekurencyjne, a zwłaszcza LSTM, dobrze modelują tego rodzaju zależności czasowe.

## Co zawiera projekt

Pełny pipeline, nie tylko eksperyment w notebooku:

- **Wczytywanie i przygotowanie danych** — surowe dane CSV z akcelerometru/żyroskopu → sekwencje okienkowe (`data_loader.py`, `prepare_sequences.py`, `prepare_lstm_data.py`)
- **Trening modelu** — autoenkoder LSTM uczony rekonstrukcji normalnych sekwencji ruchu (`train_lstm_autoencoder.py`)
- **Ocena anomalii** — próg błędu rekonstrukcji wyliczony z rozkładu danych treningowych, stosowany do oznaczania anomalnych sekwencji (`compute_threshold.py`, `detect_anomalies.py`)
- **Wizualizacja** danych treningowych i wykrytych anomalii (`visualize_results.py`)
- Notebook eksploracyjny (`RNN_EDA.ipynb`)
- Konteneryzacja (Docker) dla powtarzalności wyników

## Dane

Nagrania akcelerometru/żyroskopu z dwóch rzeczywistych stanowisk: ramienia robota oraz silnika bezszczotkowego.

## Wyniki

Dla ocenianego przebiegu: **6140** sekwencji, **803** oznaczone jako anomalie (**~13,1%** wykrywalności) przy progu błędu rekonstrukcji **~0,0020**.

## Stos technologiczny

Python · TensorFlow/Keras (autoenkoder LSTM) · NumPy · pandas · scikit-learn · SciPy · Docker

## Status

Projekt akademicki ze studiów. Nierozwijany aktywnie — utrzymywany jako element portfolio.
