# Projekt: Wykrywanie anomalii sensorycznych przy użyciu rekurencyjnych sieci neuronowych

## 1. Informacje podstawowe

**Typ projektu:** Projekt badawczy z wykorzystaniem LSTM  
**Technologie:** Python, LSTM, Deep Learning  
**Status:** Faza koncepcyjna

---

## 2. Temat i cel projektu

### Temat

Wykrywanie anomalii sensorycznych używając rekurencyjnych sieci neuronowych

### Cel

Wykrywanie anomalii sensorycznych polega na identyfikacji nietypowych wzorców w danych pomiarowych, które odbiegają od normalnego zachowania systemu (np. sygnały EKG, PPS, temperatury, drgań).

Takie anomalie mogą oznaczać:

- awarie czujnika
- artefakty szumowe
- błędy transmisji
- lub faktyczne nieprawidłowości w mierzonym zjawisku (np. arytmie serca)

---

## 3. Charakterystyka danych sensorycznych

Dane sensoryczne mają **charakter sekwencyjny** – każda próbka jest zależna od poprzednich.

Z tego powodu używa się **rekurencyjnych sieci neuronowych (RNN)**, które potrafią modelować zależności w czasie.

---

## 4. Metodologia

### Model: Prawdopodobne LSTM (Long Short-Term Memory)

LSTM to zaawansowana forma RNN, która radzi sobie z:

- długoterminowymi zależnościami w danych
- problemem zanikającego gradientu
- uczeniem się wzorców czasowych

### Architektura sieci

Sieć składa się z:

- **Neuronów z rekurencją** (Embed Network / Unfolded/Unfolded Network)
- Struktura pozwala na przetwarzanie sekwencji danych wejściowych
- Model uczy się normalnych wzorców, a następnie wykrywa odstępstwa

---

## 5. Dane wejściowe

### Przykładowe dane (pierwsze 5 wierszy):

|Time|Δ x [g]|Δ y [g]|Δ z [g]|
|---|---|---|---|
|0|343.292029|-0.009272|-0.067344|
|1|343.292170|-0.004392|-0.065392|
|2|343.292312|0.005856|-0.065880|
|3|343.292453|0.005856|-0.059048|
|4|343.292595|0.005368|-0.062464|

Dane pochodzą  z **akcelerometru/żyroskopu** (dane z robota).

---

## 6. Dane eksperymentalne

### Dane gotowe:

- **Robotic Arm** - akcelerometr i żyroskop
- **Brushless motor** - akcelerometr i żyroskop

### Dane do pozyskania eksperymentalnego:

- akcelerometr
- żyroskop

---

## 7. Pipeline projektu

```
Dane gotowe → Metoda → Prawdopodobne LSTM → Dane pozyskane
                ↓                              ↓
              Temat  ←─────────────────────── LSTM
                                               ↓
                                     TESTY PRAKTYCZNE 
```

---

## 8. Licencja i klasyfikacja danych

### Schemat licencjonowania:

- **Nadzonowane** (Supervised Learning)
- **Nienadzorowane** (Unsupervised Learning)

Obie metody mogą być wykorzystane w zależności od dostępności danych oznaczonych.

---

## 9. Testowanie

**Testy praktyczne - do pozyskania**

Wymagane są testy na rzeczywistych danych, aby zweryfikować skuteczność modelu w warunkach praktycznych.

---

## 10. Następne kroki

1. **Przygotowanie danych**: Czyszczenie i normalizacja danych z akcelerometru/żyroskopu
2. **Implementacja modelu LSTM**: Stworzenie architektury sieci neuronowej
3. **Trening modelu**: Uczenie na danych normalnych
4. **Wykrywanie anomalii**: Testowanie na danych z anomaliami
5. **Walidacja**: Testy praktyczne na rzeczywistych urządzeniach
6. **Optymalizacja**: Dostrojenie hiperparametrów
7. **Dokumentacja**: Przygotowanie raportu końcowego

---
