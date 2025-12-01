FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Instalacja zależności systemowych (jeśli potrzebne)
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalacja zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie kodu źródłowego
COPY src/ ./src/

COPY data/ ./data/
# Ustawienie katalogu roboczego
WORKDIR /app/src

# Komenda domyślna
CMD ["python3", "main.py", "--all"]