# Image de base
FROM python:3.12-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système utiles
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du projet
COPY . .

# Port exposé
EXPOSE 8000

# Démarrage FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

