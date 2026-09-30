# Image de base
FROM python:3.12-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système utiles


# Copie des dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du projet
COPY . .

RUN mkdir -p database

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

