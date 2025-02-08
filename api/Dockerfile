# Utiliser l'image officielle Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers nécessaires
COPY requirements.txt ./
COPY movie_api.py ./
COPY model.py ./
COPY train_classifieur.py ./
# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'API
EXPOSE 5005

# Commande pour démarrer l'API Flask
CMD ["python", "movie_api.py"]
