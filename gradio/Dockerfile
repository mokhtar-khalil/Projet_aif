# Utiliser l'image officielle Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY movie_webapp.py ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port de l'application Gradio
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860

# Lancer Gradio
CMD ["python", "movie_webapp.py"]
