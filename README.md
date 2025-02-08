# **Movie Recommender System**

## **1. Présentation du projet**
Ce projet est un système de recommandation de films et de classification d'affiches basé sur l'intelligence artificielle. Il permet :
1. **Classification d'affiches** : Déterminer si une image est une affiche de film et prédire son genre à l’aide d’un modèle CNN.
2. **Recommandation par affiche** : Trouver des films similaires à partir d’une affiche en utilisant un modèle mobilenet.
3. **Recommandation par description** : Recommander des films similaires à partir d’une description en utilisant trois techniques d’embeddings textuels :
   - **TF-IDF**
   - **GloVe**
   - **BERT**
4. **Interface utilisateur** : Une interface **Gradio** permet d’interagir facilement avec le modèle via une API **FastAPI**.

## **2. Déroulement du projet**
### **Phase 1 : Classification des affiches**
- Création d’un modèle CNN **PosterCNN** pour classifier les affiches de films selon leur genre.
- Entraînement du modèle et sauvegarde des poids sous le fichier **poster.pth**.

### **Phase 2 : Recommandation d’affiches par affiches**
- Utilisation de l’extraction de caractéristiques avec **MobileNet** et **ViT**.
- Construction d’un index **Annoy** pour stocker les embeddings des affiches.
- Sauvegarde des modèles et index sous les fichiers **poster.ann** et **poster.pkl**.

### **Phase 3 : Recommandation de films par description**
- Extraction des embeddings des synopsis avec **TF-IDF**, **GloVe** et **BERT**.
- Création des fichiers d’index **Annoy** pour accélérer la recherche de films similaires.
- Sauvegarde des modèles sous les fichiers **tfidf.pkl**, **glove.pkl**, **bert.pkl** et les index correspondants **tfidf.ann**, **glove.ann**, **bert.ann**.

### **Phase 4 : Création de l’interface Gradio et API FastAPI**
- Développement d’une API **Flask/FastAPI** pour exposer les fonctionnalités de prédiction et de recommandation.
- Intégration de **Gradio** pour une interface interactive permettant d’uploader des affiches et de rechercher des films.

### **Phase 5 : Stockage des données sur Google Drive**
- Stockage des modèles et données sur **Google Drive** pour alléger le dépôt Git.
- Modification du code pour télécharger automatiquement les fichiers lors du démarrage de l’API.

### **Phase 6 : Conteneurisation avec Docker**
- Création d’**images Docker** pour exécuter l’API et l’application Gradio.
- Définition d’un fichier **docker-compose** pour orchestrer les conteneurs.

---

## **3. Installation et Exécution**

### **Prérequis**
- **Python 3.9+**
- **Docker & Docker Compose**

### **Cloner le dépôt**
```bash
git clone https://github.com/votre-repo/movie-recommender.git
cd movie-recommender
```

### **Exécution avec Docker**
#### ** Costruire l'image Docker et exécuter les conteneurs**
```bash
docker-compose up --build
```
- **L'API** sera disponible sur : `http://127.0.0.1:5005/`
- **L'interface Gradio** sera accessible sur : `http://127.0.0.1:7860/`

#### ** Exécution manuelle (sans Docker)**
Dans un premier terminal, lancer l’API Flask :
```bash
python movie_api.py
```
Dans un second terminal, lancer l’interface utilisateur Gradio :
```bash
python movie_webapp.py
```

---

## **4. Utilisation**
### **API Flask**
#### **Prédiction du genre d’une affiche**
```bash
curl -X POST -F "file=@poster.jpg" http://127.0.0.1:5005/predict
```

#### **Recommandation de films par affiche**
```bash
curl -X POST -F "file=@poster.jpg" http://127.0.0.1:5005/recommend
```

#### **Recommandation de films par description**
```bash
curl -X POST http://127.0.0.1:5005/recommend_by_description -H "Content-Type: application/json" -d '{
  "description": "A space adventure with a heroic mission",
  "method": "bert"
}'
```

### **Interface Gradio**
1. Accéder à `http://127.0.0.1:7860/`
2. Charger une affiche pour prédire son genre ou obtenir des films similaires.
3. Saisir une description pour rechercher des films correspondants.
4. Sélectionner la méthode d’embedding (TF-IDF, GloVe, BERT).

---

## **5. Architecture du projet**
```
 movie-recommender
 ├── api/                   # API Flask
 │   ├── movie_api.py       # Code de l'API
 │   ├── model.py           # Modèles CNN
 │   ├── weights/               # Modèles et index Annoy
 │   ├── Dockerfile         # Dockerfile pour l'API
 │   ├── train_classifieur # entrainer le classifier de posters
 │   ├── requirements.txt   # Dépendances API
 ├── gradio/                # Interface utilisateur
 │   ├── movie_webapp.py    # Code de l'interface Gradio
 │   ├── Dockerfile         # Dockerfile pour l'interface
 │   ├── requirements.txt   # Dépendances Gradio
 ├── posters/               # Affiches de films
 ├── docker-compose.yml     # Configuration Docker Compose
 ├── README.md              # Documentation du projet
```

---



### **2️⃣ Vérification des conteneurs Docker**
```bash
docker ps -a
```
Si un conteneur plante :
```bash
docker-compose down && docker-compose up --build
```

### **3️⃣ L’API ou Gradio ne répond pas**
- Vérifier si les **ports 5005 et 7860** sont utilisés.
- Relancer les services manuellement.

---





## **8. Équipe du projet**
- **Mohamed El Moctar AHMED**
- **Skander RAHAL**
- **Adnane BEN ALI**

Projet réalisé dans le cadre d'un projet universitaire en **Machine Learning & Deep Learning**. 🚀
