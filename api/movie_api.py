import argparse
import torch
import pickle
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import pandas as pd
import zipfile
from annoy import AnnoyIndex
import os
import gdown
import torch.nn.functional as F
from model import MoviePosterNet, RecommenderNet
from torchvision.models import vit_b_16
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


#   Configuration du périphérique (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#   Initialisation de Flask
app = Flask(__name__)

#   Définition des fichiers Google Drive
drive_links = {
    "bert_model": "https://drive.google.com/uc?id=1fltSMAOK_x6zCEPQqFf6CKR6TiM4EA64",
    "bert_index": "https://drive.google.com/uc?id=1v5zoAibfmFF1xeGGegg0wvO9vxbSc9sO",
    "glove_model": "https://drive.google.com/uc?id=1nsa_9aw5Vpe4gz0VPzPDTGKh6dDjGnU1",
    "glove_index": "https://drive.google.com/uc?id=13QJug_qN6jEv348JeUXvRRj6hA7ZTu1M",
    "tfidf_model": "https://drive.google.com/uc?id=1IfMXE6CDllcotYQHbQWSs-TlWyXTqQPA",
    "tfidf_index": "https://drive.google.com/uc?id=1kniZDmnqaa4r41C8P_Yjt-aSLlXqbMRU",
    "poster_net": "https://drive.google.com/uc?id=1zGq9NNs56LY9yvbHjAP6tNXRUbEnw7WN",
    "poster_index": "https://drive.google.com/uc?id=1cpmvLBdcmuudczzhYJ0rpBRZijLcgkXU",
    "movies_metadata": "https://drive.google.com/uc?id=1sKK14f9qYJBKFh7mEA0yx11jlEsirZut"
}

#  Lien du fichier ZIP des posters
zip_url = "https://drive.google.com/uc?export=download&id=1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf"
zip_path = "../posters.zip"

#   Définition des répertoires
weights_dir = "weights"
data_dir = "data"
posters_dir = "../posters"
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(posters_dir, exist_ok=True)

#  Fonction pour télécharger un fichier Google Drive
def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f" Téléchargement en cours : {output_path} ...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Fichier déjà présent : {output_path}")

#   Télécharger les fichiers nécessaires
for name, link in drive_links.items():
    # Déterminer le bon dossier (weights_dir pour modèles et index, data_dir pour les métadonnées)
    target_dir = weights_dir if name not in ["movies_metadata"] else data_dir

    # Déterminer l'extension correcte du fichier
    if "poster_net" == name:
        extension = ".pth"  # Modèles PyTorch (réseaux de neurones)
    elif "model" in name:
        extension = ".pkl"  # Modèles de vectorisation (ex: TF-IDF, GloVe, BERT)
    elif "index" in name:
        extension = ".ann"  # Index Annoy
    else:
        extension = ".csv"  # Métadonnées (ex: movies_metadata)

    # Construire le chemin de sortie correct
    output_path = os.path.join(target_dir, name + extension)

    # Télécharger le fichier si non présent
    download_file(link, output_path)

#   Extraire le fichier ZIP des posters
def extract_zip(zip_file, output_folder):
    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        print(f"Extraction du fichier ZIP dans {output_folder}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
    else:
        print(f"Dossier {output_folder} déjà extrait.")

#   Télécharger et extraire les posters
download_file(zip_url, zip_path)
extract_zip(zip_path, posters_dir)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

#   Chargement du dataset avec ImageFolder
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

poster_dataset = ImageFolder(root=posters_dir, transform=transform)
poster_loader = DataLoader(poster_dataset, batch_size=32, shuffle=False)

#   Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=os.path.join(weights_dir, 'poster_net.pth'))
parser.add_argument('--annoy_index', type=str, default=os.path.join(weights_dir, 'poster_index.ann'))
parser.add_argument('--tfidf_index', type=str, default=os.path.join(weights_dir, 'tfidf_index.ann'))
parser.add_argument('--bert_index', type=str, default=os.path.join(weights_dir, 'bert_index.ann'))
parser.add_argument('--glove_index', type=str, default=os.path.join(weights_dir, 'glove_index.ann'))
parser.add_argument('--tfidf_vectorizer', type=str, default=os.path.join(weights_dir, 'tfidf_model.pkl'))
parser.add_argument('--bert_model', type=str, default=os.path.join(weights_dir, 'bert_model.pkl'))
parser.add_argument('--glove_model', type=str, default=os.path.join(weights_dir, 'glove_model.pkl'))
parser.add_argument('--metadata_csv', type=str, default=os.path.join(data_dir, 'movies_metadata.csv'))
parser.add_argument('--poster_csv', type=str, default='weights/paths_list.csv')

args = parser.parse_args()

#   Chargement des modèles et des index
print("Chargement des modèles et des index Annoy...")
predict_model = MoviePosterNet().to(device)
predict_model.load_state_dict(torch.load(args.model_path, map_location=device))
predict_model.eval()

recommend_model = RecommenderNet(embedding_dim=576).to(device)
recommend_model.eval()

annoy_index = AnnoyIndex(576, 'angular')
annoy_index.load(args.annoy_index)

df_metadata = pd.read_csv(args.metadata_csv)[['title', 'overview']].dropna()

tfidf_annoy = AnnoyIndex(5, 'angular')
tfidf_annoy.load(args.tfidf_index)

glove_annoy = AnnoyIndex(100, 'angular')
glove_annoy.load(args.glove_index)

bert_annoy = AnnoyIndex(768, 'angular')
bert_annoy.load(args.bert_index)

with open(args.tfidf_vectorizer, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open(args.glove_model, 'rb') as f:
    glove_model = pickle.load(f)

with open(args.bert_model, 'rb') as f:
    bert_model = pickle.load(f)


# Mapping des genres
genre_mapping = ['Action', 'Animation', 'Comedy', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']

#   Routes de l'API



@app.route('/predict', methods=['POST'])
def predict():
    """ Prédire le genre d'un film """
    try:
        file = request.files['file']
        img_pil = Image.open(file.stream).convert('RGB')
        tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = predict_model(tensor)
            _, predicted = outputs.max(1)
        return jsonify({"prediction": genre_mapping[int(predicted[0])]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """ Recommander des films à partir d'un poster """
    try:
        file = request.files['file']
        img_pil = Image.open(file.stream).convert('RGB')
        #print("img_pil = ",img_pil)
        tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = recommend_model(tensor).squeeze(0).tolist()
        indices = annoy_index.get_nns_by_vector(vector, 5)
        #print("indices : ",indices)
        recommended_paths = [os.path.abspath(os.path.join(BASE_DIR, os.path.normpath(poster_dataset.imgs[i][0]))) for i in indices]
        #print(recommended_paths)
        return jsonify({"recommendations": recommended_paths})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend_by_description', methods=['POST'])
def recommend_by_description():
    """
    Recommande des films en fonction d'une description avec le choix de l'embedding.
    """
    try:
        data = request.json
        description = data.get("description", "")
        method = data.get("method", "tfidf")  # Default: TF-IDF

        if not description:
            return jsonify({"error": "No description provided"}), 400

        # Choix de l'embedding
        if method == "tfidf":
            vector = tfidf_vectorizer.transform([description]).toarray().flatten()
            annoy_index = tfidf_annoy
        elif method == "glove":
            words = description.split()
            vectors = [glove_model[word] for word in words if word in glove_model]
            vector = np.mean(vectors, axis=0) if vectors else np.zeros(50)
            annoy_index = glove_annoy
        elif method == "bert":
            vector = bert_model.encode(description)
            annoy_index = bert_annoy
        else:
            return jsonify({"error": "Invalid method. Choose from tfidf, glove, bert."}), 400

        # Recherche dans Annoy
        indices = annoy_index.get_nns_by_vector(vector, 5)
        recommendations = df_metadata.iloc[indices][['title']].to_dict(orient="records")

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5005, debug=False)
