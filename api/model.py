import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Modèle pour la prédiction des genres
class MoviePosterNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MoviePosterNet, self).__init__()
        # Load EfficientNet avec des poids pré-entraînés
        self.base_model = models.efficientnet_b1(pretrained=True)
        # Remplacer la dernière couche pour correspondre au nombre de genres
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


# Modèle pour les recommandations (extraction d'embeddings)
class RecommenderNet(nn.Module):
    def __init__(self, embedding_dim=576):
        super(RecommenderNet, self).__init__()
        # Utiliser MobileNet comme modèle de base
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.features = mobilenet.features  # Les caractéristiques extraites par MobileNet
        self.pool = mobilenet.avgpool       # Pooling global
        self.flatten = nn.Flatten()        # Applatir le tenseur final pour obtenir les embeddings

    def forward(self, x):
        x = self.features(x)  # Extraire les caractéristiques
        x = self.pool(x)      # Appliquer le pooling global
        x = self.flatten(x)   # Applatir le tenseur
        return x
class PosterCNN(nn.Module):
    def __init__(self):
        super(PosterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)  # Sortie unique pour probabilité
        self.sigmoid = nn.Sigmoid()  # Activation pour obtenir une probabilité

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))  # Probabilité entre [0,1]
        return x  # Retourne un score de probabilité


if __name__ == '__main__':
    # Test du modèle MoviePosterNet
    x = torch.rand(16, 3, 224, 224)
    num_classes = 10  # Nombre de genres
    genre_model = MoviePosterNet(num_classes=num_classes)
    y = genre_model(x)
    assert y.shape == (16, num_classes)  # Sortie : (batch_size, num_classes)
    print("MoviePosterNet forward pass successful. Output shape:", y.shape)

    # Test du modèle RecommenderNet
    embedding_dim = 576  # Dimension des embeddings
    recommender_model = RecommenderNet(embedding_dim=embedding_dim)
    embeddings = recommender_model(x)
    assert embeddings.shape == (16, embedding_dim)  # Sortie : (batch_size, embedding_dim)
    print("RecommenderNet forward pass successful. Output shape:", embeddings.shape)
