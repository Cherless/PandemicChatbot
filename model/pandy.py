import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Paramètres du modèle
input_size = 3  # Trois colonnes en entrée : Cumulative number of case(s), Number of deaths, Number recovered
hidden_size = 128  # Nombre de neurones dans la couche cachée
output_size = 1  # On va prédire une seule valeur (par ex. le nombre de morts ou récupérés)
num_layers = 2  # Nombre de couches LSTM
learning_rate = 0.001
num_epochs = 100
batch_size = 32  # Taille du batch


# Fonction de chargement du dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Convertir les colonnes nécessaires en entrées
    inputs = torch.tensor(df[['Cumulative number of case(s)', 'Number of deaths', 'Number recovered']].values,
                          dtype=torch.float32)
    # Par exemple, utiliser la colonne 'Number of deaths' comme label (à ajuster si nécessaire)
    labels = torch.tensor(df['Number of deaths'].values, dtype=torch.float32)

    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Charger le dataset
train_loader = load_data('../data/SARS/sars_2003_complete_dataset_clean.csv')


# Définition du modèle
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ChatbotModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)  # Ici, on utilise une simple couche linéaire pour le début
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))  # Ajouter une dimension pour le LSTM
        out = self.fc(out[:, -1, :])
        return out


# Initialiser le modèle
model = ChatbotModel(input_size, hidden_size, output_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()  # Utiliser la perte pour régression si on prédit un nombre
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Training finished!")
