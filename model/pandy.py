import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Hypothèses : 100 dimensions pour l'input et 2 classes pour l'output (binaire)
input_size = 100  # embedding size (ce nombre doit correspondre à ton dataset)
hidden_size = 128  # nombre de neurones dans la couche cachée
output_size = 2  # classification binaire (ou modifie pour plus de classes)
num_layers = 2  # Nombre de couches LSTM
learning_rate = 0.001
num_epochs = 20
batch_size = 32  # Choisis une taille adaptée à ton dataset

# Dataset loader (ajuster avec ton dataset)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Supposons que les colonnes d'entrée soient de 'features' et la sortie soit 'label'
    inputs = torch.tensor(df[['feature1', 'feature2', 'feature3']].values, dtype=torch.float32)
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = load_data('your_dataset.csv')  # Charger le dataset

# Définition du modèle
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ChatbotModel, self).__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # Passer dans LSTM
        out = self.fc(out[:, -1, :])
        return out

# Initialiser le modèle
model = ChatbotModel(input_size, hidden_size, output_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
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
        loss = criterion(outputs, labels)

        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

print("Training finished!")
