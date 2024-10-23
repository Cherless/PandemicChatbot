import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils import process_message_pipeline  # Importer le prétraitement
import numpy as np

# Hyperparamètres
input_size = 50  # La taille de l'input (vocab_size ou embedding_dim)
hidden_size = 128  # Nombre de neurones dans la couche cachée
output_size = 2  # Binaire (ou modifie pour plus de classes)
num_layers = 2  # Nombre de couches LSTM
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Exemple de Dataset personnalisé
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# La fonction `collate` pour le padding des séquences
def collate_fn(batch):
    texts, labels = zip(*batch)

    # Appliquer le padding aux séquences pour les rendre toutes de la même taille
    texts = [torch.tensor(text) for text in texts]
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

# Exemple de données et de labels
texts = [
    process_message_pipeline("Hello, this is a test message."),
    process_message_pipeline("Another example."),
    process_message_pipeline("Yet another message to process."),
]
# Convertir les tokens de mots en indices arbitraires pour l'exemple
word_to_idx = {word: i+1 for i, word in enumerate(set(word for text in texts for word in text))}
texts_idx = [[word_to_idx[word] for word in text] for text in texts]

labels = [0, 1, 0]  # Exemples de labels (binary)

# Split en train et test
train_texts, test_texts, train_labels, test_labels = train_test_split(texts_idx, labels, test_size=0.2)

# Charger le dataset
train_dataset = TextDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

test_dataset = TextDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Définition du modèle
class NLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(NLPModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Utiliser la dernière sortie LSTM
        return out

# Initialiser le modèle
model = NLPModel(input_size=len(word_to_idx)+1, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()
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

# Tester le modèle
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculer l'accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")
