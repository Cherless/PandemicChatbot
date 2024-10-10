import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# input dimensions
input_size = 100  # embedding size
hidden_size = 128  # number of neurones in the hidden layer
output_size = 2 #for binary classification or multi whatever
num_layers = 2  # Number of LSTM layers


#def the model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(ChatbotModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM Layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer (output layer)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Embedding input words
        x = self.embedding(x)

        # LSTM forward pass
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # LSTM outputs

        # Pass the LSTM output through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out


# Initialize the model
model = ChatbotModel(input_size, hidden_size, output_size, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# criterion = nn.CrossEntropyLoss()  #
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
#
# # Training loop
# for epoch in range(num_epochs):
#     model.train()  # training mode
#    running_loss = 0.0
#
#     for inputs, labels in train_loader:
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#
#         # Backward pass et optimisation
#         optimizer.zero_grad()  # gradient to zero
#         loss.backward()  # retropropagation
#         optimizer.step()  # updating weights
#
#         running_loss += loss.item()
#
#         print(f"epoch {epoch + 1}/{num_epochs}, Loss : {running_loss / len(train_loader)}")
