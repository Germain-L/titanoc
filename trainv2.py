import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data Loading and Preprocessing
def load_and_preprocess_data():
    print("Loading data...")
    train_data = pd.read_csv('./train.csv')
    test_data = pd.read_csv('./test.csv')
    
    # Feature Engineering
    print("Feature engineering...")
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
    train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)
    test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

    # Fill missing values
    print("Handling missing values...")
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Embarked'].fillna('S', inplace=True)
    train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

    # Encoding categorical data
    print("Encoding categorical data...")
    for column in ['Sex', 'Embarked']:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])

    # Normalizing data
    print("Normalizing data...")
    scaler = StandardScaler()
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    train_data[features] = scaler.fit_transform(train_data[features])
    test_data[features] = scaler.transform(test_data[features])

    # Splitting the data
    X_train, X_val, y_train, y_val = train_test_split(train_data[features], train_data['Survived'], test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.long).to(device)

    print("Data loading and preprocessing complete.")
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = load_and_preprocess_data()

# Neural Network with Dropout and BatchNorm
class TitanicNN(nn.Module):
    def __init__(self):
        super(TitanicNN, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

model = TitanicNN().to(device)
print("Model initialized.")

# Training the Model
def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_val).sum().item()
            accuracy = correct / len(y_val)

        print(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}')
    print("Training complete.")

train_model(model, X_train, y_train, X_val, y_val)

