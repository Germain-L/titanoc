import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
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
    
    print("Handling missing values...")
    for dataset in [train_data, test_data]:
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
    
    print("Encoding categorical data...")
    for column in ['Sex', 'Embarked']:
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        test_data[column] = le.transform(test_data[column])
    
    print("Normalizing data...")
    scaler = StandardScaler()
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_data[features] = scaler.fit_transform(train_data[features])
    test_data[features] = scaler.transform(test_data[features])
    
    # Splitting the training data for training and validation
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_data[features], train_data['Survived'], test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train = torch.tensor(train_features.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels.values, dtype=torch.long).to(device)
    X_val = torch.tensor(val_features.values, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_labels.values, dtype=torch.long).to(device)
    X_test = torch.tensor(test_data[features].values, dtype=torch.float32).to(device)
    
    print("Data loading and preprocessing complete.")
    return X_train, y_train, X_val, y_val, X_test

X_train, y_train, X_val, y_val, X_test = load_and_preprocess_data()

# Neural Network with Dropout and BatchNorm
class TitanicNN(nn.Module):
    def __init__(self):
        super(TitanicNN, self).__init__()
        self.layer1 = nn.Linear(7, 128)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.batchnorm2(self.layer2(x)))
        x = self.layer3(x)
        return x

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
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_val).sum().item()
            accuracy = correct / len(y_val)

        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {accuracy}')
    print("Training complete.")

train_model(model, X_train, y_train, X_val, y_val)

# Making Predictions on the Test Set
def make_predictions(model, X_test):
    print("Making predictions on test data...")
    model.eval()
    with torch.no_grad():
        predictions = torch.max(model(X_test), 1)[1]
    return predictions.cpu()

predictions = make_predictions(model, X_test)
print("Predictions:", predictions.numpy())

test_data = pd.read_csv('./test.csv')
# Create a DataFrame with the predictions
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": predictions.numpy()
})

# Save the DataFrame to a CSV file
submission.to_csv('./submission.csv', index=False)
print("Submission file is saved as 'submission.csv'.")

# Output the first few rows of your submission file to check
print(submission.head())

def calculate_accuracy(model, X, y):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        accuracy = 100 * correct / total
    return accuracy

# Calculate validation accuracy
val_accuracy = calculate_accuracy(model, X_val, y_val)
print(f'Validation Accuracy: {val_accuracy:.2f}%')