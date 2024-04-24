import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load data
train_data_path = './train.csv'
test_data_path = './test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Data Preprocessing and Feature Engineering
def preprocess_data(train_data, test_data):
    # Fill missing values
    for dataset in [train_data, test_data]:
        dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
        dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # Extract titles from names
    for dataset in [train_data, test_data]:
        dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Encode categorical variables
    label_encoder_features = ['Sex', 'Ticket', 'Embarked', 'Title']
    for column in label_encoder_features:
        le = LabelEncoder()
        concat_data = pd.concat([train_data[column], test_data[column]])  # Combine data to encode uniformly
        le.fit(concat_data)
        train_data[column] = le.transform(train_data[column])
        test_data[column] = le.transform(test_data[column])

    # Create family size and is alone features
    for dataset in [train_data, test_data]:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 1
        dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    # Drop columns not used
    drop_elements = ['PassengerId', 'Name', 'Cabin', 'SibSp', 'Parch']
    train_data = train_data.drop(drop_elements, axis = 1)
    test_data = test_data.drop(drop_elements, axis = 1)

    return train_data, test_data

train_data, test_data = preprocess_data(train_data, test_data)

# Prepare data for training
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(X_train, y_train)
predictions = rf.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, predictions)}')

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(X_train, y_train)
print(f'Best Parameters: {CV_rf.best_params_}')

# Retrain model with best parameters
rf_best = RandomForestClassifier(**CV_rf.best_params_)
rf_best.fit(X_train, y_train)
predictions = rf_best.predict(X_val)
print(f'Improved Validation Accuracy: {accuracy_score(y_val, predictions)}')

# Prepare submission
test_predictions = rf_best.predict(test_data)
submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_data_path)['PassengerId'],
    "Survived": test_predictions
})
submission.to_csv('./submission.csv', index=False)
print("Submission file is saved as 'submission.csv'.")
