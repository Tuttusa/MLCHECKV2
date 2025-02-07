import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from utils.mlCheck import Assume, Assert, propCheck
from typing import Tuple, Dict
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.preprocessing import LabelEncoder
from utils.Dataframe2XML import funcWriteXml
from datetime import datetime
import traceback

def get_adult_dataset():
    """
    Fetch and preprocess the UCI Adult dataset with consistent column naming.
    """
    # Fetch Adult dataset
    adult = fetch_ucirepo(id=2)
    df = adult.data.features
    target = adult.data.targets

    # Define columns to use with consistent naming
    columns_to_keep = [
        'age', 'workclass', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]

    # Select relevant columns
    df = df[columns_to_keep]

    # Rename columns to use underscores consistently
    column_mapping = {
        'education-num': 'education_num',
        'marital-status': 'marital_status',
        'capital-gain': 'capital_gain',
        'capital-loss': 'capital_loss',
        'hours-per-week': 'hours_per_week',
        'native-country': 'native_country'
    }
    df = df.rename(columns=column_mapping)

    # Handle missing values
    df = df.replace('?', np.nan)
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables
    le_dict = {}
    for col in df.select_dtypes(include=['object']):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Convert to numpy arrays
    X = df.values
    Y = np.where(target.values.ravel() <= '<=50K', -1, 1)

    # Create sensitive attributes dictionary
    sensitive = {
        'sex': df['sex'].values,
        'race': df['race'].values
    }

    return X, Y, sensitive

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def predict(self, X):
        """Convert predictions back to [-1,1] range"""
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            outputs = self(X_tensor)
            # Convert from [0,1] back to [-1,1]
            predictions = (outputs.numpy() > 0.5).astype(int) * 2 - 1
            return predictions.reshape(-1)

def train_model(X, y, model_type='dt'):
    """
    Train either a decision tree or neural network model
    
    Args:
        X: Feature matrix
        y: Target labels 
        model_type: 'dt' for decision tree or 'nn' for neural network
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'dt':
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
    else:  # neural network
        model = SimpleNN(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.BCELoss()
        
        # Convert to tensors and transform y from [-1,1] to [0,1]
        X_train_tensor = torch.FloatTensor(X_train)
        # Transform y from [-1,1] to [0,1]: y = (y + 1) / 2
        y_train_transformed = (y_train + 1) / 2
        y_train_tensor = torch.FloatTensor(y_train_transformed.reshape(-1, 1))
        
        # Training loop
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
    
    # Save model
    if model_type == 'dt':
        import joblib
        joblib.dump(model, f'model_{model_type}.joblib')
    else:
        torch.save(model.state_dict(), f'model_{model_type}.pt')
    
    return model


def test_individual_discrimination(X, y, protected_features=['sex', 'race']):
    """
    Test both decision tree and neural network models for individual discrimination
    
    Args:
        X: Feature matrix with column names
        y: Target labels
        protected_features: List of protected attribute names
    """
    # Define column names from Adult dataset
    columns = [
        'age', 'workclass', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'
    ]
    
    # Create DataFrame with features only (no target)
    df = pd.DataFrame(X, columns=columns)
    funcWriteXml(df)  # This will now only write feature columns to XML
    
    # Test both model types
    for model_type in ['dt', 'nn']:
        print(f"\nTesting {model_type.upper()} model for discrimination:")
        model_path = f'model_{model_type}.{"joblib" if model_type=="dt" else "pt"}'
        
        try:
            # Initialize MLCheck with correct case for model type
            propCheck(
                model_path=model_path,
                instance_list=['x', 'y'],
                xml_file='dataInput.xml',
                model_type='sklearn' if model_type=='dt' else 'Pytorch'  # Fixed case sensitivity
            )
            
            # Test for each protected feature
            for protected_attr in protected_features:
                print(f"\nTesting discrimination based on {protected_attr}")
                n_features = X.shape[1]
                
                # Set up assumptions
                for i in range(n_features):
                    col_name = columns[i]
                    if col_name == protected_attr:
                        # Different values for protected attribute
                        Assume('x[i] != y[i]', i)
                    else:
                        # Same values for all other attributes
                        Assume('x[i] = y[i]', i)
                
                # Assert prediction should be the same
                Assert('model.predict(x) == model.predict(y)')
                
        except Exception as e:
            print(f"Error testing {model_type} model: {str(e)}")
            # Log error details
            with open('mlcheck_error.log', 'a') as f:
                f.write(f"\nError testing {model_type} model at {datetime.now()}:\n")
                f.write(str(e))
                f.write("\n" + traceback.format_exc())

def main():
    # Get Adult dataset
    X, y, sensitive = get_adult_dataset()
    
    # Train both types of models
    dt_model = train_model(X, y, model_type='dt')
    nn_model = train_model(X, y, model_type='nn')
    
    # Test for discrimination
    test_individual_discrimination(X, y, protected_features=['sex', 'race'])

if __name__ == "__main__":
    main()