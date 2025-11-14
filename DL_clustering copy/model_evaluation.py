import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataloader import create_sequences, extract_target_points_of_interest, convert_target_seq_to_binary, TimeSeriesDataset
from model import LSTMForcast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from arguments import LAGS, HORIZON

if __name__ == "__main__":
    csv_path_losses = "./results/lstm_losses.csv"

    losses = pd.read_csv(csv_path_losses)
        
    plt.figure(figsize=(14,10))
    plt.plot(losses["train_loss"], 'b-', label='Train Loss')
    plt.plot(losses["test_loss"], 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    csv_path_features = "./clean_raw_data.csv"

    data = pd.read_csv(csv_path_features)
    data.set_index('Time', inplace=True)
    data = data[~data.index.duplicated(keep='first')]

    forcasting_data = data[[ 
        # 'Vitesse Moteur Four', 
        # 'Intensite Moteur Four',
        # 'Debit moyen entree farine Four  (CF)',
        # 'Debit Injection doseur tout produit ', 
        # 'Debit Injection SQ7',
        # 'Debit moyen SOLVEN in Kiln', 
        'Température sur chambre chaude ',        
        'Temperature Air Secondaire',    
        'Temperature Zone', 
        'Oxygene Sortie Four', 
        'NOx Sortie Four',
        'SO2 Sortie Four', 
        'CO Sortie Four', 
        # 'Vitesse Grille Refroidisseur', 
        # 'Vitesse Grille Lepol',
        # 'Vitesse Ventilateur Fumees ', 
        # 'Vitesse Ventilateur Recyclage',
        'Mesure traitee CaOl Alcatron'
    ]]

    # Standardize the data
    scaler = StandardScaler()
    forcasting_data_values = scaler.fit_transform(forcasting_data.values)
    forcasting_data = pd.DataFrame(
        forcasting_data_values,
        columns=forcasting_data.columns,
        index=forcasting_data.index
    )

    timestamps = forcasting_data.index
    
    
    print("Creating sequences...")
    X_seq, y_seq = create_sequences(forcasting_data, timestamps, flatten=False, stride=1)

    print(f"X shape: {X_seq.shape}, y shape: {y_seq.shape}")  
    
    y_seq_binary = convert_target_seq_to_binary(y_seq, model_type='classifier')

    mask = extract_target_points_of_interest(y_seq)
    
    train_split = 0.9

    X_seq_train, X_seq_test, y_seq_train, y_seq_test, mask_train, mask_test = train_test_split(
        X_seq, y_seq_binary, mask, 
        test_size=1-train_split, 
        shuffle=True, 
        random_state=42
    )

    train_dataset = TimeSeriesDataset(X_seq_train, y_seq_train, mask_train)
    test_dataset = TimeSeriesDataset(X_seq_test, y_seq_test, mask_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    model = LSTMForcast(
        input_size=len(forcasting_data.keys())-1, 
        hidden_size=64,
        num_layer=3,  
        dropout_rate=0.2,  
        )    
    model.load_state_dict(torch.load('./results/lstm_model.pt'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x_batch, y_batch, mask_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)

            output = model(x_batch).squeeze(0)  
            y_batch = y_batch.squeeze(0)
            mask_batch = mask_batch.squeeze(0)

            valid_preds = output[mask_batch.bool()].cpu().numpy()
            valid_targets = y_batch[mask_batch.bool()].cpu().numpy()

            preds.extend(valid_preds)
            targets.extend(valid_targets)

    binary_preds = (np.array(preds) > 0.5).astype(int)
    binary_targets = np.array(targets).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(binary_targets, binary_preds)
    labels = ['Class 0', 'Class 1']

    # Plot using seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    

    