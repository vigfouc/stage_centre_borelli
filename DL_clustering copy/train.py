import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataloader import create_sequences, extract_target_points_of_interest, convert_target_seq_to_binary, TimeSeriesDataset
from model import LSTMForcast
from custom_loss import WeightedMSELoss, FocalLoss
import argparse
import os

def train(X_seq,
          y_seq,
          mask,
          model,
          device,
          lr,
          num_epoch,
          train_split,
          batchsize,
          model_type,
          loss_type
          ):
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    #initializing model, optimizer and loss
    #---------------------------------------
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model_type == 'forecasting':
        if loss_type == 'mse':
            criterion = nn.MSELoss(reduction='none')
        if loss_type == 'weighted_mse':
            criterion = WeightedMSELoss()
    if model_type == 'classifier':
        if loss_type == 'bce':
            criterion = nn.BCELoss(reduction='none')
        if loss_type == 'focal':
            criterion = FocalLoss(alpha=0.1, gamma=2.0)
    #---------------------------------------

    #creating test and train set
    #---------------------------------------
    X_seq_train, X_seq_test, y_seq_train, y_seq_test, mask_train, mask_test = train_test_split(
        X_seq, y_seq, mask, 
        test_size=1-train_split, 
        shuffle=True, 
        random_state=42
        )

    train_dataset = TimeSeriesDataset(X_seq_train, y_seq_train, mask_train)
    test_dataset = TimeSeriesDataset(X_seq_test, y_seq_test, mask_test)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize)
#---------------------------------------

    train_losses, test_losses = [], []
    for epoch in range(num_epoch):
        
        #training
        #---------------------------------------
        model.train()
        train_loss = 0
        batch_idx = 0
        for X_batch, y_batch, mask_batch in train_loader:
            X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            masked_loss = loss[mask_batch.view(-1)].mean()
            masked_loss.backward()
            optimizer.step()
            
            train_loss += masked_loss.item()
            print(f"Epoch {epoch+1}/{num_epoch}, Batch {batch_idx+1}: training loss: {masked_loss:.4f}")
            
            batch_idx += 1
            
        train_losses.append(train_loss / len(train_loader))
        #---------------------------------------

        #testing
        #---------------------------------------        
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for X_batch, y_batch, mask_batch in test_loader:
                X_batch, y_batch, mask_batch = X_batch.to(device), y_batch.to(device), mask_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                masked_loss = loss[mask_batch.view(-1)].mean()
                test_loss += masked_loss.item()
            test_losses.append(test_loss / len(test_loader))
        print(f"Epoch {epoch+1}/{num_epoch} - Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
            
    epochs = np.arange(num_epoch)+1
    loss_df = pd.DataFrame({"train_loss": train_losses, "test_loss": test_losses, "epoch": epochs})
    loss_df.set_index("epoch", inplace=True)
    loss_df.to_csv(os.path.join(save_dir, "lstm_losses.csv"), index=False)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "lstm_model.pt"))          
        
    
if __name__ == "__main__":

    print("Begin Training")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epoch', type=int, default=15, help='number of epochs')
    parser.add_argument('--train_split', type=float, default=0.9, help='train split')
    parser.add_argument('--batchsize', type=int, default=1024, help='batchsize')
    parser.add_argument('--model_type', type=str, default='classifier', help='type of model: classifier or forecasting')
    parser.add_argument('--loss_type', type=str, default='focal', help='type of loss, be carefull that it fits with the model_type')
    args = parser.parse_args()

    device = torch.device(args.device)

    csv_path = "./clean_raw_data.csv"

    data = pd.read_csv(csv_path)
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
    X_seq, y_seq = create_sequences(forcasting_data, timestamps, flatten=False)

    # X_seq = X_seq[:300]
    # y_seq = y_seq[:300]

    print(f"X shape: {X_seq.shape}, y shape: {y_seq.shape}")  
        
    mask = extract_target_points_of_interest(y_seq)
    
    y_seq = convert_target_seq_to_binary(y_seq, model_type=args.model_type)

    model = LSTMForcast(
        input_size=len(forcasting_data.keys())-1, 
        hidden_size=64,
        num_layer=3,  
        dropout_rate=0.3,  
        model=args.model_type)
    
    train(X_seq,
          y_seq,
          mask,
          model,
          device,
          args.lr,
          args.num_epoch,
          args.train_split,
          args.batchsize,
          args.model_type,
          args.loss_type
          )