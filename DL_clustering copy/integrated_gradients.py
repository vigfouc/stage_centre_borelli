import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
from arguments import LAGS, HORIZON
from dataloader import create_sequences, convert_target_seq_to_binary, extract_target_points_of_interest, TimeSeriesDataset
from model import LSTMForcast  


csv_path = "./clean_raw_data.csv"
data = pd.read_csv(csv_path)
data.set_index('Time', inplace=True)

feature_cols = [
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
    ]

forcasting_data = data[feature_cols]
timestamps = forcasting_data.index

print("Creating sequences...")
X_seq, y_seq = create_sequences(forcasting_data, timestamps, flatten=False, stride=HORIZON)
y_seq_binary = convert_target_seq_to_binary(y_seq, model_type='classifier')
mask = extract_target_points_of_interest(y_seq)

forcasting_data = forcasting_data.drop("Mesure traitee CaOl Alcatron", axis=1)

test_dataset = TimeSeriesDataset(X_seq, y_seq_binary, mask)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMForcast(
        input_size=len(forcasting_data.keys()), 
        hidden_size=64,
        num_layer=3,  
        dropout_rate=0.3,  
        model="classifier")

model_path = './results/lstm_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


def forward_func(input_tensor):
    output = model(input_tensor)  # shape: (1, num_classes)
    return F.softmax(output, dim=1)[:, 1]  # assuming binary classification and class 1 is the focus

ig = IntegratedGradients(forward_func)

feature_names = list(forcasting_data.columns)
num_samples = int(len(test_loader) * 0.1)

all_attributions = []

for i, (inputs, target, _) in enumerate(test_loader):
    if i >= num_samples:
        break
    
    inputs = inputs.to(device).float()  # shape: (1, seq_len, num_features)
    inputs.requires_grad = True

    # Compute baseline as mean across the full dataset
    feature_means = torch.tensor(forcasting_data.mean().values, dtype=torch.float32)
    baseline = feature_means.repeat(inputs.shape[1], 1).unsqueeze(0).to(device)  # shape: (1, seq_len, num_features)

    # Run IG
    attributions, delta = ig.attribute(
        inputs=inputs,
        baselines=baseline,
        return_convergence_delta=True
    )
    
    all_attributions.append(attributions.squeeze(0).detach().cpu().numpy())


mean_attr = np.mean(np.stack(all_attributions), axis=0)  # shape: (seq_len, num_features)

# Transpose to get features on y-axis and time on x-axis
mean_attr_transposed = mean_attr.T  # shape: (num_features, seq_len)

# Normalize each feature (row) independently to [0, 1]
normalized_attr = np.zeros_like(mean_attr_transposed)
for i in range(mean_attr_transposed.shape[0]):
    row = mean_attr_transposed[i]
    row_min = row.min()
    row_max = row.max()
    if row_max - row_min > 1e-10:  # Avoid division by zero
        normalized_attr[i] = (row - row_min) / (row_max - row_min)
    else:
        normalized_attr[i] = 0  # If constant, set to 0

# Create heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(
    normalized_attr, 
    cmap='YlOrRd',  # Yellow-Orange-Red colormap for normalized importance
    vmin=0,
    vmax=1,
    yticklabels=feature_names,
    xticklabels=range(normalized_attr.shape[1]),
    cbar_kws={'label': 'Normalized Importance (0-1)'}
)

plt.title('Normalized Feature Importance over Time (Integrated Gradients)', fontsize=16, pad=20)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()