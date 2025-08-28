import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/StressLevelDataset.csv")

X = df.drop(
    ["stress_level", "anxiety_level", "self_esteem", "mental_health_history", "depression", "blood_pressure"], axis=1).values
Y = df["stress_level"].values 

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.long)

dataset = TensorDataset(X, Y)

train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

input_size = X.shape[1]

if __name__ == "__main__":
    print(X)
    print(X.shape)
    
    print(Y.shape)
    
    max_values = df.max()
    print(max_values)