import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("mushrooms.csv")

nuemrical = [   {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's': 5},
                {'f': 0, 'g': 1, 'y': 2, 's': 3},
                {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9},
                {'t': 1, 'f': 0},
                {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8},
                {'a': 0, 'd': 1, 'f': 2, 'n': 3},
                {'c': 0, 'w': 1, 'd': 2},
                {'b': 0, 'n': 1},
                {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g': 4, 'r': 5, 'o': 6, 'p': 7, 'u': 8, 'e': 9, 'w': 10, 'y': 11},
                {'e': 0, 't': 1},
                {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?': 6},
                {'f': 0, 'y': 1, 'k': 2, 's': 3},
                {'f': 0, 'y': 1, 'k': 2, 's': 3},
                {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
                {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o': 4, 'p': 5, 'e': 6, 'w': 7, 'y': 8},
                {'p': 0, 'u': 1},
                {'n': 0, 'o': 1, 'w': 2, 'y': 3},
                {'n': 0, 'o': 1, 't': 2},
                {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n': 4, 'p': 5, 's': 6},
                {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r': 4, 'o': 5, 'u': 6, 'w': 7, 'y': 8},
                {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v': 4, 'y': 5},
                {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u': 4, 'w': 5, 'd': 6}  ]

y = df['class'].map({'e': 0, 'p': 1}).values
X = df.drop('class', axis=1)

for i, col in enumerate(X.columns):
    X[col] = X[col].map(nuemrical[i])

X_tensor = torch.tensor(X.values, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(1)

train_split = int(0.8 * len(X_tensor))
X_train, y_train = X_tensor[:train_split], y_tensor[:train_split]
X_test, y_test = X_tensor[train_split:], y_tensor[train_split:]

class MushroomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=22, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_0 = MushroomModel()

model_0.to("mps")
device = 'mps'

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(model_0.parameters(), lr=0.001)

epochs = 10000

for epoch in range(epochs):
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
    #if epoch % 100 == 0:
        #print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test Loss: {test_loss:.5f}")

model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

results = y_preds[-10:].detach().cpu().numpy()
results_list = []
for r in results:
    results_list.append(round(r[0]))

actual_values = []
for r in y_test[-10:].detach().cpu().numpy():
    actual_values.append(int(r[0]))

print(results_list)
print(actual_values)

#I use the last 10 predictions and round them, even with the loss at around 0.15, the model still gives accurate binary classifications after rounding