import csv
import json
import math
import random

# ==========================
# 1) LOAD DATA
# ==========================
def load_dataset(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append([
                float(row["f1_X_count"]),
                float(row["f2_O_count"]),
                float(row["f3_X_almost_win"]),
                float(row["f4_O_almost_win"]),
                float(row["f5_X_center"]),
                float(row["f6_X_corners"]),
                float(row["label"])
            ])
    return data


data = load_dataset("dataset.csv")
random.shuffle(data)

# Split X/Y
X = [row[:-1] for row in data]
Y = [row[-1] for row in data]
N = len(X)

# ==========================
# 2) NORMALIZATION (Min-Max)
# ==========================
def normalize(dataset):
    mins = [min(col) for col in zip(*dataset)]
    maxs = [max(col) for col in zip(*dataset)]
    
    norm = []
    for row in dataset:
        norm.append([(row[i] - mins[i]) / (maxs[i] - mins[i] + 1e-9) for i in range(len(row))])
    return norm, mins, maxs

X, mins, maxs = normalize(X)

# ==========================
# 3) TRAIN / VAL / TEST SPLIT
# ==========================
train_end = int(0.7 * N)
val_end   = int(0.85 * N)

X_train = X[:train_end]
Y_train = Y[:train_end]

X_val = X[train_end:val_end]
Y_val = Y[train_end:val_end]

X_test = X[val_end:]
Y_test = Y[val_end:]

# ==========================
# 4) NETWORK ARCHITECTURE
# 6 → 4 → 1  (tanh)
# ==========================
input_size = 6
hidden_size = 4
output_size = 1

# Xavier Initialization
def xavier(n_in, n_out):
    return [[random.uniform(-1, 1) * math.sqrt(6 / (n_in + n_out)) 
             for _ in range(n_out)] for _ in range(n_in)]

W1 = xavier(input_size, hidden_size)
B1 = [0.0] * hidden_size

W2 = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
B2 = 0.0

# ==========================
# 5) ACTIVATIONS
# ==========================
def tanh(x):
    return math.tanh(x)

def tanh_deriv(x):
    return 1 - math.tanh(x)**2

# ==========================
# 6) FORWARD PASS
# ==========================
def forward(x):
    # Hidden layer
    h_raw = []
    h_act = []
    for j in range(hidden_size):
        s = B1[j]
        for i in range(input_size):
            s += x[i] * W1[i][j]
        h_raw.append(s)
        h_act.append(tanh(s))

    # Output layer
    out_raw = B2
    for j in range(hidden_size):
        out_raw += h_act[j] * W2[j]

    out = tanh(out_raw)
    return h_raw, h_act, out_raw, out

# ==========================
# 7) BACKPROP
# ==========================
def train_step(x, y, lr):
    global W1, B1, W2, B2

    # Forward
    h_raw, h_act, out_raw, out = forward(x)

    # Loss derivative (MSE): dL/dout = 2(out - y)
    d_out = 2 * (out - y) * tanh_deriv(out_raw)

    # Gradients for W2, B2
    dW2 = [d_out * h_act[j] for j in range(hidden_size)]
    dB2 = d_out

    # Gradients for W1, B1
    dW1 = [[0.0]*hidden_size for _ in range(input_size)]
    dB1 = [0.0]*hidden_size

    for j in range(hidden_size):
        dh = d_out * W2[j] * tanh_deriv(h_raw[j])
        dB1[j] += dh
        for i in range(input_size):
            dW1[i][j] += dh * x[i]

    # Update weights
    for j in range(hidden_size):
        W2[j] -= lr * dW2[j]
    B2 -= lr * dB2

    for i in range(input_size):
        for j in range(hidden_size):
            W1[i][j] -= lr * dW1[i][j]

    for j in range(hidden_size):
        B1[j] -= lr * dB1[j]

# ==========================
# 8) TRAIN LOOP
# ==========================
epochs = 200
lr = 0.01

for e in range(epochs):
    # Train
    for x, y in zip(X_train, Y_train):
        train_step(x, y, lr)

    # Validation MSE
    val_loss = 0
    for x, y in zip(X_val, Y_val):
        _, _, _, out = forward(x)
        val_loss += (out - y)**2
    val_loss /= len(X_val)

    if e % 20 == 0:
        print(f"Epoch {e} | Val Loss = {val_loss:.4f}")

# ==========================
# 9) TEST EVALUATION
# ==========================
test_loss = 0
for x,y in zip(X_test, Y_test):
    _, _, _, out = forward(x)
    test_loss += (out-y)**2
test_loss /= len(X_test)

print("Test Loss =", test_loss)

# ==========================
# 10) SAVE WEIGHTS TO JSON
# ==========================
weights_json = {
    "W1": W1,
    "B1": B1,
    "W2": W2,
    "B2": B2,
    "mins": mins,   # needed to normalize in JS
    "maxs": maxs
}

with open("weights.json", "w") as f:
    json.dump(weights_json, f, indent=4)

print("Saved weights.json!")
