import csv
import json
import random
import math

# ================================================
# 1) LOAD DATA
# ================================================
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


# Load
data = load_dataset("dataset.csv")
random.shuffle(data)

# Separate features/labels
X = [row[:-1] for row in data]
y = [row[-1] for row in data]

# ================================================
# 2) NORMALIZATION (Min–Max)
# ================================================
n_samples = len(X)
n_features = len(X[0])

mins = [min(X[i][j] for i in range(n_samples)) for j in range(n_features)]
maxs = [max(X[i][j] for i in range(n_samples)) for j in range(n_features)]

def normalize_row(row):
    out = []
    for i in range(n_features):
        if maxs[i] - mins[i] == 0:
            out.append(0)
        else:
            out.append((row[i] - mins[i]) / (maxs[i] - mins[i]))
    return out

X = [normalize_row(row) for row in X]

# ================================================
# 3) TRAIN / VALIDATION / TEST SPLIT
# ================================================
train_split = int(0.7 * n_samples)
val_split   = int(0.85 * n_samples)

X_train = X[:train_split]
y_train = y[:train_split]

X_val = X[train_split:val_split]
y_val = y[train_split:val_split]

X_test = X[val_split:]
y_test = y[val_split:]


# ================================================
# 4) Initialize NN parameters
# 6 inputs → 4 hidden → 1 output
# ================================================
hidden = 4
W1 = [[random.uniform(-1, 1) for _ in range(hidden)] for _ in range(n_features)]
B1 = [0.0 for _ in range(hidden)]

W2 = [random.uniform(-1, 1) for _ in range(hidden)]
B2 = 0.0

# Activation
def tanh(x):
    return math.tanh(x)

def dtanh(x):
    return 1 - math.tanh(x)**2


# ================================================
# 5) Forward pass
# ================================================
def forward(x):
    hidden_out = []
    for j in range(hidden):
        s = B1[j]
        for i in range(n_features):
            s += x[i] * W1[i][j]
        hidden_out.append(tanh(s))

    out = B2
    for j in range(hidden):
        out += hidden_out[j] * W2[j]

    return tanh(out), hidden_out


# ================================================
# 6) TRAINING (Gradient Descent)
# ================================================
lr = 0.01
epochs = 500

for epoch in range(epochs):

    for x, target in zip(X_train, y_train):

        # forward
        out, hidden_out = forward(x)

        # error
        err = out - target

        # Backprop – Output layer
        d_out = err * dtanh(out)

        for j in range(hidden):
            W2[j] -= lr * d_out * hidden_out[j]
        B2 -= lr * d_out

        # Backprop – Hidden layer
        for j in range(hidden):
            d_h = d_out * W2[j] * dtanh(hidden_out[j])
            for i in range(n_features):
                W1[i][j] -= lr * d_h * x[i]
            B1[j] -= lr * d_h

    # Every 50 epochs: compute accuracy
    if epoch % 50 == 0:

        # Validation accuracy
        correct = 0
        for x, label in zip(X_val, y_val):
            pred, _ = forward(x)
            predicted_label = 1 if pred > 0 else -1
            if predicted_label == label:
                correct += 1

        acc = correct / len(X_val)
        print(f"Epoch {epoch}: Validation Accuracy = {acc:.4f}")


# ================================================
# 7) TEST ACCURACY
# ================================================
correct_test = 0
for x, label in zip(X_test, y_test):
    pred, _ = forward(x)
    predicted_label = 1 if pred > 0 else -1
    if predicted_label == label:
        correct_test += 1

test_acc = correct_test / len(X_test)
print("\n======================")
print("Final Test Accuracy =", test_acc)
print("======================\n")


# ================================================
# 8) SAVE MODEL WEIGHTS AS JSON
# ================================================
model = {
    "W1": W1,
    "B1": B1,
    "W2": W2,
    "B2": B2,
    "mins": mins,
    "maxs": maxs
}

with open("model.json", "w") as f:
    json.dump(model, f, indent=4)

print("Model saved as model.json")
