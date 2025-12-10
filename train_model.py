import csv
import random
import math

# ============================
# 1. Load dataset
# ============================
def load_dataset(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append([
                float(r["f1_X_count"]),
                float(r["f2_O_count"]),
                float(r["f3_X_almost_win"]),
                float(r["f4_O_almost_win"]),
                float(r["f5_X_center"]),
                float(r["f6_X_corners"]),
                float(r["label"])
            ])
    return rows


data = load_dataset("dataset.csv")
random.shuffle(data)

# ============================
# 2. Train-test split (80/20)
# ============================
split = int(len(data) * 0.8)
train = data[:split]
test = data[split:]

X_train = [row[:-1] for row in train]
y_train = [row[-1] for row in train]

X_test = [row[:-1] for row in test]
y_test = [row[-1] for row in test]

n_features = len(X_train[0])

# ============================
# 3. Initialize weights
# ============================
weights = [0.0] * n_features
bias = 0.0

# ============================
# 4. Linear Regression (MSE)
#     predicted = W·X + b
# ============================

def predict(features):
    return sum(w*f for w, f in zip(weights, features)) + bias


# ============================
# 5. Train using Gradient Descent
# ============================
learning_rate = 0.001
epochs = 2000

for epoch in range(epochs):
    dw = [0.0] * n_features
    db = 0.0
    m = len(X_train)

    for x, y in zip(X_train, y_train):
        y_pred = predict(x)
        error = y_pred - y

        for i in range(n_features):
            dw[i] += (2/m) * error * x[i]
        db += (2/m) * error

    # update weights
    for i in range(n_features):
        weights[i] -= learning_rate * dw[i]
    bias -= learning_rate * db

    if epoch % 200 == 0:
        mse = sum((predict(x) - y)**2 for x, y in zip(X_train, y_train)) / m
        print(f"Epoch {epoch}, MSE={mse:.4f}")

# ============================
# 6. Evaluate on test set
# ============================
m_test = len(X_test)
mse_test = sum((predict(x) - y)**2 for x, y in zip(X_test, y_test)) / m_test

print("\n===== Final Results =====")
print("Weights:", weights)
print("Bias:", bias)
print("Test MSE:", mse_test)


# ============================
# 7. Print JS function
# ============================
print("\n===== JavaScript Model =====")
print("function mlEvaluation(features) {")
for i, w in enumerate(weights):
    print(f"  const w{i} = {w};")
print(f"  const bias = {bias};")
print("  return (")
print("    " + " + ".join([f"w{i}*features[{i}]" for i in range(n_features)]) + " + bias")
print("  );")
print("}")
