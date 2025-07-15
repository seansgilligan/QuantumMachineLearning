import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from sklearn.metrics import classification_report, roc_curve, auc

# 1. Veri yükleme ve ön işleme
print("[INFO] Dataset yükleniyor...")
df = pd.read_csv("weatherHistory.csv")
df = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Precip Type']].dropna()
df['precip'] = (df['Precip Type'] == 'rain').astype(int)

X_raw = df[['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']].astype(float)

# Nem anomalisi için eşik belirle
humidity_mean = df['Humidity'].mean()
humidity_std = df['Humidity'].std()
threshold = humidity_mean + 0.6 * humidity_std
Y = (df['Humidity'] > threshold).astype(int)  # Anomali: 1, Normal: 0
Y = 2 * Y - 1  # {-1, +1} formatına dönüştür

# Özellikleri [0, π] aralığına ölçekle (AngleEmbedding için)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X_raw)

X = np.array(X, requires_grad=False)
Y = np.array(Y.values, requires_grad=False)

print(f"[INFO] Toplam örnek: {len(X)}")

# 2. Quantum cihaz ve devre tanımı
n_wires = 3
dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev, interface="autograd")
def circuit(x, weights):
    qml.AngleEmbedding(x, wires=range(n_wires))
    qml.BasicEntanglerLayers(weights, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

# 3. Mini-batch kayıp fonksiyonu
def loss_batch(weights, X_batch, Y_batch):
    preds = np.array([circuit(x, weights) for x in X_batch])
    return np.mean((preds - Y_batch)**2)

# 4. Başlangıç ağırlıkları ve optimizer
weights = 0.01 * np.random.randn(3, n_wires, requires_grad=True)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

epochs = 30
batch_size = 16
num_batches = math.ceil(len(X) / batch_size)

losses = []

# 5. Eğitim döngüsü
print("[INFO] Mini-batch eğitim başlıyor...")
for epoch in range(epochs):
    perm = np.random.permutation(len(X))  # Her epoch başı verileri karıştır
    epoch_loss = 0
    for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
        idx = perm[i*batch_size:(i+1)*batch_size]
        X_batch, Y_batch = X[idx], Y[idx]
        weights = opt.step(lambda w: loss_batch(w, X_batch, Y_batch), weights)
        batch_loss = loss_batch(weights, X_batch, Y_batch)
        epoch_loss += batch_loss * len(X_batch)
    epoch_loss /= len(X)
    losses.append(epoch_loss)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} - Loss: {epoch_loss:.5f}")

# 6. Eğitim kaybını çiz
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Mini-batch Quantum Eğitim Kaybı")
plt.grid(True)
plt.show()

# 7. Model değerlendirme
print("[INFO] Model değerlendiriliyor...")
preds = np.array([circuit(x, weights) for x in tqdm(X, desc="Tahmin")])
labels = (preds > 0).astype(int)
true_labels = ((Y + 1) // 2)

print("[RESULT] Sınıflandırma Raporu:")
print(classification_report(true_labels, labels, digits=4))

fpr, tpr, _ = roc_curve(true_labels, preds)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi")
plt.legend()
plt.grid(True)
plt.show()
