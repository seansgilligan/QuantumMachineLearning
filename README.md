# Quantum Anomaly Detection on Weather Data

This project demonstrates a simple **quantum machine learning** approach for anomaly detection on a weather dataset.  
Using **PennyLane** and **Qiskit**, a variational quantum circuit is trained to classify anomalies in humidity levels based on temperature, humidity, and wind speed data.

---

## Dataset

- Dataset: [Weather History Dataset](https://www.kaggle.com/muthuj7/weather-dataset)
- The project uses features:
  - Temperature (°C)
  - Humidity (%)
  - Wind Speed (km/h)
  - Precipitation type (rain or snow)

---

## Methodology

- The data is preprocessed and normalized to fit quantum angle embedding requirements.
- Humidity anomalies are defined as humidity readings significantly above the mean (mean + 0.6 × std).
- A 3-qubit quantum circuit is constructed using:
  - AngleEmbedding to encode classical data into qubits.
  - BasicEntanglerLayers as variational layers.
- The model is trained using mini-batch gradient descent optimizer.
- Training progress is visualized with a loss curve.
- Final model performance is evaluated via classification report and ROC curve.

---

## Requirements

- Python 3.8+
- PennyLane
- Qiskit
- Scikit-learn
- Pandas
- Matplotlib
- Tqdm

Install dependencies via:

```bash
pip install pennylane qiskit scikit-learn pandas matplotlib tqdm


