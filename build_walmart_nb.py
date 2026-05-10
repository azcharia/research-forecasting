import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# --- Cell 1: Imports ---
cell_imports = """
import os
import requests
import zipfile
import io
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
"""
cells.append(nbf.v4.new_code_cell(cell_imports))

# --- Cell 2: Step 1 ---
cell_step1 = """
# 1. Pengambilan dan Pemahaman Data
dataset_path = "dataset_walmart"
file_path = os.path.join(dataset_path, "Walmart.csv")

if not os.path.exists(file_path):
    print("Mengunduh dataset Walmart...")
    os.makedirs(dataset_path, exist_ok=True)
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required to download from Kaggle API.")
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://www.kaggle.com/api/v1/datasets/download/yasserh/walmart-dataset"
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(dataset_path)
        print("Dataset berhasil diunduh dan diekstrak.")
    else:
        print(f"Gagal mengunduh: HTTP {response.status_code}")
else:
    print("Dataset sudah ada.")

df_raw = pd.read_csv(file_path)

print("=== 1. Pemahaman Data ===")
print("a. Variabel target: 'Weekly_Sales'")
print("b. Makna variabel target: Total penjualan (dalam USD) selama satu minggu pada suatu toko.")
print(f"c. Jumlah data observasi awal: {len(df_raw)} baris (untuk semua toko).")
print("d. Nama kolom waktu: 'Date'")
print(f"e. Variabel lain yang tersedia: {', '.join([c for c in df_raw.columns if c not in ['Weekly_Sales', 'Date']])}")
print(f"f. Missing value: {'Ya' if df_raw.isnull().sum().sum() > 0 else 'Tidak'}")
print("g. Pola penjualan: Biasanya ritel memiliki pola musiman (lonjakan di akhir tahun/liburan) dan fluktuasi mingguan.")
"""
cells.append(nbf.v4.new_code_cell(cell_step1))

# --- Cell 3: Step 2 ---
cell_step2 = """
# 2. Pra-pemrosesan Data
print("=== 2. Pra-pemrosesan Data ===")

# a. Mengubah kolom tanggal menjadi format datetime (format asal dd-mm-yyyy)
df = df_raw.copy()
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# f. Memilih satu toko tertentu (Sesuai instruksi: Store 30)
df_store30 = df[df['Store'] == 30].copy()
print(f"Jumlah data untuk Store 30: {len(df_store30)} minggu.")

# b & c. Menjadikan kolom tanggal sebagai index dan mengurutkan secara kronologis
df_store30.set_index('Date', inplace=True)
df_store30.sort_index(inplace=True)

# Memastikan data periodik mingguan (resample)
df_store30 = df_store30.resample('W-FRI').mean()

# d. Menangani missing value
if df_store30.isnull().sum().sum() > 0:
    print("Menangani missing value dengan interpolasi linier.")
    df_store30 = df_store30.interpolate(method='linear')

# e. Memilih variabel target
target_col = 'Weekly_Sales'
data_target = df_store30[[target_col]]

# g. Membagi data menjadi latih dan uji (80:20)
train_size = int(len(data_target) * 0.8)
train_data = data_target.iloc[:train_size]
test_data = data_target.iloc[train_size:]
print(f"Data latih: {len(train_data)} minggu, Data uji: {len(test_data)} minggu.")

# h. Normalisasi khusus RNN, LSTM, GRU
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)
"""
cells.append(nbf.v4.new_code_cell(cell_step2))

# --- Cell 4: Step 3 ---
cell_step3 = """
# 3. Eksplorasi Data Time Series
print("=== 3. Eksplorasi Data ===")

# a. Grafik penjualan mingguan berdasarkan waktu
plt.figure(figsize=(15, 4))
plt.plot(data_target.index, data_target[target_col], color='blue', linewidth=1.5)
plt.title("a. Penjualan Mingguan (Store 30)")
plt.ylabel("Weekly Sales")
plt.show()

# b. Grafik penjualan dalam satu tahun tertentu (2011)
plt.figure(figsize=(15, 4))
df_2011 = data_target[data_target.index.year == 2011]
plt.plot(df_2011.index, df_2011[target_col], color='orange', linewidth=2, marker='o')
plt.title("b. Penjualan Mingguan Tahun 2011")
plt.ylabel("Weekly Sales")
plt.show()

# c. Grafik rata-rata penjualan berdasarkan bulan
monthly_avg = data_target.resample('M').mean()
plt.figure(figsize=(15, 4))
plt.plot(monthly_avg.index, monthly_avg[target_col], color='green', marker='s')
plt.title("c. Rata-rata Penjualan Bulanan")
plt.ylabel("Avg Weekly Sales")
plt.show()

# e & f. ACF dan PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
plot_acf(data_target[target_col], lags=40, ax=axes[0], color='purple')
axes[0].set_title("e. Autocorrelation Function (ACF)")
plot_pacf(data_target[target_col], lags=40, ax=axes[1], color='brown')
axes[1].set_title("f. Partial Autocorrelation Function (PACF)")
plt.show()

print("Interpretasi (d):")
print("- Tren: Tidak terlihat tren naik/turun yang ekstrem; cenderung stasioner secara tren jangka panjang.")
print("- Musiman: Terlihat jelas adanya paku (lonjakan) berkala di grafik penjualan dan pola siklikal di plot ACF, menandakan ada musiman tahunan.")
"""
cells.append(nbf.v4.new_code_cell(cell_step3))

# --- Cell 5: Step 4 (SARIMA) ---
cell_step4 = """
# 4. Pemodelan Menggunakan SARIMA
print("=== 4. Pemodelan SARIMA ===")

# a. Uji stasioneritas (ADF Test)
adf_test = adfuller(train_data[target_col])
print(f"ADF Statistic: {adf_test[0]:.4f}, p-value: {adf_test[1]:.4f}")
if adf_test[1] < 0.05:
    print("-> Data stasioner.")
else:
    print("-> Data tidak stasioner.")

# b & c. Parameter SARIMA (karena data mingguan, s=52 untuk musiman tahunan)
# Agar proses fitting di notebook tidak terlalu lama, kita pakai orde musiman rendah
order = (1, 1, 1)
seasonal_order = (1, 1, 0, 52) 
print(f"Menggunakan Parameter: order={order}, seasonal_order={seasonal_order}")

# d. Melatih model SARIMA
model_sarima = SARIMAX(train_data[target_col], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
fitted_sarima = model_sarima.fit(disp=False)

# e. Prediksi
sarima_pred = fitted_sarima.forecast(steps=len(test_data)).values

# f & g. Evaluasi sementara (dibahas lebih lanjut di tahap 8 dan 9)
print("Model SARIMA berhasil dilatih dan menghasilkan prediksi.")
"""
cells.append(nbf.v4.new_code_cell(cell_step4))

# --- Cell 6: DL Helpers ---
cell_dl_helpers = """
# Helper Functions untuk Deep Learning (RNN, LSTM, GRU)
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

class TimeSeriesModel(nn.Module):
    def __init__(self, model_type, input_size=1, hidden_size=64, num_layers=1):
        super(TimeSeriesModel, self).__init__()
        self.model_type = model_type
        if model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_dl_model(model_type, scaled_train, scaled_test, scaler, seq_length=4, epochs=50, batch_size=16):
    # b. Membentuk sequence dengan timestep = 4 (4 minggu)
    X_train, y_train = create_sequences(scaled_train, seq_length)
    
    # Menyiapkan data test
    test_inputs = np.concatenate((scaled_train[-seq_length:], scaled_test), axis=0)
    X_test, y_test = create_sequences(test_inputs, seq_length)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    # c. Membagi data
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # d. Arsitektur model
    model = TimeSeriesModel(model_type=model_type)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # e. Melatih model
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
    # f. Melakukan prediksi
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_t).numpy()
        
    # g. Mengembalikan ke skala asli
    test_preds = scaler.inverse_transform(test_preds_scaled).flatten()
    return test_preds
"""
cells.append(nbf.v4.new_code_cell(cell_dl_helpers))

# --- Cell 7: Step 5 (RNN) ---
cell_step5 = """
# 5. Pemodelan Menggunakan RNN
print("=== 5. Pemodelan RNN ===")
# Parameter: seq_length=4 (4 minggu), epoch=50, batch_size=16
rnn_pred = train_dl_model('RNN', scaled_train, scaled_test, scaler, seq_length=4)
print("Prediksi RNN selesai.")
"""
cells.append(nbf.v4.new_code_cell(cell_step5))

# --- Cell 8: Step 6 (LSTM) ---
cell_step6 = """
# 6. Pemodelan Menggunakan LSTM
print("=== 6. Pemodelan LSTM ===")
lstm_pred = train_dl_model('LSTM', scaled_train, scaled_test, scaler, seq_length=4)
print("Prediksi LSTM selesai.")
"""
cells.append(nbf.v4.new_code_cell(cell_step6))

# --- Cell 9: Step 7 (GRU) ---
cell_step7 = """
# 7. Pemodelan Menggunakan GRU
print("=== 7. Pemodelan GRU ===")
gru_pred = train_dl_model('GRU', scaled_train, scaled_test, scaler, seq_length=4)
print("Prediksi GRU selesai.")
"""
cells.append(nbf.v4.new_code_cell(cell_step7))

# --- Cell 10: Step 8 (Evaluasi) ---
cell_step8 = """
# 8. Evaluasi Model
print("=== 8. Evaluasi Model ===")

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = test_data[target_col].values
models = ['SARIMA', 'RNN', 'LSTM', 'GRU']
preds = [sarima_pred, rnn_pred, lstm_pred, gru_pred]

results = []
for m, p in zip(models, preds):
    mae = mean_absolute_error(y_true, p)
    rmse = np.sqrt(mean_squared_error(y_true, p))
    mape = calculate_mape(y_true, p)
    r2 = r2_score(y_true, p)
    results.append([m, mae, rmse, mape, r2])

df_eval = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'MAPE', 'R2 Score'])
display(df_eval)

best_model = df_eval.loc[df_eval['RMSE'].idxmin()]['Model']
print(f"\nBerdasarkan nilai evaluasi (RMSE terkecil), model dengan performa terbaik adalah: {best_model}")
"""
cells.append(nbf.v4.new_code_cell(cell_step8))

# --- Cell 11: Step 9 (Visualisasi & Kesimpulan) ---
cell_step9 = """
# 9. Visualisasi, Analisis, dan Kesimpulan
print("=== 9. Visualisasi, Analisis, dan Kesimpulan ===")

test_index = test_data.index
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# a, b, c, d. Grafik masing-masing model
for m, p, c in zip(models, preds, colors):
    plt.figure(figsize=(15, 4))
    plt.plot(test_index, y_true, label='Aktual', color='#1f77b4', linewidth=2)
    plt.plot(test_index, p, label=f'Prediksi {m}', color=c, linewidth=2, linestyle='--')
    plt.title(f"Data Aktual vs Prediksi {m}")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.show()

# e. Grafik Gabungan
plt.figure(figsize=(15, 6))
plt.plot(test_index, y_true, label='Aktual', color='black', linewidth=3)
for m, p, c in zip(models, preds, colors):
    plt.plot(test_index, p, label=f'Prediksi {m}', color=c, linewidth=1.5, linestyle='--')
plt.title("Data Aktual vs Prediksi Seluruh Model (SARIMA, RNN, LSTM, GRU)")
plt.ylabel("Weekly Sales")
plt.legend()
plt.show()

# Analisis
rmse_lstm = df_eval.loc[df_eval['Model']=='LSTM', 'RMSE'].values[0]
rmse_rnn = df_eval.loc[df_eval['Model']=='RNN', 'RMSE'].values[0]
rmse_gru = df_eval.loc[df_eval['Model']=='GRU', 'RMSE'].values[0]

print("\n--- Analisis ---")
print(f"a. Model dengan error paling kecil: {best_model}.")
print("b. Apakah SARIMA mampu menangkap pola musiman? Ya, karena parameter seasonal s=52 membuat prediksi membentuk siklus yang relevan dengan data historis setahun ke belakang.")
print(f"c. Apakah LSTM memiliki hasil yang lebih baik dibandingkan RNN? {'Ya, LSTM terbukti memiliki tingkat error yang lebih rendah.' if rmse_lstm < rmse_rnn else 'Tidak, pada eksperimen dataset Store 30 ini RNN memiliki error yang lebih kecil dibandingkan LSTM.'}")
print(f"d. Apakah GRU lebih efisien/akurat dibandingkan LSTM? {'Ya, GRU mengungguli LSTM.' if rmse_gru < rmse_lstm else 'Tidak, LSTM tetap lebih superior dibandingkan GRU.'}")
print(f"e. Model yang paling direkomendasikan: {best_model}.")
print("f. Kelebihan & Kelemahan:")
print("   - SARIMA: Kelebihan: sangat kuat untuk data musiman berfrekuensi reguler yang panjang. Kelemahan: penentuan parameter (p,d,q) butuh komputasi yang berat dan sulit beradaptasi jika pola bergeser tiba-tiba.")
print("   - RNN: Kelebihan: Arsitektur sederhana dan cepat ditraining. Kelemahan: kesulitan menangkap pola jarak jauh karena vanishing gradient.")
print("   - LSTM: Kelebihan: Mampu mengingat memori jangka panjang (long-term). Kelemahan: Membutuhkan waktu komputasi yang lebih lambat dan rentan overfitting jika data sedikit.")
print("   - GRU: Kelebihan: Alternatif yang lebih cepat dan efisien dibanding LSTM dengan performa sebanding. Kelemahan: Karena gerbangnya lebih sedikit, memori komputasi tidak se-ekstensif LSTM pada pola sangat kompleks.")
"""
cells.append(nbf.v4.new_code_cell(cell_step9))

nb.cells = cells
with open("walmart_sales_forecasting.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook walmart_sales_forecasting.ipynb successfully created!")
