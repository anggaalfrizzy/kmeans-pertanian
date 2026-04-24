import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =====================
# STYLE (BERSIH)
# =====================
sns.set(style="white")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("dataset_clustering_pertanian_clean.csv")

print("Kolom dataset:")
print(df.columns)

# =====================
# PILIH FITUR
# =====================
features = [
    "produksi_padi", 
    "produksi_jagung", 
    "luas_panen", 
    "produksi_sayuran"
]

X = df[features]

# =====================
# NORMALISASI
# =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# ELBOW METHOD
# =====================
inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(k_range, inertia, marker='o')
plt.title("Metode Elbow")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

# =====================
# K-MEANS (K = 3)
# =====================
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# =====================
# SILHOUETTE SCORE
# =====================
score = silhouette_score(X_scaled, kmeans.labels_)
print("Silhouette Score:", score)

# =====================
# HASIL
# =====================
print("\nHasil Clustering:")
print(df[['nama_kabupaten_kota', 'Cluster']])

# =====================
# VISUALISASI FINAL (UPGRADE WARNA)
# =====================
plt.figure(figsize=(7,5))

sns.scatterplot(
    x=df['produksi_padi'], 
    y=df['produksi_jagung'], 
    hue=df['Cluster'],
    palette='Set1',      # 🔥 warna jelas
    s=60,               # 🔥 titik besar
    edgecolor='black'    # 🔥 outline
)

plt.xscale('log')
plt.yscale('log')

plt.title("Clustering Pertanian Jawa Barat")
plt.xlabel("Produksi Padi")
plt.ylabel("Produksi Jagung")

plt.legend(title="Cluster")

plt.tight_layout()
plt.show()
