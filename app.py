import os
import time
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, flash, url_for, jsonify
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'agricluster_secret_2025'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PLOT_FOLDER']   = 'static/plots'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'],   exist_ok=True)

DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), 'dataset_clustering_pertanian_clean.csv')
FEATURE_COLS    = ['produksi_padi', 'produksi_jagung', 'luas_panen', 'produksi_sayuran']
REQUIRED_COLS   = ['nama_kabupaten_kota'] + FEATURE_COLS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_clustering(df, n_clusters):
    df = df.dropna(subset=FEATURE_COLS).copy()
    X        = df[FEATURE_COLS].copy()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    inertia   = round(float(kmeans.inertia_), 2)
    sil_score = round(float(silhouette_score(X_scaled, df['Cluster'])), 4) if n_clusters < len(df) else None
    cluster_counts  = df['Cluster'].value_counts().sort_index().to_dict()
    cluster_summary = df.groupby('Cluster')[FEATURE_COLS].mean().round(1).reset_index().to_dict(orient='records')
    pca           = PCA(n_components=2, random_state=42)
    X_pca         = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    explained_var = round(float(sum(pca.explained_variance_ratio_)) * 100, 1)
    timestamp     = int(time.time())
    plot_filename  = _make_cluster_plot(X_pca, df['Cluster'].values, centroids_pca, n_clusters, timestamp)
    elbow_filename = _make_elbow_plot(X_scaled, 10, timestamp)
    return dict(
        table_data      = df[['nama_kabupaten_kota', 'Cluster']].to_dict(orient='records'),
        raw_data        = df[REQUIRED_COLS].to_dict(orient='records'),
        plot_url        = url_for('static', filename=f'plots/{plot_filename}'),
        elbow_url       = url_for('static', filename=f'plots/{elbow_filename}'),
        n_clusters      = n_clusters,
        inertia         = inertia,
        sil_score       = sil_score,
        cluster_counts  = cluster_counts,
        cluster_summary = cluster_summary,
        feature_cols    = FEATURE_COLS,
        n_samples       = len(df),
        explained_var   = explained_var,
    )

def _make_cluster_plot(X_pca, labels, centroids_pca, n_clusters, timestamp):
    PALETTE = ['#00D4FF','#FF6B6B','#FFE66D','#A8FF78','#FF8EFF','#FF9A3C','#6BFFB8','#FF6B9D','#C3FF6B','#6B9DFF']
    fig, ax = plt.subplots(figsize=(11, 8), facecolor='#0B0F1A')
    ax.set_facecolor('#0B0F1A')
    for c in range(n_clusters):
        mask = labels == c
        col  = PALETTE[c % len(PALETTE)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=col, s=130, alpha=0.85,
                   edgecolors='white', linewidths=0.5, label=f'Cluster {c}', zorder=3)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='white', marker='*', s=420,
               zorder=5, edgecolors='#FFE66D', linewidths=1.5, label='Centroids')
    ax.grid(color='#1E2A45', linestyle='--', linewidth=0.6, alpha=0.8)
    ax.spines[:].set_visible(False)
    ax.set_xlabel('Principal Component 1', color='#8899BB', fontsize=11, labelpad=10)
    ax.set_ylabel('Principal Component 2', color='#8899BB', fontsize=11, labelpad=10)
    ax.tick_params(colors='#8899BB', labelsize=9)
    ax.legend(loc='upper right', framealpha=0.15, facecolor='#1A2035', edgecolor='#2D3F60', labelcolor='white', fontsize=9)
    ax.set_title(f'K-Means Clustering  ·  k = {n_clusters}  ·  PCA 2D Projection', color='white', fontsize=14, fontweight='bold', pad=18)
    fig.tight_layout()
    fname = f'cluster_{timestamp}.png'
    fig.savefig(os.path.join(app.config['PLOT_FOLDER'], fname), dpi=160, bbox_inches='tight', facecolor='#0B0F1A')
    plt.close(fig)
    return fname

def _make_elbow_plot(X_scaled, max_k, timestamp):
    ks       = list(range(2, min(max_k + 1, len(X_scaled))))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0B0F1A')
    ax.set_facecolor('#0B0F1A')
    ax.plot(ks, inertias, color='#00D4FF', linewidth=2.5, marker='o', markersize=8,
            markerfacecolor='#FFE66D', markeredgecolor='#0B0F1A')
    ax.fill_between(ks, inertias, alpha=0.12, color='#00D4FF')
    ax.grid(color='#1E2A45', linestyle='--', linewidth=0.6)
    ax.spines[:].set_visible(False)
    ax.set_xlabel('Number of Clusters (k)', color='#8899BB', fontsize=10)
    ax.set_ylabel('Inertia (WCSS)', color='#8899BB', fontsize=10)
    ax.tick_params(colors='#8899BB')
    ax.set_title('Elbow Method', color='white', fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    fname = f'elbow_{timestamp}.png'
    fig.savefig(os.path.join(app.config['PLOT_FOLDER'], fname), dpi=150, bbox_inches='tight', facecolor='#0B0F1A')
    plt.close(fig)
    return fname

@app.route('/', methods=['GET', 'POST'])
def index():
    n_clusters = 3
    if request.method == 'POST':
        mode = request.form.get('mode', 'upload')
        try:
            n_clusters = int(request.form.get('clusters', 3))
            n_clusters = max(2, min(10, n_clusters))
        except Exception:
            n_clusters = 3

        if mode == 'manual':
            try:
                rows = json.loads(request.form.get('rows_json', '[]'))
                df   = pd.DataFrame(rows)
                for col in FEATURE_COLS:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                if len(df) < n_clusters:
                    flash(f'Data hanya {len(df)} baris, minimal sama dengan k={n_clusters}.', 'danger')
                    return render_template('index.html', n_clusters=n_clusters)
                return render_template('index.html', **run_clustering(df, n_clusters))
            except Exception as e:
                flash(f'Error memproses data manual: {e}', 'danger')
                return render_template('index.html', n_clusters=n_clusters)

        if mode == 'default':
            df = pd.read_csv(DEFAULT_DATASET)
            return render_template('index.html', **run_clustering(df, n_clusters))

        # CSV upload
        if 'file' not in request.files or request.files['file'].filename == '':
            flash('Tidak ada file yang dipilih.', 'danger')
            return render_template('index.html', n_clusters=n_clusters)
        file = request.files['file']
        if not allowed_file(file.filename):
            flash('Hanya file CSV yang diperbolehkan.', 'danger')
            return render_template('index.html', n_clusters=n_clusters)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            df      = pd.read_csv(filepath)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                flash(f'Kolom tidak ditemukan: {missing}', 'danger')
                return render_template('index.html', n_clusters=n_clusters)
            return render_template('index.html', **run_clustering(df, n_clusters))
        except Exception as e:
            flash(f'Error membaca CSV: {e}', 'danger')
            return render_template('index.html', n_clusters=n_clusters)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    # GET — load default immediately
    try:
        df = pd.read_csv(DEFAULT_DATASET)
        return render_template('index.html', **run_clustering(df, n_clusters))
    except Exception as e:
        flash(f'Gagal memuat dataset default: {e}', 'danger')
        return render_template('index.html', n_clusters=n_clusters)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))