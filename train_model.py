import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_and_export():
    csv_file = "BMKG_Earthquake_Catalog.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} tidak ditemukan di folder kerja!")

    print(f"1. Membaca dataset '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print(f"   Ukuran awal data mentah: {df.shape}")

    # 1. Pembersihan & Feature Engineering (sesuai Cell 0 gempa.ipynb)
    cols_to_drop = [c for c in ['Unnamed: 2', 'No'] if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop).copy()

    # Ekstraksi Tanggal & Jam UTC (sesuai Cell 0 & Cell 4 gempa.ipynb)
    print("2. Ekstraksi tanggal & waktu (Tahun, Bulan, Jam UTC)...")
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d-%b-%Y')
    df_clean['Tahun'] = df_clean['Date'].dt.year.astype(int)
    df_clean['Bulan'] = df_clean['Date'].dt.month.astype(int)
    
    time_series = pd.to_datetime(df_clean['Time (UTC)'], format='%H:%M:%S', errors='coerce')
    df_clean['Jam'] = time_series.dt.hour.fillna(0).astype(int)

    # 5 Kategori Magnitudo
    bins = [0, 3.0, 4.0, 5.0, 6.0, 10.0]
    labels = ['Mikro (<3.0)', 'Minor (3.0-3.9)', 'Ringan (4.0-4.9)', 'Sedang (5.0-5.9)', 'Kuat (>=6.0)']
    df_clean['Kategori_Magnitudo'] = pd.cut(
        df_clean['Magnitude'], bins=bins, labels=labels, right=False
    )
    print("   Distribusi target Kategori_Magnitudo:")
    print(df_clean['Kategori_Magnitudo'].value_counts())

    # 2. Model Klasterisasi (K-Means k=4) (sesuai Cell 3 gempa.ipynb)
    print("\n3. Melatih klasterisasi spasial K-Means (k=4)...")
    features_clust = ['Latitude', 'Longitude', 'Depth (km)']
    scaler_cluster = StandardScaler()
    X_clust_scaled = scaler_cluster.fit_transform(df_clean[features_clust])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_clean['Cluster_KMeans'] = kmeans.fit_predict(X_clust_scaled)

    # Identifikasi otomatis mapping ID klaster (0-3) berdasarkan rata-rata koordinat dan kedalamannya
    cluster_stats = df_clean.groupby('Cluster_KMeans')[features_clust].mean()
    print("\n   Statistik Klaster (Rata-rata):")
    print(cluster_stats)

    cluster_mapping = {}
    shallow_clusters = []

    for cid in range(4):
        mean_depth = cluster_stats.loc[cid, 'Depth (km)']
        if mean_depth > 200:
            cluster_mapping[cid] = "Zona Subduksi Dalam (Laut Banda & Busur Dalam)"
        else:
            shallow_clusters.append(cid)

    for cid in shallow_clusters:
        mean_lon = cluster_stats.loc[cid, 'Longitude']
        if mean_lon < 105.0:
            cluster_mapping[cid] = "Zona Dangkal Barat (Sumatra–Selat Sunda)"
        elif 105.0 <= mean_lon <= 121.0:
            cluster_mapping[cid] = "Zona Dangkal Tengah–Selatan (Jawa–Bali–Nusra–Sulawesi)"
        else:
            cluster_mapping[cid] = "Zona Dangkal Timur (Maluku–Papua)"

    print("\n   Mapping Klaster Zona Seismik:")
    for cid, name in cluster_mapping.items():
        print(f"   - Cluster {cid}: {name} (Mean Lon: {cluster_stats.loc[cid, 'Longitude']:.2f}, Mean Depth: {cluster_stats.loc[cid, 'Depth (km)']:.2f} km)")

    # 3. Evaluasi Resmi 80:20 Stratified (sesuai Cell 4 & 5 gempa.ipynb)
    print("\n4. Evaluasi Resmi 80:20 Stratified (sesuai Cell 4 & 5 gempa.ipynb)...")
    num_cols = ['Latitude', 'Longitude', 'Depth (km)', 'Tahun', 'Bulan', 'Jam']
    cat_cols = ['Source', 'Mag Type', 'Cluster_KMeans']

    X = df_clean[num_cols + cat_cols]
    y = df_clean['Kategori_Magnitudo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Jumlah Data Latih: {len(X_train):,} | Data Uji: {len(X_test):,}")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    models = {
        'Logistic Regression (Baseline)': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=18, class_weight='balanced_subsample', random_state=42, n_jobs=-1
        )
    }

    # Sampel stratified 30.000 baris khusus hitung 3-Fold CV (sesuai Cell 4 gempa.ipynb)
    print("   Menghitung 3-Fold CV pada sampel stratified 30.000 baris...")
    X_cv, _, y_cv, _ = train_test_split(X_train, y_train, train_size=30000, random_state=42, stratify=y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    hasil_model = []
    rf_test_pipe = None
    rf_test_pred = None

    for nama, clf in models.items():
        print(f"   - Melatih & mengevaluasi {nama}...")
        pipe = Pipeline([('prep', preprocessor), ('model', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        f1_wei = f1_score(y_test, y_pred, average='weighted')
        cv_f1 = cross_val_score(pipe, X_cv, y_cv, cv=cv, scoring='f1_macro', n_jobs=-1).mean()
        
        hasil_model.append({
            'Model': nama,
            'Accuracy': round(acc, 4),
            'Macro F1-Score': round(f1_mac, 4),
            'Weighted F1-Score': round(f1_wei, 4),
            'Rata-rata CV (Macro F1)': round(cv_f1, 4)
        })
        
        if nama == 'Random Forest':
            rf_test_pipe = pipe
            rf_test_pred = y_pred

    df_hasil_model = pd.DataFrame(hasil_model)
    print("\n   Tabel Perbandingan Model Klasifikasi (Bab 5.2):")
    print(df_hasil_model.to_string(index=False))

    # Confusion Matrix & Detail Metrik per Kelas (sesuai Cell 5 gempa.ipynb)
    print("\n   Menghitung Confusion Matrix & Metrik per Kelas...")
    cm = confusion_matrix(y_test, rf_test_pred, labels=labels)
    total_uji = cm.sum()

    detail_kelas = []
    for i, kelas in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total_uji - (tp + fn + fp))
        
        prec = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
        rec = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        f1 = float(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
        
        detail_kelas.append({
            'Kategori Magnitudo': kelas,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })

    df_eval_kelas = pd.DataFrame(detail_kelas)
    print("\n   Tabel Rincian TP, TN, FP, FN & F1-Score per Kelas:")
    print(df_eval_kelas.to_string(index=False))

    # Feature Importance asli dari model Random Forest uji
    feature_names = rf_test_pipe.named_steps['prep'].get_feature_names_out()
    importances = rf_test_pipe.named_steps['model'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n   Top 10 Feature Importance:")
    print(feat_imp.head(10))

    # 4. Pelatihan Model Akhir Random Forest pada 100% Data untuk Aplikasi Web
    print("\n5. Melatih Model Akhir Random Forest pada 100% Data...")
    rf_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', rf_final)
    ])
    rf_pipeline.fit(X, y)
    print("   Model akhir berhasil dilatih pada seluruh data (217.807 baris)!")

    # 5. Buat Bank Sampel 500 Data Acak Historis Terstratifikasi (1998 - 2023)
    print("\n6. Mengambil 500 sampel data historis acak terstratifikasi (1998–2023)...")
    df_hist = df_clean[df_clean['Tahun'] <= 2023].copy()
    sample_targets = {
        'Mikro (<3.0)': 140,
        'Minor (3.0-3.9)': 140,
        'Ringan (4.0-4.9)': 100,
        'Sedang (5.0-5.9)': 70,
        'Kuat (>=6.0)': min(50, len(df_hist[df_hist['Kategori_Magnitudo'] == 'Kuat (>=6.0)']))
    }
    sampled_list = []
    for cat_name, n_sample in sample_targets.items():
        sub_df = df_hist[df_hist['Kategori_Magnitudo'] == cat_name]
        if len(sub_df) > 0:
            s = sub_df.sample(n=min(n_sample, len(sub_df)), random_state=42)
            sampled_list.append(s)
            
    df_samples = pd.concat(sampled_list).sample(frac=1.0, random_state=42) # Shuffle
    historical_samples = []
    for _, row in df_samples.iterrows():
        historical_samples.append({
            'Event ID': str(row.get('Event ID', '')),
            'Date': str(pd.to_datetime(row['Date']).strftime('%d-%b-%Y')),
            'Time (UTC)': str(row.get('Time (UTC)', '')),
            'Latitude': float(row['Latitude']),
            'Longitude': float(row['Longitude']),
            'Depth (km)': float(row['Depth (km)']),
            'Magnitude': float(row['Magnitude']),
            'Mag Type': str(row.get('Mag Type', '-')),
            'Source': str(row.get('Source', 'NC')),
            'Tahun': int(row['Tahun']),
            'Bulan': int(row['Bulan']),
            'Jam': int(row['Jam']),
            'Kategori_Magnitudo': str(row['Kategori_Magnitudo'])
        })
    print(f"   Berhasil membuat {len(historical_samples)} sampel historis BMKG!")

    # 6. Simpan artefak dengan kompresi LZMA (compress=('lzma', 3))
    output_model_file = "model_gempa_bmkg.joblib"
    print(f"\n7. Menyimpan model ke '{output_model_file}' dengan kompresi LZMA...")
    
    eval_kmeans = [
        {'k': 2, 'Inertia': 464190, 'Silhouette Score': 0.3789, 'Status': 'Kurang representatif'},
        {'k': 3, 'Inertia': 336186, 'Silhouette Score': 0.4462, 'Status': 'Cukup'},
        {'k': 4, 'Inertia': 217549, 'Silhouette Score': 0.4823, 'Status': 'Dipilih (Optimal Geologis)'},
        {'k': 5, 'Inertia': 170582, 'Silhouette Score': 0.4984, 'Status': 'Over-segmented'},
        {'k': 6, 'Inertia': 142754, 'Silhouette Score': 0.4479, 'Status': 'Over-segmented'}
    ]
    eval_dbscan = {
        'kmeans_k': 4,
        'kmeans_noise_pct': 0.0,
        'kmeans_silhouette': 0.4823,
        'dbscan_clusters': 4,
        'dbscan_noise_pct': 2.01,
        'dbscan_silhouette': 0.3147
    }

    payload = {
        'scaler_cluster': scaler_cluster,
        'kmeans': kmeans,
        'rf_pipeline': rf_pipeline,
        'labels': labels,
        'cluster_mapping': cluster_mapping,
        'cluster_stats': cluster_stats.to_dict(),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'total_samples': len(df_clean),
        # Data Evaluasi Hasil Riset Notebook
        'eval_model_comparison': hasil_model,
        'eval_confusion_matrix': cm.tolist(),
        'eval_cm_classes': labels,
        'eval_metrics_per_class': detail_kelas,
        'eval_feature_importances': feat_imp.to_dict(),
        'eval_top_10_features': feat_imp.head(10).to_dict(),
        'eval_test_size': len(y_test),
        'eval_train_size': len(X_train),
        'eval_kmeans': eval_kmeans,
        'eval_dbscan': eval_dbscan,
        # Bank Sampel Historis
        'historical_samples': historical_samples
    }

    joblib.dump(payload, output_model_file, compress=('lzma', 3))
    file_size_mb = os.path.getsize(output_model_file) / (1024 * 1024)
    print(f"   Selesai! File '{output_model_file}' berhasil disimpan.")
    print(f"   Ukuran file model terkompresi LZMA: {file_size_mb:.2f} MB")

if __name__ == '__main__':
    train_and_export()
