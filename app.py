import os
import math
import time
import random
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests

# -------------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# -------------------------------------------------------------
st.set_page_config(
    page_title="GeoSeismik BMKG — Klasterisasi & Klasifikasi AI",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling CSS Modern
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
    }
    
    .card-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    }
    
    .impact-card {
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid transparent;
        margin-bottom: 12px;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. DAFTAR KOTA & WILAYAH REFERENSI SEISMIK (41 WILAYAH)
# -------------------------------------------------------------
INDONESIA_CITIES = [
    {"name": "Cianjur, Jawa Barat", "lat": -6.82, "lon": 107.14, "depth": 11.0, "source": "NC", "mag_type": "MLv"},
    {"name": "Bandung, Jawa Barat", "lat": -6.91, "lon": 107.60, "depth": 14.0, "source": "RC_07", "mag_type": "MLv"},
    {"name": "Sukabumi, Jawa Barat", "lat": -6.92, "lon": 106.92, "depth": 22.0, "source": "RC_07", "mag_type": "MLv"},
    {"name": "DKI Jakarta", "lat": -6.20, "lon": 106.81, "depth": 10.0, "source": "NC", "mag_type": "MLv"},
    {"name": "Selat Sunda / Banten", "lat": -6.95, "lon": 105.25, "depth": 15.0, "source": "NC", "mag_type": "Mw(mB)"},
    {"name": "Semarang, Jawa Tengah", "lat": -6.99, "lon": 110.42, "depth": 12.0, "source": "NC", "mag_type": "MLv"},
    {"name": "Yogyakarta / Bantul", "lat": -7.79, "lon": 110.36, "depth": 18.0, "source": "NC", "mag_type": "MLv"},
    {"name": "Surabaya, Jawa Timur", "lat": -7.25, "lon": 112.75, "depth": 10.0, "source": "NC", "mag_type": "MLv"},
    {"name": "Malang, Jawa Timur", "lat": -8.20, "lon": 112.50, "depth": 45.0, "source": "NC", "mag_type": "mb"},
    {"name": "Denpasar / Bali", "lat": -8.67, "lon": 115.21, "depth": 16.0, "source": "RC_03", "mag_type": "MLv"},
    {"name": "Mataram / Lombok, NTB", "lat": -8.58, "lon": 116.11, "depth": 15.0, "source": "RC_03", "mag_type": "MLv"},
    {"name": "Bima / Sumbawa, NTB", "lat": -8.46, "lon": 118.72, "depth": 25.0, "source": "RC_03", "mag_type": "MLv"},
    {"name": "Kupang / NTT", "lat": -10.17, "lon": 123.60, "depth": 30.0, "source": "RC_08", "mag_type": "mb"},
    {"name": "Banda Aceh", "lat": 5.55, "lon": 95.32, "depth": 15.0, "source": "RC_01", "mag_type": "Mw(mB)"},
    {"name": "Medan, Sumatera Utara", "lat": 3.59, "lon": 98.67, "depth": 18.0, "source": "RC_01", "mag_type": "MLv"},
    {"name": "Nias, Sumatera Utara", "lat": 1.15, "lon": 97.60, "depth": 28.0, "source": "RC_01", "mag_type": "mb"},
    {"name": "Padang, Sumatera Barat", "lat": -0.95, "lon": 100.35, "depth": 32.0, "source": "RC_06", "mag_type": "MLv"},
    {"name": "Bukittinggi, Sumatera Barat", "lat": -0.30, "lon": 100.37, "depth": 10.0, "source": "RC_06", "mag_type": "MLv"},
    {"name": "Pekanbaru, Riau", "lat": 0.50, "lon": 101.44, "depth": 20.0, "source": "RC_02", "mag_type": "MLv"},
    {"name": "Jambi", "lat": -1.61, "lon": 103.61, "depth": 15.0, "source": "RC_02", "mag_type": "MLv"},
    {"name": "Bengkulu", "lat": -3.80, "lon": 102.26, "depth": 35.0, "source": "RC_02", "mag_type": "mb"},
    {"name": "Palembang, Sumatera Selatan", "lat": -2.99, "lon": 104.75, "depth": 15.0, "source": "RC_02", "mag_type": "MLv"},
    {"name": "Bandar Lampung", "lat": -5.42, "lon": 105.26, "depth": 20.0, "source": "RC_02", "mag_type": "MLv"},
    {"name": "Pontianak, Kalimantan Barat", "lat": -0.02, "lon": 109.34, "depth": 10.0, "source": "NC", "mag_type": "ML"},
    {"name": "Banjarmasin, Kalimantan Selatan", "lat": -3.32, "lon": 114.59, "depth": 10.0, "source": "NC", "mag_type": "ML"},
    {"name": "Balikpapan, Kalimantan Timur", "lat": -1.26, "lon": 116.83, "depth": 12.0, "source": "NC", "mag_type": "ML"},
    {"name": "Samarinda, Kalimantan Timur", "lat": -0.50, "lon": 117.15, "depth": 10.0, "source": "NC", "mag_type": "ML"},
    {"name": "IKN Nusantara, Kaltim", "lat": -0.97, "lon": 116.70, "depth": 10.0, "source": "NC", "mag_type": "MLv"},
    {"name": "Palu / Donggala, Sulawesi Tengah", "lat": -0.89, "lon": 119.87, "depth": 10.0, "source": "RC_04", "mag_type": "Mw(mB)"},
    {"name": "Mamuju, Sulawesi Barat", "lat": -2.67, "lon": 118.89, "depth": 14.0, "source": "RC_04", "mag_type": "MLv"},
    {"name": "Makassar, Sulawesi Selatan", "lat": -5.14, "lon": 119.43, "depth": 15.0, "source": "RC_04", "mag_type": "MLv"},
    {"name": "Kendari, Sulawesi Tenggara", "lat": -3.99, "lon": 122.51, "depth": 18.0, "source": "RC_04", "mag_type": "MLv"},
    {"name": "Gorontalo", "lat": 0.54, "lon": 123.06, "depth": 30.0, "source": "RC_10", "mag_type": "mb"},
    {"name": "Manado, Sulawesi Utara", "lat": 1.49, "lon": 124.84, "depth": 25.0, "source": "RC_10", "mag_type": "mb"},
    {"name": "Ambon, Maluku", "lat": -3.69, "lon": 128.18, "depth": 15.0, "source": "RC_09", "mag_type": "MLv"},
    {"name": "Laut Banda (Subduksi Dalam)", "lat": -6.50, "lon": 129.50, "depth": 420.0, "source": "NC", "mag_type": "mb"},
    {"name": "Ternate, Maluku Utara", "lat": 0.78, "lon": 127.38, "depth": 20.0, "source": "RC_09", "mag_type": "mb"},
    {"name": "Sorong, Papua Barat Daya", "lat": -0.87, "lon": 131.25, "depth": 18.0, "source": "RC_05", "mag_type": "MLv"},
    {"name": "Manokwari, Papua Barat", "lat": -0.86, "lon": 134.06, "depth": 22.0, "source": "RC_05", "mag_type": "MLv"},
    {"name": "Jayapura, Papua", "lat": -2.53, "lon": 140.71, "depth": 18.0, "source": "RC_05", "mag_type": "MLv"},
    {"name": "Merauke, Papua Selatan", "lat": -8.49, "lon": 140.39, "depth": 15.0, "source": "RC_05", "mag_type": "MLv"}
]

# -------------------------------------------------------------
# 3. LOAD MODEL & RESOURCE CACHE (@st.cache_resource)
# -------------------------------------------------------------
MODEL_PATH = "model_gempa_bmkg.joblib"

@st.cache_resource(show_spinner="Memuat model komputasi K-Means, DBSCAN & Random Forest BMKG...")
def load_ml_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File artefak `{MODEL_PATH}` tidak ditemukan! Jalankan skrip `train_model.py` terlebih dahulu.")
        st.stop()
    return joblib.load(MODEL_PATH)

ml_data = load_ml_model()

# Profil Tektonik 4 Klaster K-Means (Berdasarkan Hasil Riset geologis)
CLUSTER_PROFILES = {
    0: {
        "nama": "Zona Dangkal Timur (Maluku–Papua)",
        "kedalaman_rata": "33.6 km (Kerak Dangkal)",
        "lempeng": "Sesar Geser Sorong, Busur Halmahera & Tunjaman Laut Maluku",
        "karakter": "Aktivitas seismik tektonik konvergen kompleks antara lempeng Pasifik, Filipina, dan mikro-kontinen Halmahera."
    },
    1: {
        "nama": "Zona Dangkal Tengah–Selatan (Jawa–Bali–Nusra–Sulawesi)",
        "kedalaman_rata": "34.8 km (Kerak Dangkal)",
        "lempeng": "Palung Busur Sunda, Busur Banda & Zona Sesar Palu-Koro",
        "karakter": "Subduksi frontal lempeng Indo-Australia terhadap Eurasia dengan aktivitas gempa dangkal berkerapatan tinggi."
    },
    2: {
        "nama": "Zona Dangkal Barat (Sumatra–Selat Sunda)",
        "kedalaman_rata": "33.7 km (Kerak Dangkal)",
        "lempeng": "Megathrust Sumatra & Zona Sesar Aktif Semangko (GSL)",
        "karakter": "Penunjaman miring (oblique subduction) lempeng Samudra Hindia dengan akumulasi regangan tektonik intensif."
    },
    3: {
        "nama": "Zona Subduksi Dalam (Laut Banda & Busur Dalam)",
        "kedalaman_rata": "387.6 km (Hiposenter Dalam)",
        "lempeng": "Zona Wadati-Benioff Subduksi Lempeng Tua Busur Banda",
        "karakter": "Patahan lempeng samudera purba yang menunjam ke astenosfer mantel bumi hingga kedalaman ratusan kilometer."
    }
}

# Koordinat sampel historis terstandarisasi untuk evaluasi DBSCAN
@st.cache_data
def get_historical_scaled():
    samples = ml_data.get("historical_samples", [])
    coords_arr = np.array([[s['Latitude'], s['Longitude'], s['Depth (km)']] for s in samples])
    coords_scaled = ml_data['scaler_cluster'].transform(pd.DataFrame(coords_arr, columns=['Latitude', 'Longitude', 'Depth (km)']))
    return coords_arr, coords_scaled

HIST_COORDS, HIST_COORDS_SCALED = get_historical_scaled()

def evaluate_dbscan_density(lat: float, lon: float, depth: float, eps: float = 0.40):
    """Mengevaluasi kepadatan spasial titik koordinat terhadap konsentrasi spasial DBSCAN katalog BMKG."""
    df_pt = pd.DataFrame([{'Latitude': lat, 'Longitude': lon, 'Depth (km)': depth}])
    pt_scaled = ml_data['scaler_cluster'].transform(df_pt[['Latitude', 'Longitude', 'Depth (km)']])
    dists = np.linalg.norm(HIST_COORDS_SCALED - pt_scaled, axis=1)
    neighbors_count = int(np.sum(dists <= eps))
    min_dist = float(np.min(dists))
    
    if neighbors_count >= 3 or min_dist <= 0.25:
        return {
            "status": "Zona Inti Padat Seismik (Core Cluster)",
            "is_core": True,
            "neighbors": neighbors_count,
            "min_dist": round(min_dist, 3),
            "bg": "#EFF6FF",
            "border": "#BFDBFE",
            "text": "#1E40AF",
            "icon": "🎯",
            "desc": f"Titik berada di dalam koridor sabuk seismik aktif berkepadatan kejadian tinggi ({neighbors_count} tetangga klaster historis)."
        }
    else:
        return {
            "status": "Titik Anomali / Jarang (Noise / Outlier)",
            "is_core": False,
            "neighbors": neighbors_count,
            "min_dist": round(min_dist, 3),
            "bg": "#FFFBEB",
            "border": "#FDE68A",
            "text": "#92400E",
            "icon": "⚠️",
            "desc": f"Titik berada di luar konsentrasi utama densitas gempa historis atau memiliki kedalaman anomali ({neighbors_count} tetangga spasial)."
        }

# -------------------------------------------------------------
# 4. INTEGRASI MONGODB ATLAS (RIWAYAT SIMULASI)
# -------------------------------------------------------------
def get_mongo_collection():
    """Mengambil koleksi MongoDB Atlas jika st.secrets['MONGO_URI'] dikonfigurasi."""
    try:
        mongo_uri = st.secrets.get("MONGO_URI")
        if not mongo_uri:
            return None
        import pymongo
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command('ping')
        db = client["geoseismik_bmkg"]
        return db["riwayat_simulasi"]
    except Exception:
        return None

def save_simulation(record):
    """Menyimpan log simulasi ke MongoDB Atlas (jika ada) dan selalu simpan ke session state lokal."""
    if "local_simulations" not in st.session_state:
        st.session_state["local_simulations"] = []
    
    st.session_state["local_simulations"].insert(0, record)
    if len(st.session_state["local_simulations"]) > 30:
        st.session_state["local_simulations"].pop()
        
    try:
        col = get_mongo_collection()
        if col is not None:
            col.insert_one(record.copy())
            return True
    except Exception:
        pass
    return False

def get_recent_history(limit=10):
    """Mengambil riwayat terbaru dari MongoDB Atlas (prioritas) atau session state."""
    try:
        col = get_mongo_collection()
        if col is not None:
            cursor = col.find({}, {"_id": 0}).sort("waktu_wib", -1).limit(limit)
            docs = list(cursor)
            if docs:
                return docs, True
    except Exception:
        pass
    return st.session_state.get("local_simulations", [])[:limit], False

# -------------------------------------------------------------
# 5. FUNGSI JARAK & KOTA TERDEKAT
# -------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def find_nearest_city(lat: float, lon: float):
    best_city = INDONESIA_CITIES[0]
    min_dist = float('inf')
    for city in INDONESIA_CITIES:
        d = haversine_distance(lat, lon, city["lat"], city["lon"])
        if d < min_dist:
            min_dist = d
            best_city = city
    return best_city["name"], round(min_dist, 1)

# -------------------------------------------------------------
# 6. HEADER APLIKASI
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 28px;">🌋</span>
                <h1 style="margin: 0; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">GeoSeismik BMKG</h1>
                <span style="background: #3B82F6; color: white; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;">RISET SPASIAL</span>
            </div>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 13px;">
                Klasterisasi Zona Seismik & Klasifikasi Kategori Magnitudo Gempa Indonesia (Katalog BMKG 1998–2024 | 217.807 Kejadian)
            </p>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <div class="status-badge">
                <span class="pulse-dot"></span>
                <span>K-Means (k=4) • DBSCAN • Random Forest</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 7. TAB UTAMA SISTEM
# -------------------------------------------------------------
tab_simulasi, tab_batch, tab_performa, tab_panduan = st.tabs([
    "🌐 Simulasi & Peta",
    "📂 Analisis Batch CSV",
    "📈 Performa Model",
    "📖 Panduan & Kamus Data"
])

# =============================================================
# TAB 1: SIMULASI & PETA (50:50 SYMMETRICAL LAYOUT)
# =============================================================
with tab_simulasi:
    # Inisialisasi Session State
    if "input_lat" not in st.session_state: st.session_state["input_lat"] = -6.840
    if "input_lon" not in st.session_state: st.session_state["input_lon"] = 107.050
    if "input_depth" not in st.session_state: st.session_state["input_depth"] = 11.0
    if "year" not in st.session_state: st.session_state["year"] = 2024
    if "month" not in st.session_state: st.session_state["month"] = 10
    if "hour" not in st.session_state: st.session_state["hour"] = 12
    if "source" not in st.session_state: st.session_state["source"] = "NC"
    if "mag_type" not in st.session_state: st.session_state["mag_type"] = "MLv"
    if "historical_actual" not in st.session_state: st.session_state["historical_actual"] = None
    if "prediction_result" not in st.session_state: st.session_state["prediction_result"] = None

    col_left, col_right = st.columns([1, 1], gap="large")

    # ------------------ KOLOM KIRI (INPUT & PETA) ------------------
    with col_left:
        st.markdown("### 📍 Parameter Lokasi Gempa")
        
        # Baris Aksi Cepat: Acak Data Historis & Preset Kota
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🎲 Acak Data Historis", use_container_width=True, help="Ambil 1 kejadian gempa nyata dari 500 katalog historis BMKG (1998-2023)"):
                samples = ml_data.get("historical_samples", [])
                if samples:
                    picked = random.choice(samples)
                    st.session_state["input_lat"] = float(picked.get("Latitude", -6.84))
                    st.session_state["input_lon"] = float(picked.get("Longitude", 107.05))
                    st.session_state["input_depth"] = float(picked.get("Depth (km)", 10.0))
                    st.session_state["year"] = int(picked.get("Tahun", 2022))
                    st.session_state["month"] = int(picked.get("Bulan", 11))
                    st.session_state["hour"] = int(picked.get("Jam", 6))
                    st.session_state["source"] = str(picked.get("Source", "NC"))
                    st.session_state["mag_type"] = str(picked.get("Mag Type", "MLv"))
                    st.session_state["historical_actual"] = {
                        "mag": picked.get("Magnitude"),
                        "cat": picked.get("Kategori_Magnitudo"),
                        "year": picked.get("Tahun"),
                        "id": picked.get("Event ID", "HISTORIS")
                    }
                    st.session_state["run_auto"] = True
                    st.rerun()

        with col_btn2:
            city_names = ["-- Pilih Kota / Skenario --"] + [c["name"] for c in INDONESIA_CITIES]
            selected_city = st.selectbox(
                "Pilih Wilayah Referensi",
                options=city_names,
                index=0,
                label_visibility="collapsed"
            )
            if selected_city != "-- Pilih Kota / Skenario --":
                chosen = next((c for c in INDONESIA_CITIES if c["name"] == selected_city), None)
                if chosen:
                    st.session_state["input_lat"] = float(chosen["lat"])
                    st.session_state["input_lon"] = float(chosen["lon"])
                    st.session_state["input_depth"] = float(chosen["depth"])
                    st.session_state["source"] = chosen["source"]
                    st.session_state["mag_type"] = chosen["mag_type"]
                    st.session_state["historical_actual"] = None
                    st.rerun()

        # Kotak Pencarian Nama Kota/Kecamatan Bebas (Nominatim OpenStreetMap)
        col_search1, col_search2 = st.columns([2.8, 1.2])
        with col_search1:
            search_query = st.text_input(
                "Atau ketik nama kota/kecamatan bebas di Indonesia (contoh: Salatiga, Garut, Banyuwangi)",
                placeholder="Ketik nama kota/kecamatan (contoh: Salatiga, Garut, Banyuwangi)...",
                label_visibility="collapsed",
                key="nominatim_search_input"
            )
        with col_search2:
            btn_search = st.button("🔍 Cari Lokasi", use_container_width=True)

        if btn_search and search_query.strip():
            try:
                headers = {"User-Agent": "GeoSeismikBMKG/1.0"}
                query_clean = search_query.strip()
                url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(query_clean)}, Indonesia&format=json&limit=1"
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200 and len(resp.json()) > 0:
                    geo = resp.json()[0]
                    st.session_state["input_lat"] = round(float(geo["lat"]), 3)
                    st.session_state["input_lon"] = round(float(geo["lon"]), 3)
                    st.session_state["historical_actual"] = None
                    st.success(f"📍 Ditemukan: {geo.get('display_name', query_clean)}")
                    st.rerun()
                else:
                    st.warning("⚠️ Wilayah tidak ditemukan. Coba ketik nama kota/kabupaten dengan ejaan yang lebih spesifik.")
            except Exception as e:
                st.warning(f"⚠️ Gagal mencari lokasi (periksa koneksi internet): {e}")

        # Input Koordinat Terikat Key
        c_lat, c_lon = st.columns(2)
        with c_lat:
            val_lat = st.number_input(
                "Lintang / Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                step=0.01,
                format="%.3f",
                key="input_lat",
                help="Wilayah Indonesia: -15.5° s.d. 8.5°"
            )

        with c_lon:
            val_lon = st.number_input(
                "Bujur / Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                step=0.01,
                format="%.3f",
                key="input_lon",
                help="Wilayah Indonesia: 90.0° s.d. 145.0°"
            )

        # Slider Kedalaman Terikat Key
        val_depth = st.slider(
            "Kedalaman / Depth (km)",
            min_value=1.0,
            max_value=700.0,
            step=1.0,
            key="input_depth"
        )
        
        # Badge Keterangan Kedalaman
        if val_depth <= 30:
            depth_cat = "🟢 Kerak Sangat Dangkal (≤ 30 km) — Konsentrasi Deformasi Kerak Atas"
        elif val_depth <= 70:
            depth_cat = "🟡 Kerak Dangkal (30–70 km) — Transisi Patahan Lempeng Kontak Antar Muka"
        elif val_depth <= 300:
            depth_cat = "🟠 Menengah / Intermediate (70–300 km) — Penunjaman Subduksi Aktif"
        else:
            depth_cat = "🔵 Zona Subduksi Dalam (> 300 km) — Mantel Bumi / Wadati-Benioff Zone"
        st.caption(depth_cat)

        # Expander Parameter Teknis BMKG (Opsional)
        with st.expander("⚙️ Parameter Teknis BMKG (Opsional)", expanded=False):
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.session_state["year"] = st.number_input("Tahun Kejadian", min_value=1998, max_value=2030, value=int(st.session_state["year"]))
                st.session_state["month"] = st.number_input("Bulan (1–12)", min_value=1, max_value=12, value=int(st.session_state["month"]))
                st.session_state["hour"] = st.number_input("Jam UTC (0–23)", min_value=0, max_value=23, value=int(st.session_state["hour"]))
            with exp_col2:
                sources = ["NC", "RC_01", "RC_02", "RC_03", "RC_04", "RC_05", "RC_06", "RC_07", "RC_08", "RC_09", "RC_10"]
                st.session_state["source"] = st.selectbox("Stasiun / Source", options=sources, index=sources.index(st.session_state["source"]) if st.session_state["source"] in sources else 0)
                mag_types = ["MLv", "mb", "Mw(mB)", "Mw", "M", "ML", "-"]
                st.session_state["mag_type"] = st.selectbox("Tipe Magnitudo (Mag Type)", options=mag_types, index=mag_types.index(st.session_state["mag_type"]) if st.session_state["mag_type"] in mag_types else 0)

        # Validasi Batas Wilayah Indonesia
        is_valid_indo = (-15.5 <= val_lat <= 8.5) and (90.0 <= val_lon <= 145.0)
        if not is_valid_indo:
            st.error("⚠️ Koordinat di luar batas wilayah pemantauan BMKG Indonesia (-15.5° s.d. 8.5° Lintang, 90.0° s.d. 145.0° Bujur). Silakan sesuaikan.")

        # Peta Folium Interaktif (OpenStreetMap + 2 Radius Spasial Episenter: 50 km & 150 km)
        st.markdown("**Peta Episenter & Radius Referensi Spasial (50 km & 150 km):**")
        
        m = folium.Map(
            location=[val_lat, val_lon],
            zoom_start=6 if is_valid_indo else 4,
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
        )
        
        # Radius 1: Jarak Spasial Dekat (50 km)
        folium.Circle(
            location=[val_lat, val_lon],
            radius=50000,
            color="#3B82F6",
            weight=1.8,
            fill=True,
            fill_color="#93C5FD",
            fill_opacity=0.18,
            tooltip="Radius Spasial 50 km dari Episenter"
        ).add_to(m)
        
        # Radius 2: Jarak Spasial Menengah (150 km)
        folium.Circle(
            location=[val_lat, val_lon],
            radius=150000,
            color="#64748B",
            weight=1.5,
            fill=True,
            fill_color="#CBD5E1",
            fill_opacity=0.12,
            tooltip="Radius Spasial 150 km dari Episenter"
        ).add_to(m)
        
        # Marker Episenter
        folium.Marker(
            location=[val_lat, val_lon],
            popup=f"<b>Episenter Gempa Uji</b><br>Lat: {val_lat:.3f}°<br>Lon: {val_lon:.3f}°<br>Kedalaman: {val_depth} km",
            tooltip="Titik Episenter Uji",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
        ).add_to(m)
        
        st_folium(m, height=280, use_container_width=True, key=f"map_{round(val_lat,3)}_{round(val_lon,3)}")

        # Tombol Eksekusi Analisis
        btn_analyze = st.button(
            "⚡ Analisis Zona & Klasifikasi",
            type="primary",
            use_container_width=True,
            disabled=not is_valid_indo
        )

    # ------------------ LOGIKA INFERENSI ML ------------------
    should_run = btn_analyze or st.session_state.get("run_auto", False)
    if should_run and is_valid_indo:
        st.session_state["run_auto"] = False
        with st.spinner("Menghitung klaster zona K-Means, evaluasi densitas DBSCAN & klasifikasi Random Forest..."):
            # 1. Klaster Spasial K-Means (k=4)
            df_in = pd.DataFrame([{
                'Latitude': val_lat,
                'Longitude': val_lon,
                'Depth (km)': val_depth,
                'Tahun': int(st.session_state["year"]),
                'Bulan': int(st.session_state["month"]),
                'Jam': int(st.session_state["hour"]),
                'Source': st.session_state["source"],
                'Mag Type': st.session_state["mag_type"]
            }])
            
            coords_scaled = ml_data['scaler_cluster'].transform(df_in[['Latitude', 'Longitude', 'Depth (km)']])
            cid = int(ml_data['kmeans'].predict(coords_scaled)[0])
            cname = ml_data['cluster_mapping'].get(cid, f"Klaster {cid}")
            df_in['Cluster_KMeans'] = cid
            
            # 2. Evaluasi Kepadatan Spasial DBSCAN
            dbscan_info = evaluate_dbscan_density(val_lat, val_lon, val_depth, eps=0.40)
            
            # 3. Klasifikasi Random Forest
            feats = ml_data['num_cols'] + ml_data['cat_cols']
            pred_class = str(ml_data['rf_pipeline'].predict(df_in[feats])[0])
            probas = ml_data['rf_pipeline'].predict_proba(df_in[feats])[0]
            
            classes = list(ml_data['rf_pipeline'].classes_)
            proba_dict = {cls: round(float(p) * 100.0, 1) for cls, p in zip(classes, probas)}
            
            city_near, dist_near = find_nearest_city(val_lat, val_lon)
            wib_now = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S WIB")
            
            # Simpan hasil ke session state
            res_obj = {
                "waktu_wib": wib_now,
                "lat": val_lat,
                "lon": val_lon,
                "depth": val_depth,
                "cluster_id": cid,
                "cluster_name": cname,
                "dbscan": dbscan_info,
                "category": pred_class,
                "confidence": proba_dict.get(pred_class, 0.0),
                "probabilities": proba_dict,
                "nearest_city": city_near,
                "distance_km": dist_near
            }
            st.session_state["prediction_result"] = res_obj
            
            # Simpan ke MongoDB Atlas / Session State
            save_simulation({
                "waktu_wib": wib_now,
                "latitude": val_lat,
                "longitude": val_lon,
                "depth_km": val_depth,
                "zona_kmeans": cname,
                "status_dbscan": dbscan_info["status"],
                "kategori_prediksi": pred_class,
                "confidence_pct": proba_dict.get(pred_class, 0.0),
                "kota_terdekat": f"{city_near} ({dist_near} km)"
            })

    # ------------------ KOLOM KANAN (HASIL OUTPUT RISET) ------------------
    with col_right:
        st.markdown("### 📊 Hasil Analisis Model Riset Seismik")
        
        res = st.session_state.get("prediction_result")
        
        if res is None:
            # KONDISI STANDBY
            st.markdown("""
            <div class="card-box" style="text-align: center; padding: 48px 24px; min-height: 480px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="font-size: 54px; margin-bottom: 12px;">📡</div>
                <h3 style="margin: 0 0 8px 0; color: #0F172A; font-weight: 800;">Menunggu Analisis Seismik</h3>
                <p style="color: #64748B; font-size: 14px; max-width: 380px; margin: 0 auto 20px auto; line-height: 1.5;">
                    Silakan tentukan parameter koordinat di kolom kiri atau tekan tombol <b>"🎲 Acak Data Historis"</b>, lalu tekan <b>"Analisis Zona & Klasifikasi"</b>.
                </p>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; text-align: left; max-width: 440px; font-size: 12px; color: #475569;">
                    <b>Pipeline Evaluasi Riset:</b><br>
                    • <b>K-Means (k=4):</b> Segmentasi 4 Klaster Zona Seismik Tektonik Indonesia<br>
                    • <b>DBSCAN:</b> Evaluasi Kepadatan Spasial (Core Cluster vs Outlier Spasial)<br>
                    • <b>Random Forest:</b> Klasifikasi 5 Kategori Magnitudo (Mikro s.d. Kuat)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # KONDISI ACTIVE
            # 1. Badge Zona Seismik & Pembanding Historis
            c_badge1, c_badge2 = st.columns([1.2, 1])
            with c_badge1:
                st.markdown(f"""
                <div style="background: #E2E8F0; padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 700; color: #1E293B; display: inline-block;">
                    🗺️ {res['cluster_name']}
                </div>
                """, unsafe_allow_html=True)
            
            with c_badge2:
                hist = st.session_state.get("historical_actual")
                if hist:
                    st.markdown(f"""
                    <div style="background: #EFF6FF; border: 1px solid #BFDBFE; color: #1D4ED8; padding: 4px 10px; border-radius: 8px; font-size: 11px; font-weight: 700; text-align: right;">
                        🎯 Aktual BMKG ({hist['year']}): <b>{hist['mag']} SR</b> ({hist['cat']})
                    </div>
                    """, unsafe_allow_html=True)

            # 2. Kartu Utama Prediksi Kategori Magnitudo
            cat_color_map = {
                'Mikro (<3.0)': {'bg': '#ECFDF5', 'border': '#A7F3D0', 'text': '#065F46', 'badge': '#10B981'},
                'Minor (3.0-3.9)': {'bg': '#F0F9FF', 'border': '#BAE6FD', 'text': '#0369A1', 'badge': '#0EA5E9'},
                'Ringan (4.0-4.9)': {'bg': '#FFFBEB', 'border': '#FDE68A', 'text': '#92400E', 'badge': '#F59E0B'},
                'Sedang (5.0-5.9)': {'bg': '#FFF7ED', 'border': '#FED7AA', 'text': '#9A3412', 'badge': '#F97316'},
                'Kuat (>=6.0)': {'bg': '#FFF1F2', 'border': '#FECDD3', 'text': '#9F1239', 'badge': '#EF4444'}
            }
            c_style = cat_color_map.get(res['category'], cat_color_map['Sedang (5.0-5.9)'])
            
            st.markdown(f"""
            <div style="background: {c_style['bg']}; border: 1.5px solid {c_style['border']}; border-radius: 14px; padding: 20px; margin: 12px 0 16px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="background: {c_style['badge']}; color: white; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 800; text-transform: uppercase;">
                        Prediksi Kategori Random Forest
                    </span>
                    <span style="font-size: 24px; font-weight: 800; color: {c_style['text']};">
                        {res['confidence']}%
                    </span>
                </div>
                <div style="font-size: 24px; font-weight: 800; color: {c_style['text']}; letter-spacing: -0.5px;">
                    {res['category']}
                </div>
                <p style="margin: 6px 0 0 0; font-size: 12px; color: {c_style['text']}; opacity: 0.9;">
                    Titik episenter dekat <b>{res['nearest_city']}</b> ({res['distance_km']} km) • Kedalaman hiposenter {res['depth']} km
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 3. Dua Kartu Analisis Model Riset (K-Means & DBSCAN)
            c_km, c_db = st.columns(2)
            prof = CLUSTER_PROFILES.get(res['cluster_id'], CLUSTER_PROFILES[1])
            
            with c_km:
                st.markdown(f"""
                <div class="impact-card" style="background: #F8FAFC; border: 1.5px solid #E2E8F0; min-height: 125px;">
                    <div style="font-size: 12px; font-weight: 800; color: #0F172A; margin-bottom: 4px;">
                        📌 Segmentasi K-Means (k=4)
                    </div>
                    <div style="font-size: 11px; color: #334155; margin-bottom: 3px;">
                        <b>Zona:</b> {prof['nama']}<br>
                        <b>Rerata Kedalaman:</b> {prof['kedalaman_rata']}
                    </div>
                    <div style="font-size: 11px; color: #64748B; line-height: 1.4;">
                        <b>Karakter Tektonik:</b> {prof['karakter']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with c_db:
                db = res['dbscan']
                st.markdown(f"""
                <div class="impact-card" style="background: {db['bg']}; border: 1.5px solid {db['border']}; min-height: 125px;">
                    <div style="font-size: 12px; font-weight: 800; color: {db['text']}; margin-bottom: 4px;">
                        {db['icon']} Densitas Spasial DBSCAN
                    </div>
                    <div style="font-size: 11px; color: {db['text']}; margin-bottom: 3px;">
                        <b>Status:</b> {db['status']}<br>
                        <b>Kepadatan:</b> {db['neighbors']} tetangga (jarak terdekat: {db['min_dist']})
                    </div>
                    <div style="font-size: 11px; color: {db['text']}; line-height: 1.4; opacity: 0.95;">
                        {db['desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 4. Distribusi Probabilitas 5 Kategori Magnitudo
            st.markdown("**Distribusi Probabilitas Model Random Forest:**")
            ordered_cats = ['Mikro (<3.0)', 'Minor (3.0-3.9)', 'Ringan (4.0-4.9)', 'Sedang (5.0-5.9)', 'Kuat (>=6.0)']
            
            for cat in ordered_cats:
                p_val = res['probabilities'].get(cat, 0.0)
                is_selected = (cat == res['category'])
                
                c_lbl, c_val = st.columns([3, 1])
                with c_lbl:
                    st.write(f"**{cat}** {'⭐ (Pilihan AI)' if is_selected else ''}")
                with c_val:
                    st.write(f"**{p_val:.1f}%**")
                st.progress(float(p_val / 100.0))

    # ------------------ BAGIAN BAWAH: LOG RIWAYAT SIMULASI ------------------
    st.markdown("---")
    c_hist_title, c_hist_status = st.columns([3, 1])
    with c_hist_title:
        st.subheader("📜 10 Riwayat Uji Coba Simulasi Terbaru")
    
    recent_records, is_mongo = get_recent_history(limit=10)
    with c_hist_status:
        if is_mongo:
            st.success("🟢 Terhubung ke MongoDB Atlas")
        else:
            st.info("🔵 Sesi Lokal (Atur MONGO_URI untuk Cloud)")
            
    if recent_records:
        df_hist = pd.DataFrame(recent_records)
        cols_display = [c for c in ["waktu_wib", "latitude", "longitude", "depth_km", "zona_kmeans", "status_dbscan", "kategori_prediksi", "confidence_pct", "kota_terdekat"] if c in df_hist.columns]
        df_show = df_hist[cols_display].rename(columns={
            "waktu_wib": "Waktu (WIB)",
            "latitude": "Lintang",
            "longitude": "Bujur",
            "depth_km": "Kedalaman (km)",
            "zona_kmeans": "Zona Klaster K-Means",
            "status_dbscan": "Kepadatan Spasial DBSCAN",
            "kategori_prediksi": "Prediksi Magnitudo",
            "confidence_pct": "Keyakinan AI (%)",
            "kota_terdekat": "Kota Terdekat"
        })
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada riwayat simulasi pada sesi ini. Jalankan analisis untuk merekam log.")

# =============================================================
# TAB 2: ANALISIS BATCH CSV
# =============================================================
with tab_batch:
    st.markdown("### 📂 Analisis Gempa Massal (Batch CSV)")
    st.write("Unggah berkas CSV katalog gempa bumi untuk klasifikasi zona seismik K-Means, evaluasi kepadatan spasial DBSCAN, dan prediksi kategori magnitudo Random Forest secara serentak.")
    
    if os.path.exists("template_gempa_bmkg.csv"):
        with open("template_gempa_bmkg.csv", "rb") as fp:
            st.download_button(
                "📥 Unduh Template CSV Contoh BMKG",
                data=fp,
                file_name="template_gempa_bmkg.csv",
                mime="text/csv"
            )
            
    uploaded_file = st.file_uploader("Pilih Berkas CSV", type=["csv"], help="Mendukung kolom Latitude, Longitude, Depth (km), serta format katalog BMKG.")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Berkas berhasil dimuat: **{len(df_batch)} baris data**.")
            
            for art in ['Unnamed: 2', 'No']:
                if art in df_batch.columns: df_batch = df_batch.drop(columns=[art])
                
            if 'Date' in df_batch.columns and ('Tahun' not in df_batch.columns or 'Bulan' not in df_batch.columns):
                date_dt = pd.to_datetime(df_batch['Date'], errors='coerce')
                if 'Tahun' not in df_batch.columns: df_batch['Tahun'] = date_dt.dt.year.fillna(2023).astype(int)
                if 'Bulan' not in df_batch.columns: df_batch['Bulan'] = date_dt.dt.month.fillna(1).astype(int)
                
            if 'Time (UTC)' in df_batch.columns and 'Jam' not in df_batch.columns:
                time_dt = pd.to_datetime(df_batch['Time (UTC)'], format='%H:%M:%S', errors='coerce')
                df_batch['Jam'] = time_dt.dt.hour.fillna(0).astype(int)
                
            if 'Tahun' not in df_batch.columns: df_batch['Tahun'] = 2023
            if 'Bulan' not in df_batch.columns: df_batch['Bulan'] = 6
            if 'Jam' not in df_batch.columns: df_batch['Jam'] = 12
            if 'Source' not in df_batch.columns: df_batch['Source'] = "NC"
            if 'Mag Type' not in df_batch.columns: df_batch['Mag Type'] = "MLv"
            
            missing_cols = [c for c in ['Latitude', 'Longitude', 'Depth (km)'] if c not in df_batch.columns]
            if missing_cols:
                st.error(f"Kolom wajib tidak ditemukan: {missing_cols}. Mohon sesuaikan dengan template.")
            else:
                with st.spinner("Memproses seluruh data menggunakan model ML..."):
                    # 1. K-Means
                    coords = ml_data['scaler_cluster'].transform(df_batch[['Latitude', 'Longitude', 'Depth (km)']])
                    cids = ml_data['kmeans'].predict(coords)
                    df_batch['Cluster_KMeans'] = cids
                    df_batch['Zona_KMeans'] = [ml_data['cluster_mapping'].get(cid, f"Klaster {cid}") for cid in cids]
                    
                    # 2. Random Forest
                    feats = ml_data['num_cols'] + ml_data['cat_cols']
                    preds = ml_data['rf_pipeline'].predict(df_batch[feats])
                    probas = ml_data['rf_pipeline'].predict_proba(df_batch[feats])
                    df_batch['Prediksi_Kategori'] = preds
                    df_batch['Keyakinan_AI (%)'] = np.round(np.max(probas, axis=1) * 100.0, 1)
                    
                    # 3. DBSCAN & Kota Terdekat
                    db_list, city_list = [], []
                    for _, r in df_batch.iterrows():
                        db_res = evaluate_dbscan_density(float(r['Latitude']), float(r['Longitude']), float(r['Depth (km)']))
                        c_near, d_near = find_nearest_city(float(r['Latitude']), float(r['Longitude']))
                        db_list.append(db_res['status'])
                        city_list.append(f"{c_near} ({d_near} km)")
                        
                    df_batch['Kepadatan_DBSCAN'] = db_list
                    df_batch['Kota_Terdekat'] = city_list
                    
                # Ringkasan KPI
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Total Kejadian Gempa", f"{len(df_batch):,}")
                kpi2.metric("Zona K-Means Terbanyak", df_batch['Zona_KMeans'].value_counts().index[0] if len(df_batch)>0 else "-")
                kpi3.metric("Kategori Dominan", df_batch['Prediksi_Kategori'].value_counts().index[0] if len(df_batch)>0 else "-")
                n_core = df_batch['Kepadatan_DBSCAN'].str.contains("Core").sum()
                kpi4.metric("Kejadian Zona Inti (Core)", f"{n_core:,}", f"{(n_core/len(df_batch)*100.0):.1f}%")
                
                # Peta Sebaran Spasial Gempa
                st.markdown("**Peta Sebaran Spasial Gempa Hasil Batch:**")
                st.map(df_batch[['Latitude', 'Longitude']].rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}))
                
                # Tabel Data & Ekspor CSV
                st.markdown("**Tabel Rincian Hasil Klasifikasi:**")
                cols_export = ['Latitude', 'Longitude', 'Depth (km)', 'Zona_KMeans', 'Kepadatan_DBSCAN', 'Prediksi_Kategori', 'Keyakinan_AI (%)', 'Kota_Terdekat']
                st.dataframe(df_batch[cols_export], use_container_width=True)
                
                csv_export = df_batch[cols_export].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Unduh Seluruh Hasil Klasifikasi (.CSV)",
                    data=csv_export,
                    file_name=f"GeoSeismik_Batch_Hasil_{int(time.time())}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Gagal memproses berkas CSV: {str(e)}")

# =============================================================
# TAB 3: PERFORMA MODEL (RINGKASAN RISET GEOFISIKA)
# =============================================================
with tab_performa:
    st.markdown("### 📈 Dashboard Evaluasi & Performa Model Machine Learning")
    st.write("Hasil evaluasi ilmiah resmi berbasis skema **80:20 Stratified Train-Test Split (random_state=42)** persis seperti riset notebook `gempa.ipynb`.")
    
    # 4 Metrik Kartu Utama
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Bersih BMKG", "217.807", "1998–2024 (27 Tahun)")
    m2.metric("Rasio Ketimpangan Kelas", "230,5 : 1", "Mikro vs Kuat (0.20%)")
    m3.metric("Silhouette K-Means (k=4)", "0.4823", "vs DBSCAN 0.3147")
    m4.metric("Akurasi Random Forest", "68.11%", "Macro F1: 51.43%")
    
    st.markdown("---")
    
    # 1. Tabel Perbandingan 3 Model
    st.markdown("#### 1. Perbandingan Performa 3 Model Klasifikasi")
    comp_data = ml_data.get("eval_model_comparison", [])
    if comp_data:
        df_comp = pd.DataFrame(comp_data)
        for col in ['Accuracy', 'Macro F1-Score', 'Weighted F1-Score', 'Rata-rata CV (Macro F1)']:
            if col in df_comp.columns:
                df_comp[col] = df_comp[col].apply(lambda x: f"{float(x)*100.0:.2f}%" if pd.notnull(x) else "-")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    # 2. Metrik per Kelas & Matriks Kebingungan (Confusion Matrix)
    c_m1, c_m2 = st.columns([1.1, 1])
    with c_m1:
        st.markdown("#### 2. Metrik per Kelas Random Forest (43.562 Data Uji)")
        class_metrics = ml_data.get("eval_metrics_per_class", [])
        if class_metrics:
            df_class = pd.DataFrame(class_metrics)
            for col in ['Precision', 'Recall', 'F1-Score']:
                if col in df_class.columns:
                    df_class[col] = df_class[col].apply(lambda x: f"{float(x)*100.0:.2f}%")
            st.dataframe(df_class, use_container_width=True, hide_index=True)
            
    with c_m2:
        st.markdown("#### 3. Matriks Kebingungan (Confusion Matrix 5x5)")
        cm = ml_data.get("eval_confusion_matrix", [])
        labels = ml_data.get("eval_cm_classes", [])
        if cm and labels:
            short_labels = [l.split(' ')[0] for l in labels]
            df_cm = pd.DataFrame(cm, index=[f"Akt: {s}" for s in short_labels], columns=[f"Pred: {s}" for s in short_labels])
            st.dataframe(df_cm, use_container_width=True)

    st.markdown("---")
    
    # 3. Feature Importance Top 10
    st.markdown("#### 4. Top 10 Feature Importance (Tingkat Kepentingan Fitur)")
    feat_imp = ml_data.get("eval_top_10_features", {})
    if feat_imp:
        df_feats = pd.DataFrame(list(feat_imp.items()), columns=["Fitur", "Importance"]).sort_values(by="Importance", ascending=True)
        df_feats["Importance (%)"] = df_feats["Importance"] * 100.0
        st.bar_chart(df_feats.set_index("Fitur")["Importance (%)"])

# =============================================================
# TAB 4: PANDUAN PENGGUNAAN & KAMUS DATA
# =============================================================
with tab_panduan:
    st.markdown("### 📖 Panduan Penggunaan Sistem & Kamus Data Seismik")
    
    # Alur 3 Langkah
    st.markdown("#### 1. Alur Penggunaan Aplikasi")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("""
        <div class="card-box">
            <h4 style="margin: 0 0 6px 0; color: #0F172A;">1️⃣ Input Parameter Spasial</h4>
            <p style="font-size: 12px; color: #64748B; margin: 0; line-height: 1.5;">
                Tentukan koordinat lintang, bujur, dan kedalaman hiposenter menggunakan tombol <b>"🎲 Acak Data Historis"</b>, preset kota, atau pencarian nama wilayah.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with step2:
        st.markdown("""
        <div class="card-box">
            <h4 style="margin: 0 0 6px 0; color: #0F172A;">2️⃣ Klasterisasi (K-Means & DBSCAN)</h4>
            <p style="font-size: 12px; color: #64748B; margin: 0; line-height: 1.5;">
                Model K-Means (k=4) memetakan titik ke dalam salah satu dari 4 zona tektonik Indonesia, sedangkan DBSCAN mengevaluasi kerapatan densitas spasial (zona inti vs outlier).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with step3:
        st.markdown("""
        <div class="card-box">
            <h4 style="margin: 0 0 6px 0; color: #0F172A;">3️⃣ Klasifikasi Random Forest</h4>
            <p style="font-size: 12px; color: #64748B; margin: 0; line-height: 1.5;">
                Pipeline Random Forest Classifier memprediksi kategori spektrum kekuatan magnitudo (Mikro &lt;3.0 s.d. Kuat &ge;6.0) beserta persentase distribusi probabilitas kelasnya.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Kamus Data 10 Variabel BMKG
    st.markdown("#### 2. Kamus Data (10 Variabel Katalog Gempa BMKG)")
    dict_data = [
        {"Variabel": "Event ID", "Tipe": "String", "Rentang Nilai": "BMKG-YYYYMMDD...", "Deskripsi": "Pengidentifikasi unik kejadian gempa bumi BMKG."},
        {"Variabel": "Date", "Tipe": "Date", "Rentang Nilai": "1998 s.d. 2024", "Deskripsi": "Tanggal kejadian gempa bumi (diekstrak menjadi Tahun dan Bulan)."},
        {"Variabel": "Time (UTC)", "Tipe": "Time", "Rentang Nilai": "00:00:00 s.d. 23:59:59", "Deskripsi": "Waktu kejadian standar Universal Time Coordinated (diekstrak menjadi Jam)."},
        {"Variabel": "Latitude", "Tipe": "Float", "Rentang Nilai": "-14.96° s.d. 7.00°", "Deskripsi": "Garis lintang episenter gempa bumi wilayah Indonesia."},
        {"Variabel": "Longitude", "Tipe": "Float", "Rentang Nilai": "94.00° s.d. 142.00°", "Deskripsi": "Garis bujur episenter gempa bumi wilayah Indonesia."},
        {"Variabel": "Depth (km)", "Tipe": "Float", "Rentang Nilai": "1 s.d. 798 km", "Deskripsi": "Kedalaman hiposenter dari permukaan bumi (fitur utama klaster & getaran)."},
        {"Variabel": "Magnitude", "Tipe": "Float", "Rentang Nilai": "0.2 s.d. 7.9 SR", "Deskripsi": "Kekuatan gempa (dikelompokkan menjadi target 5 kelas klasifikasi)."},
        {"Variabel": "Mag Type", "Tipe": "Categorical", "Rentang Nilai": "MLv, mb, Mw, Mw(mB), M", "Deskripsi": "Tipe skala magnitudo seismograf yang digunakan oleh stasiun pencatat."},
        {"Variabel": "Source", "Tipe": "Categorical", "Rentang Nilai": "NC, RC_01 s.d. RC_10", "Deskripsi": "Unit pemrosesan BMKG: Pusat Nasional (NC) atau Balai Regional (RC)."},
        {"Variabel": "Source Event ID", "Tipe": "String", "Rentang Nilai": "Kode Stasiun", "Deskripsi": "ID kejadian stasiun regional sebelum rekonsiliasi pusat (dihapus)."}
    ]
    st.dataframe(pd.DataFrame(dict_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # Etika Kebencanaan & Ruang Lingkup Riset
    st.warning("""
    **⚖️ Etika Kebencanaan & Batasan Ilmiah Pemodelan:**
    1. **Ruang Lingkup Murni Riset Machine Learning:** Aplikasi ini murni memfokuskan pada pemodelan klasterisasi spasial (K-Means & DBSCAN) dan klasifikasi probabilitas spektrum magnitudo (Random Forest) berdasarkan data tabular spasial katalog BMKG. Sistem ini tidak bertindak sebagai sistem peringatan dini tsunami realtime.
    2. **Bukan Alat Peramal Waktu Gempa:** Model tidak meramalkan waktu terjadinya gempa di masa depan, melainkan mengklasifikasikan kategori magnitudo dari parameter hiposenter yang diberikan.
    3. **Anti Data Leakage:** Nilai numerik `Magnitude` murni menjadi variabel target evaluasi dan tidak pernah dilibatkan sebagai fitur prediktor ($X$).
    """)

# -------------------------------------------------------------
# 8. FOOTER
# -------------------------------------------------------------
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 12px; margin-top: 36px; padding: 20px 0; border-top: 1px solid #E2E8F0;">
    GeoSeismik BMKG © 2024–2026 • Klasterisasi Zona Seismik & Klasifikasi AI BMKG<br>
    Didukung oleh <b>Python, Streamlit, Scikit-Learn, Folium & MongoDB Atlas</b>
</div>
""", unsafe_allow_html=True)
