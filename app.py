# app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import re
from scipy.stats import randint
import plotly.graph_objects as go
from PIL import Image, UnidentifiedImageError
import os
import time
# import base64
# import streamlit.components.v1 as components

# ---------------------------------------------
# 1. Load & Prepare Data + Model (Offline)
# ---------------------------------------------
bmi_df = pd.read_csv('bmi_dataset.csv')
nutri_df = pd.read_csv('dataset_nutrients.csv')

# Rename kolom bmi dataset agar seragam
bmi_df.rename(columns={
    'Weight (kg)': 'WeightKg',
    'Height (m)': 'HeightM'
}, inplace=True)

# Hitung BMI jika belum ada
if 'BMI' not in bmi_df.columns:
    bmi_df['BMI'] = bmi_df['WeightKg'] / (bmi_df['HeightM'] ** 2)

# Label encoding gender
le = "Models/label_encoder.pkl"

# Klasifikasi BMI → status gizi
def klasifikasi_bmi(bmi):
    if bmi < 18.5:   return 'Underweight'
    if bmi < 25.0:   return 'Normal'
    if bmi < 30.0:   return 'Overweight'
    return 'Obesity'
bmi_df['WeightStatus'] = bmi_df['BMI'].apply(klasifikasi_bmi)

# Persiapkan fitur & target
features = ['Gender', 'Age', 'HeightM', 'WeightKg']
X = bmi_df[features]
y = bmi_df['WeightStatus']

# Standardisasi
scaler = joblib.load('Models/scaler.pkl')

# Tuning Random Forest (RandomizedSearchCV)
best_rf = joblib.load('Models/random_forest_model.pkl')

# Siapkan kolom Menu di nutri_df


def extract_menu_name(filename):
    """
    Ekstrak nama menu bersih dari filename:
    1) Buang ekstensi (.jpg/.jpeg/.png)
    2) Hapus segmen terakhir (setelah hyphen terakhir) — biasanya hash atau angka acak
    3) Hapus semua angka yang tersisa
    4) Ganti '-' dan '_' menjadi spasi; rapikan spasi ganda
    """
    # Abaikan file generik
    if filename.startswith("recipe-image-legacy-id"):
        return None

    # 1) Hapus ekstensi gambar
    name = re.sub(r'\.(jpe?g|png)$', '', filename, flags=re.IGNORECASE)

    # 2) Hapus segmen terakhir (setelah hyphen terakhir)
    #    Contoh: "cod-cucumber-avocado-mango-salsa-salad-517846e" -> "cod-cucumber-avocado-mango-salsa-salad"
    name = re.sub(r'-[^-]+$', '', name)

    # 3) Hapus semua digit yang tersisa di seluruh string
    name = re.sub(r'\d+', '', name)

    # 4) Ganti '-' dan '_' menjadi spasi
    name = name.replace('-', ' ').replace('_', ' ')

    # Rapikan spasi ganda dan trim
    name = re.sub(r'\s+', ' ', name).strip()

    return name

nutri_df['Menu'] = nutri_df['image'].apply(extract_menu_name)
nutri_df.dropna(subset=['Menu'], inplace=True)
nutri_df.drop_duplicates(subset=['Menu'], inplace=True)

# ===============================
# Tabel AKG berdasar Permenkes 2019
# ===============================
akg_df = pd.DataFrame([
    # Baris untuk laki-laki (Male)
    {"Gender": "Male", "AgeMin": 13, "AgeMax": 15, "Energy": 2400, "Protein": 70, "Fat": 80, "Carbs": 350, "Fibre": 34},
    {"Gender": "Male", "AgeMin": 16, "AgeMax": 18, "Energy": 2650, "Protein": 75, "Fat": 85, "Carbs": 400, "Fibre": 37},
    {"Gender": "Male", "AgeMin": 19, "AgeMax": 29, "Energy": 2650, "Protein": 65, "Fat": 75, "Carbs": 430, "Fibre": 37},
    {"Gender": "Male", "AgeMin": 30, "AgeMax": 49, "Energy": 2550, "Protein": 65, "Fat": 70, "Carbs": 415, "Fibre": 36},
    {"Gender": "Male", "AgeMin": 50, "AgeMax": 64, "Energy": 2150, "Protein": 65, "Fat": 60, "Carbs": 340, "Fibre": 30},

    # Baris untuk perempuan (Female)
    {"Gender": "Female", "AgeMin": 13, "AgeMax": 15, "Energy": 2050, "Protein": 65, "Fat": 70, "Carbs": 300, "Fibre": 29},
    {"Gender": "Female", "AgeMin": 16, "AgeMax": 18, "Energy": 2100, "Protein": 65, "Fat": 70, "Carbs": 300, "Fibre": 29},
    {"Gender": "Female", "AgeMin": 19, "AgeMax": 29, "Energy": 2250, "Protein": 60, "Fat": 65, "Carbs": 360, "Fibre": 32},
    {"Gender": "Female", "AgeMin": 30, "AgeMax": 49, "Energy": 2150, "Protein": 60, "Fat": 60, "Carbs": 340, "Fibre": 30},
    {"Gender": "Female", "AgeMin": 50, "AgeMax": 64, "Energy": 1800, "Protein": 60, "Fat": 50, "Carbs": 280, "Fibre": 25},
])

# ============================================
# Fungsi untuk mengambil AKG user dari tabel di atas
# ============================================
def get_user_akg(gender: str, age: int) -> dict:
    # Filter baris dari akg_df yang sesuai gender dan rentang umur
    row = akg_df[
        (akg_df['Gender'] == gender) &
        (akg_df['AgeMin'] <= age) &
        (akg_df['AgeMax'] >= age)
    ]
    # Jika cocok, kembalikan sebagai dict
    return row.iloc[0].to_dict() if not row.empty else None

# ==============================================================
# Fungsi utama untuk memberikan rekomendasi makanan berbasis demografi
# ==============================================================

def recommend_menu_demographic(nutri_df, status_gizi, gender, age, activity, top_n=10):
    # 1. Ambil nilai AKG user dari tabel berdasarkan gender dan umur
    akg = get_user_akg(gender, age)
    if akg is None:
        return pd.DataFrame(columns=['Menu', 'kcal', 'protein', 'fat', 'carbs', 'fibre'])

    # 2. Penyesuaian kalori berdasarkan status gizi
    kalori_target = akg['Energy']
    if status_gizi == 'Underweight':
        kalori_target += 500
    elif status_gizi in ['Overweight', 'Obesity']:
        kalori_target -= 500
    # Untuk status Normal → tidak diubah

    # 3. Hitung target makro menggunakan % IOM 2005 (tengah)
    karbo_target = (0.55 * kalori_target) / 4
    protein_target = (0.20 * kalori_target) / 4
    lemak_target = (0.25 * kalori_target) / 9

    # 4. FILTERING awal — buang menu terlalu rendah nutrisinya
    filt = nutri_df[
        (nutri_df['kcal'] > 50) &
        (nutri_df['protein'] > 1) &
        (nutri_df['fat'] > 1) &
        (nutri_df['carbs'] > 5)
    ].copy()

    # 5. SCORING nutrisi — skor tinggi jika lebih dekat ke target
    # contoh kasar: tol_karbo = 0.65-0.45 = 0.20, tol_protein = 0.35-0.10 = 0.25, tol_lemak = 0.35-0.20 = 0.15
    w_cals = 1/0.10   # misal toleransi kalori 10% dari target
    w_c   = 1/0.20
    w_p   = 1/0.25
    w_f   = 1/0.15
    # normalisasi agar jumlah=1
    s = w_cals + w_c + w_p + w_f
    w_cals, w_c, w_p, w_f = w_cals/s, w_c/s, w_p/s, w_f/s

    filt['score_raw'] = (
        w_cals*abs(filt['kcal'] - kalori_target)/kalori_target +
        w_p   *abs(filt['protein'] - protein_target)/protein_target +
        w_f   *abs(filt['fat'] - lemak_target)/lemak_target +
        w_c   *abs(filt['carbs'] - karbo_target)/karbo_target
    )
    
    # filt['score_raw'] = (
    #     abs(filt['kcal'] - kalori_target) * 0.4
    #     + abs(filt['protein'] - protein_target) * 0.2
    #     + abs(filt['fat'] - lemak_target) * 0.2
    #     + abs(filt['carbs'] - karbo_target) * 0.2
    # )
    
    # 5b. Konversi ke skor positif (semakin kecil selisih → makin besar skor)
    filt['score'] = filt['score_raw'].max() - filt['score_raw']

    # 6. SORTING & SAMPLING berbobot
    # Hanya ambil kandidat dengan skor positif
    top_candidates = filt[filt['score'] > 0].sort_values(by='score', ascending=False).drop_duplicates('Menu').head(100)


    # Ambil top_n menu secara acak dari 100 kandidat terbaik, berbasis skor sebagai bobot
    rekom = top_candidates.sample(n=min(top_n, len(top_candidates)), weights='score', replace=False)

    return rekom[['Menu', 'image', 'kcal', 'protein', 'fat', 'carbs', 'fibre']]

# Fungsi untuk estimasi waktu perubahan berat (evidence-based, 7700 kkal = 1 kg)
def estimasi_waktu_perubahan_berat(status, berat, berat_min, berat_max, tee, tee_min, tee_max):
    def hitung_estimasi(kg_target, kal_per_hari_min, kal_per_hari_max):
        kalori_minggu_min = kal_per_hari_min * 7
        kalori_minggu_max = kal_per_hari_max * 7
        minggu_min = (kg_target * 7700) / kalori_minggu_max
        minggu_max = (kg_target * 7700) / kalori_minggu_min
        return minggu_min, minggu_max

    if status == "Underweight":
        target_kg = berat_min - berat
        surplus_min = tee_min - tee
        surplus_max = tee_max - tee
        return hitung_estimasi(target_kg, surplus_min, surplus_max)

    elif status == "Overweight":
        target_kg = berat - berat_max
        defisit_min = tee - tee_max
        defisit_max = tee - tee_min
        return hitung_estimasi(target_kg, defisit_min, defisit_max)

    elif status == "Obesity":
        target_kg_min = berat * 0.05
        target_kg_max = berat * 0.10
        defisit_min = tee - tee_max
        defisit_max = tee - tee_min
        minggu_5 = hitung_estimasi(target_kg_min, defisit_min, defisit_max)
        minggu_10 = hitung_estimasi(target_kg_max, defisit_min, defisit_max)
        return minggu_5 + minggu_10

    else:
        return (0, 0)
    
# def encode_img_to_base64(img_path):
#     with open(img_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode()

# ---------------------------------------------
# 2. Page: Home
# ---------------------------------------------
# Inisialisasi menu hanya sekali
if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Home"

# Tambahkan custom CSS untuk tombol sidebar
st.markdown("""
<style>
/* Sidebar button */
section[data-testid="stSidebar"] button {
    border: 2px solid #088F8F !important;
    border-radius: 12px !important;
    background-color: transparent !important;
    color: black !important;
    padding: 0.5em 1em !important;
    margin-bottom: 10px !important;
    width: 100% !important;
    text-align: center !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    box-shadow: none !important;
    outline: none !important;
}
section[data-testid="stSidebar"] button:hover {
    background-color: #088F8F !important;
    color: white !important;
    border-color: #088F8F !important;
}
section[data-testid="stSidebar"] button.selected,
section[data-testid="stSidebar"] button.selected:hover {
    background-color: #088F8F !important;
    color: white !important;
    border-color: #088F8F !important;
}
/* Style tombol submit "Recommend Menu" */
button[data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    color: #088F8F !important;
    border: 2px solid #088F8F !important;
    border-radius: 12px !important;
    padding: 0.5em 1em !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    width: 100% !important;
    transition: all 0.3s ease-in-out !important;
}

/* Hover effect */
button[data-testid="baseButton-secondary"]:hover {
    background-color: #088F8F !important;
    color: white !important;
    border-color: #088F8F !important;
}
</style>
""", unsafe_allow_html=True)



st.set_page_config(
    page_title="EduNutri Recommender",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def render_sidebar_button(label, internal_label):
    clicked = st.sidebar.button(label, key=internal_label)
    if clicked:
        st.session_state.menu = internal_label

    # Tambahkan class 'selected' ke tombol aktif
    if st.session_state.menu == internal_label:
        st.markdown(f"""
            <script>
            const btn = window.parent.document.querySelector('button[key="{internal_label}"]');
            if (btn) {{
                btn.classList.add("selected");
            }}
            </script>
        """, unsafe_allow_html=True)

# Tampilkan logo di sidebar
# logo = Image.open("logo/Logo_Gundarma_University.png") 
# st.sidebar.image(logo, use_container_width=True)

# Tombol navigasi
st.sidebar.markdown("### Pilih Halaman:")
render_sidebar_button("Home", "🏠 Home", use_container_width=True)
render_sidebar_button("Recommendation", "📝 Rekomendasi Menu", use_container_width=True)
render_sidebar_button("Information", "📊 Resource", use_container_width=True)
menu = st.session_state.menu

if menu == "🏠 Home":
    # --- Hero / Welcome ---
    st.title("EduNutri")
    st.subheader("“Makan Cerdas, Hidup Sehat.”")
    st.markdown(
        """
        **Selamat datang di EduNutri!**  
        EduNutri adalah aplikasi edukasi gizi yang membantu Anda **memprediksi status gizi**
        dan **menyusun rekomendasi menu harian** yang sehat, seimbang, dan mudah diikuti.
        """
    )

    # Optional hero image (gunakan placeholder bila perlu)
    # st.image("assets/hero-food.jpg", use_container_width=True)

    st.markdown("---")

    # --- Tentang Aplikasi / Value props ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔎 Prediksi Status Gizi")
        st.write("Hitung BMI & prediksi status gizi untuk memahami kondisi awal Anda.")
    with col2:
        st.markdown("### 🥗 Rekomendasi Menu")
        st.write("Dapatkan saran menu harian yang menyeimbangkan kalori dan makronutrien.")
    with col3:
        st.markdown("### 📘 Edukasi Gizi")
        st.write("Pelajari kebutuhan energi, protein, lemak, karbohidrat, dan serat Anda.")

    st.markdown("---")

    # --- Cara Menggunakan ---
    st.markdown("## Cara Menggunakan")
    st.markdown(
        """
        1. Buka menu **“Recommendation”** lalu isi data: **usia, jenis kelamin, tinggi, berat,** dan **aktivitas**.  
        2. Klik **Proses** untuk melihat **status gizi**, **kebutuhan energi**, dan **kebutuhan makronutrien**.  
        3. Gulir ke bawah untuk melihat **rekomendasi menu harian** beserta **kalori, protein, lemak, karbohidrat, dan serat**.  
        4. Buka menu **“Information”** untuk ringkasan konsep gizi dan tips penerapan.  
        """
    )

    # --- Catatan & Tips ---
    st.warning(
        "ℹ️ **Catatan:** Rekomendasi bersifat edukasional dan **bukan** pengganti konsultasi dengan tenaga kesehatan. "
        "Jika memiliki kondisi medis khusus, sebaiknya berkonsultasi dengan ahli gizi atau dokter."
    )

    st.info(
        "💡 **Tips:**\n"
        "- Ukur berat & tinggi terbaru agar prediksi lebih akurat.\n"
        "- Pilih tingkat aktivitas harian yang paling mendekati rutinitas Anda.\n"
        "- Gunakan rekomendasi sebagai panduan—sesuaikan dengan preferensi dan ketersediaan bahan."
    )

    

elif menu == "📝 Rekomendasi Menu" :
    
    st.title("EduNutri: Menu Recommender")
    
    # Input Form
    col1, spacer1, spacer2, col2 = st.columns([2, 0.2, 0.2, 3.6])
    
    # Mulai pengukuran waktu
    start_time = time.perf_counter()
    # Input Form
    with col1:
        with st.form("input_form"):
            umur = st.number_input("Age (15-59 years)", min_value=15, max_value=59, value=None, placeholder="e.g. 25")
            jenis_kelamin = st.radio("Gender", options=["Female", "Male"])
            tinggi = st.number_input("Height (m)", min_value=1.5, max_value=2.0, value=None, step=0.01, placeholder="e.g. 1.70")
            berat = st.number_input("Weight (kg)", min_value=40, max_value=130, value=None, placeholder="e.g. 65")
            activity = st.selectbox("Physical Activity Level", options=[
                "Sedentary (little to no activity)",
                "Lightly Active (1–2 times/week)",
                "Moderately Active (3–5 times/week)",
                "Very Active (6–7 times/week)",
                "Extremely Active (twice daily or intense)"
            ])

            submitted = st.form_submit_button("Recommend Menu")
            # st.markdown("<br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
            # Custom tombol submit
            # col_submit = st.columns(3)[1]  # Tengah
            # with col_submit:
            #     submitted = st.form_submit_button("Recommend Menu")

    if submitted:
        if None in (umur, tinggi, berat):
            st.warning("⚠️ Silakan isi semua data (umur, tinggi, dan berat) terlebih dahulu.")
        else:
            # Encode input
            g = 1 if jenis_kelamin=="Laki-laki" else 0
            input_vec = np.array([[g, umur, tinggi, berat]])
            input_scaled = scaler.transform(input_vec)

            # Predict status gizi
            status = best_rf.predict(input_scaled)[0]

            # 1) Tampilkan status & BMI
            bmi_user = berat / (tinggi ** 2)

            with col2:
                # 2) Gauge Chart untuk Kategori BMI
                # ————————————————
                # … setelah menghitung bmi_user …
                fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi_user,
                number={'suffix': " BMI"},
                gauge={
                    'axis': {
                        'range': [0, 40],
                        # Titik mid‑segment untuk tiap kategori
                        'tickmode': 'array',
                        'tickvals': [
                            (0 + 16) / 2,     #  8   → Underweight
                            (16 + 18.5) / 2,  # 17.25→ Mild Thinness
                            (18.5 + 25) / 2,  # 21.75→ Normal
                            (25 + 30) / 2,    # 27.5 → Overweight
                            (30 + 35) / 2,    # 32.5 → Obese I
                            (35 + 40) / 2     # 37.5 → Obese II/III
                        ],
                        'ticktext': [
                            "Underweight<br>(0–16)",
                            "Mild Thinness<br>(16–18.5)",
                            "Normal<br>(18.5–25)",
                            "Overweight<br>(25–30)",
                            "Obese I<br>(30–35)",
                            "Obese II<br>(35–40)"
                        ],
                        'tickfont': {'size': 12}
                    },
                    'bar': {'color': "black", 'thickness': 0.25},
                    'steps': [
                        {'range': [0,   16], 'color': "lightgray"},
                        {'range': [16, 18.5], 'color': "gray"},
                        {'range': [18.5,25],  'color': "green"},
                        {'range': [25, 30],   'color': "yellow"},
                        {'range': [30, 35],   'color': "orange"},
                        {'range': [35, 40],   'color': "crimson"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': bmi_user
                    }
                }
            ))
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=30)
                )
                
                # Tambahkan annotation di bawah gauge untuk memperjelas
                fig.add_annotation(
                    text=f"<b>{bmi_user:.1f}</b>",
                    x=0.5, y=0.25,
                    font=dict(size=24),
                    showarrow=False
                )
                st.plotly_chart(fig, use_container_width=True)
                # 3) Tampilkan kategori BMI user
                st.markdown("### Kategori BMI Anda")
                st.write(
                    "- **Underweight**: BMI < 18.5\n"
                    "- **Normal**: 18.5 – 24.9\n"
                    "- **Overweight**: 25 – 29.9\n"
                    "- **Obesity**: BMI ≥ 30\n"
                )
                st.markdown(f"**Anda berada di kategori: `{status}`, dengan BMI {bmi_user:.1f}.**")
            st.markdown("---")
            
            col_t1, col_t2 = st.columns([1.8, 2.2])
            # col_t1, col_t2 = st.columns([1.5, 2.5])
            
            with col_t1:
                # 2) Hitung BMR (Harris-Benedict dan Mifflin-St Jeor)
                if jenis_kelamin == "Male":
                    bmr_hb = 66.5 + (13.75 * berat) + (5 * tinggi * 100) - (6.8 * umur)
                    bmr_msj = (10 * berat) + (6.25 * tinggi * 100) - (5 * umur) + 5
                else:
                    bmr_hb = 655 + (9.6 * berat) + (1.8 * tinggi * 100) - (4.7 * umur)
                    bmr_msj = (10 * berat) + (6.25 * tinggi * 100) - (5 * umur) - 161
                
                # 3) Hitung TEE berdasarkan PAL
                pal_levels = {
                    "Sedentary (little to no activity)": 1.2,
                    "Lightly Active (1–2 times/week)": 1.375,
                    "Moderately Active (3–5 times/week)": 1.55,
                    "Very Active (6–7 times/week)": 1.725,
                    "Extremely Active (twice daily or intense)": 1.9
                }
                pal_value = pal_levels[activity]
                tee = bmr_msj * pal_value
                
                # 4) Surplus/Defisit Kalori
                if status == "Underweight":
                    tee_min, tee_max = tee + 500, tee + 1000

                elif status in ["Overweight", "Obesity"]:
                    defisit1 = tee - 1000
                    defisit2 = tee - 500
                    # Jika defisit terlalu dalam (melewati BMR), gunakan defisit ringan
                    if defisit1 < bmr_msj or defisit2 < bmr_msj:
                        tee_min = tee - 300
                        tee_max = tee - 200
                    else:
                        tee_min = defisit1
                        tee_max = defisit2
                else:  # Normal
                    tee_min = tee_max = tee
                # st.success(f"Rekomendasi kalori harian: {tee_min:.0f} – {tee_max:.0f} kcal")
                
                # 5) Rekomendasi AKG Permenkes 2019
                def get_akg(gender, age):
                    akg_data = {
                        'Male': {
                            (13, 15): [2400, 70, 80, 350, 34],
                            (16, 18): [2650, 75, 85, 400, 37],
                            (19, 29): [2650, 65, 75, 430, 37],
                            (30, 49): [2550, 65, 70, 415, 36],
                            (50, 64): [2150, 65, 60, 340, 30]
                        },
                        'Female': {
                            (13, 15): [2050, 65, 70, 300, 29],
                            (16, 18): [2100, 65, 70, 300, 29],
                            (19, 29): [2250, 60, 65, 360, 32],
                            (30, 49): [2150, 60, 60, 340, 30],
                            (50, 64): [1800, 60, 50, 280, 25]
                        }
                    }
                    for (min_age, max_age), values in akg_data[gender].items():
                        if min_age <= age <= max_age:
                            return values
                    return [None] * 5
                
                # 8) Kebutuhan zat gizi makro (IOM 2005)
                karbo_min = (0.45 * tee_min) / 4
                karbo_max = (0.65 * tee_max) / 4
                protein_min = (0.10 * tee_min) / 4
                protein_max = (0.30 * tee_max) / 4
                lemak_min = (0.20 * tee_min) / 9
                lemak_max = (0.30 * tee_max) / 9
                serat_min = 25
                serat_max = 37
                
                akg_values = get_akg(jenis_kelamin, umur)
                akg_kalori = akg_values[0]
                akg_dict = {
                    "Energy (kcal)": akg_values[0],
                    "Protein (g)": akg_values[1],
                    "Fat (g)": akg_values[2],
                    "Carbs (g)": akg_values[3],
                    "Fibre (g)": akg_values[4]
                }

                user_dict = {
                    "Energy (kcal)": f"{tee_min:.0f} – {tee_max:.0f}",
                    "Protein (g)": f"{protein_min:.0f} – {protein_max:.0f}",
                    "Fat (g)": f"{lemak_min:.0f} – {lemak_max:.0f}",
                    "Carbs (g)": f"{karbo_min:.0f} – {karbo_max:.0f}",
                    "Fibre (g)": f"{serat_min:.0f} – {serat_max:.0f}"
                }
                
                # 14) Perbandingan Kebutuhan Gizi vs AKG
                df_perbandingan = pd.DataFrame({
                    "AKG (Permenkes 2019)": akg_dict,
                    "Personal Needs": user_dict
                })
                st.markdown("### Perbandingan Kebutuhan Gizi Anda vs AKG")
                st.dataframe(df_perbandingan)
            
            with col_t2:
                # 10) Tabel Aktivitas
                st.markdown("### Tabel Kalori Berdasarkan Aktivitas")
                pal_table = pd.DataFrame({
                    "Activity Level": list(pal_levels.keys()),
                    "Calorie": [round(bmr_msj * v) for v in pal_levels.values()]
                })
                st.dataframe(pal_table, use_container_width=True)
            st.markdown("---")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                # 6) Rentang berat ideal berdasarkan BMI normal
                berat_min = 18.5 * (tinggi ** 2)
                berat_max = 24.9 * (tinggi ** 2)
                
                # 7) Estimasi minggu perubahan berat badan (Evidence-Based)
                if status == "Underweight":
                    minggu_min, minggu_max = estimasi_waktu_perubahan_berat(status, berat, berat_min, berat_max, tee, tee_min, tee_max)

                elif status == "Overweight":
                    minggu_min, minggu_max = estimasi_waktu_perubahan_berat(status, berat, berat_min, berat_max, tee, tee_min, tee_max)

                elif status == "Obesity":
                    minggu_5_min, minggu_5_max, minggu_10_min, minggu_10_max = estimasi_waktu_perubahan_berat(status, berat, berat_min, berat_max, tee, tee_min, tee_max)

                # Fungsi tambahan untuk tampilkan estimasi perubahan berat badan
                def tampilkan_estimasi_perubahan_berat(status, berat, berat_min, berat_max, tee_min, tee_max, bmr_msj):
                    if status == "Underweight":
                        kg_needed = berat_min - berat
                        minggu_min, minggu_max = kg_needed / 1.0, kg_needed / 0.5
                        return f"""
                        - Target Kenaikan Berat: **{kg_needed:.1f} kg**  
                        - **Kecepatan Kenaikan Berat: 0.5 – 1.0 kg/minggu**  
                        - Estimasi Waktu: **{minggu_min:.0f} – {minggu_max:.0f} minggu**  
                        - Kebutuhan Surplus Kalori Harian: **{tee_min:.0f} – {tee_max:.0f} kcal**
                        """
                    elif status == "Overweight":
                        kg_loss_target = berat - berat_max
                        minggu_min, minggu_max = kg_loss_target / 1.0, kg_loss_target / 0.5
                        return f"""
                        - Target penurunan berat: **{kg_loss_target:.1f} kg**
                        - Kecepatan penurunan: **0.5 – 1.0 kg/minggu**
                        - Estimasi waktu: **{minggu_min:.0f} – {minggu_max:.0f} minggu**
                        - Kebutuhan defisit kalori harian: **{tee_min:.0f} – {tee_max:.0f} kcal**
                                """
                    elif status == "Obesity":
                        target_5 = 0.05 * berat
                        target_10 = 0.10 * berat
                        min_5, max_5 = target_5 / 1.0, target_5 / 0.5
                        min_10, max_10 = target_10 / 1.0, target_10 / 0.5
                        if tee_max < bmr_msj:
                            tee_min_adj, tee_max_adj = bmr_msj - 300, bmr_msj - 200
                        else:
                            tee_min_adj, tee_max_adj = tee_min, tee_max
                        return f"""
                        - Target penurunan berat:
                            - 5% = **{target_5:.1f} kg** → **{min_5:.0f} – {max_5:.0f} minggu**
                            - 10% = **{target_10:.1f} kg** → **{min_10:.0f} – {max_10:.0f} minggu**
                        - Kebutuhan defisit kalori harian: **{tee_min_adj:.0f} – {tee_max_adj:.0f} kcal**
                                """
                    return None
                
                st.subheader(f"**Prediksi Status Gizi:** {status}")
                st.write(f"**BMI Anda:** {bmi_user:.1f}")
                
                # 12) Berat Ideal
                st.markdown("### Rentang Berat Badan Ideal")
                st.write(f"Rentang berat badan ideal berdasarkan tinggi Anda: {berat_min:.1f} – {berat_max:.1f} kg")
                
                if status == "Underweight":
                    selisih = berat_min - berat
                    st.info(f"Anda perlu menambah sekitar **{selisih:.1f} kg** untuk mencapai batas minimal berat normal.")
                elif status in ["Overweight", "Obesity"]:
                    selisih = berat - berat_max
                    st.info(f"Anda perlu mengurangi sekitar **{selisih:.1f} kg** untuk mencapai batas maksimal berat normal.")
                
                # 13) Estimasi Perubahan Berat
                if status != "Normal":
                    st.markdown("### Estimasi Perubahan Berat Badan")

                    if status == "Underweight":
                        kg_needed = berat_min - berat
                        st.info(f"""
                        - Target kenaikan berat: **{kg_needed:.1f} kg**
                        - Kecepatan: **0.5 – 1.0 kg/minggu**
                        - Estimasi waktu: **{minggu_min:.0f} – {minggu_max:.0f} minggu**
                        - Surplus kalori harian: **{tee_min - tee:.0f} – {tee_max - tee:.0f} kcal**
                        """)

                    elif status == "Overweight":
                        kg_loss = berat - berat_max
                        st.info(f"""
                        - Target Penurunan Berat: **{kg_loss:.1f} kg**  
                        - Kecepatan: **0.5–1.0 kg/minggu**  
                        - Estimasi Waktu: **{minggu_min:.0f} – {minggu_max:.0f} minggu**  
                        - Defisit Kalori Harian: **{tee - tee_max:.0f} – {tee - tee_min:.0f} kcal**
                        """)

                    elif status == "Obesity":
                        target_5 = berat * 0.05
                        target_10 = berat * 0.10
                        st.info(f"""
                        - **Target Penurunan Berat:**  
                            - 5% = **{target_5:.1f} kg** → **{minggu_5_min:.0f} – {minggu_5_max:.0f} minggu**  
                            - 10% = **{target_10:.1f} kg** → **{minggu_10_min:.0f} – {minggu_10_max:.0f} minggu**  
                        - Defisit Kalori Harian: **{tee - tee_max:.0f} – {tee - tee_min:.0f} kcal**
                        """)
                        st.warning("*Defisit disesuaikan agar tidak kurang dari BMR*")
        
            with col_info2:
                # 9) Tampilkan hasil ke Streamlit
                st.markdown("### Kebutuhan Energi Harian")
                st.write(f"- **BMR (Harris-Benedict)**: `{bmr_hb:.0f} kcal`")
                st.write(f"- **BMR (Mifflin-St Jeor)**: `{bmr_msj:.0f} kcal`")
                st.write(f"- **TEE (PAL x BMR Mifflin-St Jeor)**: `{tee:.0f} kcal`")
                st.write(f"- **AKG (Permenkes 2019)**: `{akg_dict['Energy (kcal)']} kcal`")

                if status == "Underweight":
                    st.info(f"💡 **Surplus Kalori:**\nKebutuhan kalori berkisar antara **{tee_min:.0f} – {tee_max:.0f} kcal**")
                elif status in ["Overweight", "Obesity"]:
                    if tee_min >= bmr_msj:
                        st.warning(f"⚠️ **Defisit Kalori:**\nKebutuhan kalori berkisar antara **{tee_min:.0f} – {tee_max:.0f} kcal** (tidak kurang dari BMR)")
                    else:
                        st.warning(f"⚠️ **Defisit Kalori Ringan:**\nKarena TEE terlalu rendah untuk defisit besar, digunakan defisit ringan **200–300 kcal** dari TEE.\nKebutuhan kalori Anda berkisar antara **{tee_min:.0f} – {tee_max:.0f} kcal**")
                else:
                    st.success(f"Kebutuhan kalori harian Anda: **{tee:.0f} kcal**")
                    
                # 11) Gizi makro
                st.markdown("### Kebutuhan Zat Gizi Makro Harian")
                st.write(f"- **Karbohidrat**: {karbo_min:.0f} – {karbo_max:.0f} gram")
                st.write(f"- **Protein**: {protein_min:.0f} – {protein_max:.0f} gram")
                st.write(f"- **Lemak**: {lemak_min:.0f} – {lemak_max:.0f} gram")
                st.write(f"- **Serat**: {serat_min:.0f} – {serat_max:.0f} gram")    
            st.markdown("---")
            
            # menu_rec = recommend_by_status(status, nutri_df, top_n=10)
            menu_rec = recommend_menu_demographic(nutri_df, status, jenis_kelamin, umur, activity, top_n=10)
            st.markdown("### Recommended Food Menu")
            
            # ——————————————————————————————
            # Heading untuk tiap kolom hasil rekomendasi
            # ——————————————————————————————
            col_img_h, col_menu_h, col_kcal_h, col_prot_h, col_fat_h, col_fib_h, col_carbs_h = st.columns([2, 3, 1.5, 1, 1, 1, 1])
            col_kcal_h.markdown("**Calories**")
            col_prot_h.markdown("**Protein**")
            col_fat_h.markdown("**Fat**")
            col_fib_h.markdown("**Fibre**")
            col_carbs_h.markdown("**Carbs**")

            # Loop menampilkan setiap item
            for idx, row in menu_rec.iterrows():
                col_img, col_menu, col_kcal, col_prot, col_fat, col_fib, col_carbs = st.columns([2, 3, 1.5, 1, 1, 1, 1])
                # Tampilkan gambar
                img_path = os.path.join('nutrients', 'images', row['image'])
                if os.path.exists(img_path):
                    try:
                        col_img.image(img_path, use_container_width=True)
                    except UnidentifiedImageError:
                        col_img.warning("⚠ Corrupted or unrecognizable image")
                else:
                    col_img.warning("⚠ Image not found")
                # Tampilkan nilai masing‑masing kolom
                col_menu.markdown(f"**{row['Menu']}**")
                col_kcal.markdown(f"{row['kcal']} kcal")
                col_prot.markdown(f"{row['protein']} g")
                col_fat.markdown(f"{row['fat']} g")
                col_fib.markdown(f"{row['fibre']} g")
                col_carbs.markdown(f"{row['carbs']} g")

            # Selesai pengukuran waktu
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # Tampilkan ke pengguna
            st.success(f"Waktu pemrosesan: {elapsed_time:.3f} detik")
elif menu == "📊 Resource":
    st.title("Information — EduNutri")

    # ==============================
    # 1) Definisi Gizi
    # ==============================
    st.header("Definisi Gizi")
    st.markdown(
        """
        **Gizi** adalah zat dalam makanan yang dibutuhkan tubuh untuk menunjang pertumbuhan, memperbaiki jaringan,
        dan menjaga fungsi vital. **Nutrisi** menekankan proses tubuh mencerna, menyerap, dan memanfaatkan zat tersebut.

        Gizi baik menunjang kesehatan, produktivitas, dan pencegahan penyakit, sedangkan gizi buruk dapat menurunkan
        kualitas hidup dan meningkatkan risiko penyakit.
        """
    )

    st.subheader("Gizi pada Remaja (13–18 tahun)")
    st.markdown(
        """
        - Masa **growth spurt** (lonjakan pertumbuhan tinggi/berat badan) → kebutuhan energi & protein sangat tinggi.  
        - Tantangan: pola makan tidak teratur, fast food, diet ekstrem, anemia defisiensi besi.  
        - Fokus: cukup energi, protein berkualitas, kalsium, serat, dan pola makan seimbang.
        """
    )

    st.subheader("Gizi pada Dewasa (≥19 tahun)")
    st.markdown(
        """
        - Fokus utama: **pemeliharaan** fungsi tubuh & pencegahan penyakit.  
        - Kebutuhan energi dipengaruhi usia, jenis kelamin, aktivitas, dan kondisi khusus (hamil/menyusui).  
        - Jaga makronutrisi seimbang, batasi lemak jenuh & trans.
        """
    )

    st.markdown("---")

    # ==============================
    # 2) Makronutrisi
    # ==============================
    st.header("Makronutrisi")
    st.markdown(
        """
        - **Karbohidrat**: Sumber energi utama (4 kkal/gram). Pilih karbohidrat kompleks dan berserat.  
        - **Protein**: Pembentuk & perbaikan jaringan, enzim, hormon, antibodi (4 kkal/gram).  
        - **Lemak**: Energi padat (9 kkal/gram), membantu penyerapan vitamin A, D, E, K.  
        - **Serat**: Tidak menghasilkan energi, penting untuk pencernaan dan kontrol gula darah.

        **Anjuran serat**: ≥14 g per 1000 kkal (~25–35 g/hari pada dewasa).
        """
    )

    st.markdown("---")

    # ==============================
    # 3) Kebutuhan & Kecukupan Gizi
    # ==============================
    st.header("Kebutuhan & Kecukupan Gizi")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Remaja (13–18 th)**  
            - Karbohidrat: **45–65%**  
            - Protein: **10–30%**  
            - Lemak total: **25–35%**  
            - Omega-3: **0,6–1,2%**, Omega-6: **5–10%**
            """
        )
    with col2:
        st.markdown(
            """
            **Dewasa (≥19 th)**  
            - Karbohidrat: **45–65%**  
            - Protein: **10–35%**  
            - Lemak total: **20–35%**  
            - Omega-3: **0,6–1,2%**, Omega-6: **5–10%**
            """
        )
    st.caption("Catatan: Lemak jenuh <10% energi dan lemak trans <1% dari total energi harian.")

    st.markdown("---")

    # ==============================
    # 4) Tabel AKG Permenkes 2019
    # ==============================
    st.subheader("Tabel AKG Remaja & Dewasa (Permenkes 2019)")
    st.markdown("**Remaja (13–18 tahun)**")
    akg_remaja = pd.DataFrame({
        "Kelompok": ["Laki-laki 13–15 th", "Laki-laki 16–18 th",
                    "Perempuan 13–15 th", "Perempuan 16–18 th"],
        "Energi (kkal)": [2400, 2650, 2050, 2100],
        "Protein (g)": [70, 75, 65, 65],
        "Lemak (g)": [80, 85, 70, 70],
        "Karbohidrat (g)": [350, 400, 300, 300],
        "Serat (g)": [34, 37, 29, 29]
    })
    st.dataframe(akg_remaja, use_container_width=True)

    st.markdown("**Dewasa (19–64 tahun)**")
    akg_dewasa = pd.DataFrame({
        "Kelompok": ["Laki-laki 19–29 th", "Laki-laki 30–49 th", "Laki-laki 50–64 th",
                    "Perempuan 19–29 th", "Perempuan 30–49 th", "Perempuan 50–64 th"],
        "Energi (kkal)": [2650, 2550, 2150, 2250, 2150, 1800],
        "Protein (g)": [65, 65, 65, 60, 60, 60],
        "Lemak (g)": [75, 70, 60, 65, 60, 50],
        "Karbohidrat (g)": [430, 415, 340, 360, 340, 280],
        "Serat (g)": [37, 36, 30, 32, 30, 25]
    })
    st.dataframe(akg_dewasa, use_container_width=True)

    st.markdown("---")

    # ==============================
    # 5) Rumus BMR & Kategori PAL
    # ==============================
    st.subheader("Rumus BMR (Mifflin–St Jeor)")
    st.latex(r'''
    \text{BMR Pria} = (10 \times \text{BB}) + (6.25 \times \text{TB}) - (5 \times \text{Usia}) + 5
    ''')
    st.latex(r'''
    \text{BMR Wanita} = (10 \times \text{BB}) + (6.25 \times \text{TB}) - (5 \times \text{Usia}) - 161
    ''')
    st.caption("Keterangan: BB = berat badan (kg), TB = tinggi badan (cm), Usia = umur (tahun)")

    st.subheader("Kategori Physical Activity Level (PAL)")
    pal_df = pd.DataFrame({
        "Kategori": [
            "Sangat jarang olahraga",
            "Jarang (1–3x/minggu)",
            "Sedang (3–5x/minggu)",
            "Sering (6–7x/minggu)",
            "Sangat sering (2x/hari)"
        ],
        "PAL": [1.2, 1.375, 1.55, 1.725, 1.9]
    })
    st.dataframe(pal_df, use_container_width=True)

    st.markdown("---")

    # ==============================
    # 6) Contoh Soal Cerita
    # ==============================
    st.header("Contoh Kasus Perhitungan")
    st.markdown(
        """
        **Soal:**  
        Andi, laki-laki berusia 25 tahun, memiliki berat badan 70 kg dan tinggi 175 cm.  
        Ia berolahraga 3–5 kali per minggu (PAL 1.55). Tentukan kebutuhan makronutrien hariannya
        jika proporsi yang digunakan adalah **Karbohidrat 50%**, **Protein 20%**, dan **Lemak 30%**.

        **Langkah Penyelesaian:**

        1. **Hitung BMR (Mifflin–St Jeor)**  
        BMR = (10 × 70) + (6.25 × 175) − (5 × 25) + 5  
        BMR = 700 + 1093.75 − 125 + 5 = **1673.75 kkal**

        2. **Hitung TDEE**  
           TDEE = BMR × PAL = 1673.75 × 1.55 = **2594.31 kkal**

        3. **Hitung kebutuhan makro**  
           - Karbo: (50% × 2594.31) / 4 = **324 g**  
           - Protein: (20% × 2594.31) / 4 = **130 g**  
           - Lemak: (30% × 2594.31) / 9 ≈ **86 g**  

        **Jawaban:**  
        Kebutuhan harian Andi ≈ **324 g karbohidrat**, **130 g protein**, **86 g lemak**,
        ditambah serat ± 30–35 g.
        """
    )
    
    st.markdown("---")

    # ==============================
    # 7) FAQ
    # ==============================
    st.header("FAQ (Frequently Asked Questions)")

    with st.expander("1. Apakah rekomendasi di EduNutri bisa menggantikan konsultasi dengan ahli gizi?"):
        st.write(
            "Tidak. Rekomendasi di EduNutri bersifat edukasional dan bertujuan sebagai panduan umum. "
            "Untuk kebutuhan spesifik atau kondisi medis tertentu, tetap disarankan berkonsultasi "
            "dengan ahli gizi atau tenaga kesehatan."
        )

    with st.expander("2. Apakah data yang saya masukkan disimpan?"):
        st.write(
            "Tidak. Semua data yang dimasukkan hanya diproses secara lokal saat aplikasi berjalan "
            "dan tidak disimpan di server."
        )

    with st.expander("3. Mengapa hasil perhitungan kebutuhan kalori saya berbeda dari aplikasi lain?"):
        st.write(
            "Perbedaan bisa terjadi karena metode perhitungan yang digunakan berbeda. "
            "EduNutri menggunakan rumus **Mifflin–St Jeor** yang terbukti akurat untuk populasi modern."
        )

    with st.expander("4. Apakah rekomendasi menu mempertimbangkan alergi makanan?"):
        st.write(
            "Versi saat ini belum mendukung filter alergi secara otomatis. "
            "Pengguna disarankan menyesuaikan menu yang direkomendasikan jika memiliki alergi tertentu."
        )

    with st.expander("5. Bagaimana jika saya ingin menurunkan berat badan?"):
        st.write(
            "Anda dapat mengatur defisit kalori dengan mengurangi asupan harian sekitar 500–1000 kkal "
            "dari kebutuhan total (TDEE), yang secara umum dapat menurunkan 0,5–1 kg per minggu. "
            "Pastikan tetap memenuhi kebutuhan protein dan mikronutrien."
        )


st.markdown("---")
st.caption("© 2025 EduNutri by Fajar Agus | Universitas Gunadarma")
