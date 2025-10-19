import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image

#Load model di luar fungsi
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')

#Panggil model hanya sekali
model = load_model()

# Fungsi prediksi
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    input_array = np.expand_dims(input_array, axis=0)
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    return result_index

# Informasi penyakit dan rekomendasi
disease_info = {
    "Corynespora": {
        "deskripsi": "Penyakit ini disebabkan oleh jamur *Corynespora cassiicola*, yang menyebabkan bercak coklat tidak teratur dan bercah hitam pada tulang daun pada daun.",
        "penanganan": [
            "Melakukan pemupukan nitrogen dengan intensitas tinggi (dua kali dosis anjuran) pada saat daun-daun baru mulai terbentuk.",
            "Melakukan penghembusan serbuk belerang seminggu sekali selama lima minggu.",
            "Aplikasikan fungisida berbahan aktif mancozeb '0,25%' dengan dosis 400-600 l/ha melalui penyemprotan di siang hari, atau lakukan fogging dengan minyak mineral pada malam hari.",
            "Aplikasikan juga fungisida berbahan aktif klorotalonil '0,2%' dengan dosis 0,75 kg/ha.",
           
        ]
    },
    "Healthy": {
        "deskripsi": "Daun tampak hijau dan sehat tanpa tanda infeksi jamur atau kerusakan jaringan.",
        "penanganan": [
            "Pertahankan pola perawatan yang baik.",
            "Pastikan tanaman mendapat cahaya matahari cukup dan tidak tergenang air.",
            "Lakukan pemantauan rutin untuk deteksi dini penyakit."
        ]
    },
    "Oidium": {
        "deskripsi": "Disebabkan oleh jamur *Oidium heveae*, dikenal sebagai embun tepung. Ditandai dengan lapisan putih pada permukaan daun.",
        "penanganan": [
            "Pemberian pupuk nitrogen satu kali dosis anjuran",
            "Lakukan upaya pencegahan dengan memberikan pengabutan belerang 6–7 kg/ha dengan interval 3–7 hari di malam hari.",
            "Gunakan fungisida berbahan aktif triadimefon '0,25%' dengan dosis 600 l/ha dan interval 7–10 hari dengan cara fogging, atau lakukan penyemprotan fungisida berbahan aktif mancozeb di siang hari.",
        ]
    },
    "Pestalotiopsis": {
        "deskripsi": "Disebabkan oleh jamur *Pestalotiopsis spp.*, menyebabkan bercak daun dengan tepi berwarna gelap dan pusat keabu-abuan.",
        "penanganan": [
            "Tambahkan pupuk ekstra 25% N dan K untuk membantu tanaman karet tumbuh dengan baik.",
            "Perhatikan beban penyadapan agar sesuai dengan kemampuan klon.",
            "Gunakan fungisida berbahan aktif heksakonazol, propikonazol, atau thiophanate methyl dengan cara fogging atau spraying."
        ]
    }
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Beranda", "Deteksi Penyakit", "Tentang"])

# Halaman Beranda
if app_mode == "Beranda":
    st.header("DETEKSI PENYAKIT KARET")
    image_path = "homepage.png"
    st.image(image_path, width=700)
    st.markdown("""
    Selamat datang di website kami!
    
    Sistem Deteksi Penyakit Tanaman Karet adalah sebuah platform berbasis web yang dirancang untuk membantu mengidentifikasi penyakit yang umum menyerang pohon karet, seperti **oidium**, **pestalotiopsis**, dan **corynespora**.  
    Dengan memanfaatkan teknologi **machine learning**, sistem ini mampu menganalisis gambar pohon karet yang Anda unggah secara cepat dan akurat untuk mendeteksi kemungkinan penyakit.
        """)
    with st.expander("Proses Singkat"):
        st.markdown("""
        1. Unggah gambar pohon karet melalui halaman *Deteksi Penyakit*.  
        2. Sistem memproses gambar dan menganalisis kemungkinan penyakit.  
        3. Sistem menampilkan hasil analisis.
            """
        )

    st.markdown("""
        Platform ini dirancang dengan antarmuka yang intuitif dan ramah pengguna — hasil analisis disajikan secara cepat (dalam hitungan detik) sehingga dapat membantu petani dan peneliti mengambil keputusan yang lebih tepat terkait kesehatan tanaman karet.
        """
        )
    st.caption("Gunakan menu di sidebar untuk masuk ke halaman Deteksi Penyakit atau melihat informasi lebih lengkap pada halaman Tentang.")


# Halaman Deteksi Penyakit Karet 
elif app_mode == "Deteksi Penyakit":
    st.header("Deteksi Penyakit Karet")
    st.markdown("#### Keterangan Hasil Prediksi")
    st.markdown("""
    - **Healthy** = Daun karet sehat  
    - **Pestalotiopsis** = Daun karet terjangkit jamur *Pestalotiopsis*  
    - **Oidium** = Daun karet terjangkit jamur *Oidium* (embun tepung)  
    - **Corynespora** = Daun karet terjangkit jamur *Corynespora*  
    """)

    test_image = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_container_width=True)

        if st.button("Prediksi"):
            with st.spinner("Tolong tunggu sebentar..."):
                st.balloons()
                st.write("Prediksi Kami: ")
                result_index = model_prediction(test_image)
                #Define Class
                class_name = ['Corynespora', 'Healthy', 'Oidium', 'Pestalotiopsis']
                predicted_class = class_name[result_index]
                st.success(f"Model memprediksi bahwa daun tersebut **{predicted_class}**")
                info = disease_info[predicted_class]
                st.subheader("🩺 Deskripsi Penyakit")
                st.write(info["deskripsi"])
                st.subheader("🌿 Rekomendasi Penanganan")
                for i, langkah in enumerate(info["penanganan"], start=1):
                    st.markdown(f"{i}. {langkah}")



#Halaman Tentang
elif app_mode == "Tentang":
    st.title("Tentang Sistem")
    st.markdown("""
    Sebuah platform berbasis web yang dirancang untuk membantu mengidentifikasi penyakit yang umum menyerang pohon karet, seperti *oidium*, *pestalotiopsis*, dan *corynespora*. Dengan memanfaatkan teknologi *machine learning*, sistem ini mampu menganalisis gambar pohon karet yang Anda unggah secara cepat dan akurat untuk mendeteksi kemungkinan penyakit.
                
    **Proses:**
    Pengguna hanya perlu mengunggah gambar pohon karet melalui halaman deteksi penyakit, lalu sistem akan memproses gambar tersebut dan menampilkan hasil analisis beserta rekomendasi langkah selanjutnya. Dengan antarmuka yang intuitif, ramah pengguna, serta kecepatan analisis dalam hitungan detik, platform ini diharapkan dapat mendukung petani maupun peneliti dalam mengambil keputusan yang lebih tepat terkait kesehatan tanaman karet.
    
    **Dikembangkan oleh:**  
    - Maritza Aulia  
    - Bening Cahyaningati  
    - M. Farhan Shadiq
        """
    )
    st.write("---")
