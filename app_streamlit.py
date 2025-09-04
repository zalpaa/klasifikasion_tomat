import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title = "Klasifikasi Tomat",
    page_icon = "🍅"
)

model = joblib.load("model_klasifikasi_tomat.joblib")
scaler = joblib.load("scaler_klasifikasi_tomat.joblib")

st.title("🍅 Klasifikasi Tomat")
st.markdown("Aplikasi machine learning untuk klasifikasi tomat termasuk kategori *Ekspor, Lokal Premium, atau Industri*")

berat = st.slider("Berat Tomat", 50, 210, 80)
kekenyalan = st.slider("Tingkat Kekenyalan", 2.0, 10.0, 4.2)
kadar_gula = st.slider("Kadar Gula", 1.0, 10.0, 5.3)
tebal_kulit = st.slider("Tebal Kulit", 0.1, 1.0, 0.7)

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame([[berat,kekenyalan,kadar_gula,tebal_kulit]],columns=["berat","kekenyalan","kadar_gula","tebal_kulit"])

    data_baru_scaled = scaler.transform(data_baru)
    prediksi = model.predict(data_baru_scaled)[0]
    presentase = max(model.predict_proba(data_baru_scaled)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan keyakinan **{presentase*100:.2f}%**")
    st.balloons()
    st.snow()

st.divider()
st.caption("Dibuat dengan 🍅 oleh **Zalpa Nur Arpandi**")