"""
Generador de Texto con LSTM
Aplicacion Streamlit - Curso Agentes de IA e Interfaces Multimodales
"""

import streamlit as st
import numpy as np
import json
import time
import os

st.set_page_config(
    page_title="💖 Generador LSTM 💖",
    page_icon="💅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌸 ESTILO GIRLY 🌸
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#ffe6f0 0%,#ffd6ec 50%,#ffe0f5 100%);}
.main-title{font-size:2.3rem;font-weight:800;background:linear-gradient(90deg,#ff4da6,#ff99cc,#ff66b2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;text-shadow:0px 0px 10px rgba(255,105,180,0.4);}
.subtitle{color:#ff66b2;font-size:1rem;text-align:center;}
.generated-text{background:linear-gradient(135deg,#fff0f6 0%,#ffd6ec 100%);border-radius:15px;padding:1.5rem;font-family:Georgia,serif;font-size:1.05rem;line-height:1.8;color:#b30059;border-left:5px solid #ff66b2;min-height:120px;box-shadow:0 0 15px rgba(255,105,180,0.2);}
.info-box{background:#fff0f6;border-radius:10px;padding:1rem;border:1px solid #ffb3d9;font-size:0.9rem;color:#b30059;}
.stButton>button{background:linear-gradient(90deg,#ff66b2,#ff99cc);color:white;border-radius:15px;border:none;font-weight:bold;box-shadow:0 4px 10px rgba(255,105,180,0.3);}
.stButton>button:hover{background:linear-gradient(90deg,#ff3385,#ff80bf);transform:scale(1.03);}
textarea,input{border-radius:10px!important;border:1px solid #ffb3d9!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#ffe6f2,#ffd6ec);}
.stTabs [data-baseweb="tab"]{color:#ff66b2;font-weight:bold;}
.stTabs [aria-selected="true"]{background-color:#ffd6ec!important;border-radius:10px;}
</style>
""", unsafe_allow_html=True)


# ── Funciones ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["idx_to_char"] = {int(k): v for k, v in metadata["idx_to_char"].items()}
        return model, metadata, None
    except Exception as e:
        return None, None, str(e)


def is_embedding_model(model):
    first = model.layers[0]
    return hasattr(first, "input_dim") or first.__class__.__name__ == "Embedding"


def sample_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-10) / temperature
    preds = np.exp(preds - np.max(preds))
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(1, preds, 1))


def prepare_input(window, char_to_idx, vocab_size, use_embedding):
    indices = [char_to_idx.get(c, 0) for c in window]
    if use_embedding:
        return np.array([indices], dtype=np.int32)
    else:
        x = np.array(indices, dtype=np.float32) / float(vocab_size)
        return x.reshape(1, len(window), 1)


def generate_full_text(model, seed_text, char_to_idx, idx_to_char,
                        seq_length, vocab_size, n_chars=200, temperature=0.8):
    use_emb = is_embedding_model(model)
    seed_text = seed_text.lower()
    if len(seed_text) < seq_length:
        seed_text = seed_text.rjust(seq_length)
    seed_text = seed_text[-seq_length:]
    seed_text = "".join(c if c in char_to_idx else " " for c in seed_text)

    generated = ""
    window = list(seed_text)

    for _ in range(n_chars):
        x = prepare_input(window[-seq_length:], char_to_idx, vocab_size, use_emb)
        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample_temperature(preds, temperature)]
        generated += next_char
        window.append(next_char)

    return generated


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💖 Configuración")
    st.markdown("---")
    st.markdown("### 🌸 Cargar Modelo")

    model_file    = st.file_uploader("📦 Modelo (.keras o .h5)", type=["keras", "h5"])
    metadata_file = st.file_uploader("🧾 Metadatos (.json)",    type=["json"])

    st.markdown("---")
    st.markdown("### ✨ Parámetros")

    temperature = st.slider("🔥 Temperatura", 0.1, 2.0, 0.8, 0.05)

    if temperature < 0.5:
        st.caption("💙 Frío")
    elif temperature < 1.0:
        st.caption("💗 Balanceado")
    elif temperature < 1.4:
        st.caption("💖 Creativo")
    else:
        st.caption("💥 Caótico")

    n_chars = st.slider("📏 Longitud del texto", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("### 🌈 Semillas")

    seeds = [
        "en un lugar de la mancha",
        "el caballero miro al horizonte",
        "sancho panza respondio",
        "con estas razones perdia",
        "el hidalgo tomo la espada",
    ]
    selected_seed = st.selectbox("✨ Elige una semilla:", ["(personalizada)"] + seeds)

    st.markdown("---")
    st.markdown("""<div class="info-box">
    💖 Modelo entrenado con el Quijote ✨
    </div>""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">💖 Generador de Texto LSTM 💖</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">✨ IA pero make it cute ✨</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["💅 Generar", "🌸 Comparar", "✨ Teoría"])

# (TODO tu código sigue EXACTAMENTE igual debajo 👇)
