import streamlit as st
import pickle
import os
import time
from src.text_processor import preprocess_text
from src.search_engine import query_to_vector, vector_search, rocchio_update

st.set_page_config(page_title="IR Sistem - Pretraživač video igara", layout="wide")

@st.cache_data
def load_index():
    path = os.path.join("data", "index_data.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

data = load_index()

if not data:
    st.error("Indeks nije pronađen. Pokrenite 'python offline_indexer.py' za generisanje baze.")
    st.stop()

inverted_index = data["inverted_index"]
idf = data["idf"]
doc_vectors = data["doc_vectors"]
doc_lengths = data["doc_lengths"]
games_meta = data["games_meta"]

# Inicijalizacija radnog stanja sesije
if "query_vector" not in st.session_state:
    st.session_state.query_vector = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "expanded_terms" not in st.session_state:
    st.session_state.expanded_terms = []
if "search_time" not in st.session_state:
    st.session_state.search_time = 0.0

st.title("Pretraživač video igara")
st.caption("Implementacija Vektorskog modela (TF-IDF) sa Rocchio algoritmom povratne sprege")

# Bočna traka sa parametrima
with st.sidebar:
    st.subheader("Podešavanja pretrage")
    # Slajder za broj rezultata (od 5 do 30, po defaultu 10)
    top_k = st.slider("Broj rezultata za prikaz", min_value=5, max_value=30, value=10, step=5)
    
    st.markdown("---")
    st.subheader("Parametri Rocchio algoritma")
    alpha = st.slider("Alfa (težina originalnog upita)", 0.0, 2.0, 1.0, 0.1)
    beta = st.slider("Beta (težina relevantnih dokumenata)", 0.0, 2.0, 0.8, 0.1)
    
    st.markdown("---")
    if st.button("Nova pretraga", use_container_width=True):
        st.session_state.query_vector = None
        st.session_state.search_results = []
        st.session_state.expanded_terms = []
        st.session_state.search_time = 0.0
        st.rerun()

# Glavni panel za pretragu
col_query, col_btn = st.columns([5, 1])
with col_query:
    user_query = st.text_input("Upit:", placeholder="Unesite ključne reči...", label_visibility="collapsed")
with col_btn:
    search_clicked = st.button("Pretraži", use_container_width=True)

if search_clicked and user_query:
    start_time = time.time()
    tokens = preprocess_text(user_query)
    st.session_state.query_vector = query_to_vector(tokens, idf)
    st.session_state.search_results = vector_search(
        st.session_state.query_vector, inverted_index, idf, doc_lengths, top_k=top_k
    )
    st.session_state.search_time = (time.time() - start_time) * 1000
    st.session_state.expanded_terms = []

# Prikaz statusa i modifikovanih termina
if st.session_state.search_results:
    st.caption(f"Vreme izvršavanja: **{st.session_state.search_time:.2f} ms** | Prikazano: **{len(st.session_state.search_results)}** igara")

if st.session_state.expanded_terms:
    st.info(f"Modifikovan vektor upita. Uvedeni termini: **{', '.join(st.session_state.expanded_terms)}**")

# Prikaz liste rezultata
if st.session_state.search_results:
    st.subheader("Rezultati")
    selected_docs = []

    for doc_id, score in st.session_state.search_results:
        meta = games_meta[doc_id]
        with st.container():
            c_check, c_body = st.columns([0.05, 0.95])
            with c_check:
                is_selected = st.checkbox("Rel", key=f"check_{doc_id}", label_visibility="collapsed")
                if is_selected:
                    selected_docs.append(doc_id)
            with c_body:
                st.markdown(f"**{meta['title']}** — `Skor: {score:.4f}`")
                st.write(meta['desc'])
            st.divider()

    # Dugme za pokretanje povratne sprege
    if st.button("Primeni povratnu spregu (Rocchio)", type="primary"):
        if not selected_docs:
            st.warning("Označite bar jedan dokument kao relevantan.")
        else:
            start_time = time.time()
            old_terms = set(st.session_state.query_vector.keys())
            st.session_state.query_vector = rocchio_update(
                st.session_state.query_vector, selected_docs, doc_vectors, alpha, beta
            )
            new_terms = set(st.session_state.query_vector.keys()) - old_terms
            st.session_state.expanded_terms = list(new_terms)[:6]

            # Rerangiranje sa istim brojem rezultata (top_k)
            st.session_state.search_results = vector_search(
                st.session_state.query_vector, inverted_index, idf, doc_lengths, top_k=top_k
            )
            st.session_state.search_time = (time.time() - start_time) * 1000
            st.rerun()