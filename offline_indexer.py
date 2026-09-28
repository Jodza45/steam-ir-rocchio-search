import os
import pickle
import pandas as pd
from src.text_processor import preprocess_text
from src.indexer import build_index

DATA_DIR = "data"
STEAM_CSV = os.path.join(DATA_DIR, "steam.csv")
DESC_CSV = os.path.join(DATA_DIR, "steam_description_data.csv")
INDEX_PATH = os.path.join(DATA_DIR, "index_data.pkl")

def run_indexing():
    if not os.path.exists(STEAM_CSV) or not os.path.exists(DESC_CSV):
        print(f"GRESKA: Fajlovi steam.csv i steam_description_data.csv moraju biti u folderu '{DATA_DIR}'!")
        return

    df_steam = pd.read_csv(STEAM_CSV, usecols=['appid', 'name', 'positive_ratings'])
    
    df_desc = pd.read_csv(DESC_CSV, usecols=['steam_appid', 'short_description'])

    df = pd.merge(df_steam, df_desc, left_on='appid', right_on='steam_appid')
    df = df.dropna(subset=['name', 'short_description'])

    df = df.sort_values(by='positive_ratings', ascending=False).head(1000)

    docs_tokens = {}
    games_meta = {}

    print(f"Obrada teksta (tokenizacija i stemovanje) za {len(df)} igara...")
    for idx, row in df.iterrows():
        doc_id = str(idx)
        full_text = f"{row['name']} {row['short_description']}"
        docs_tokens[doc_id] = preprocess_text(full_text)
        games_meta[doc_id] = {"title": row['name'], "desc": row['short_description']}

    print("Izgradnja invertovanog indeksa od nule...")
    inverted_index, idf, doc_vectors, doc_lengths = build_index(docs_tokens)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({
            "inverted_index": inverted_index,
            "idf": idf,
            "doc_vectors": doc_vectors,
            "doc_lengths": doc_lengths,
            "games_meta": games_meta
        }, f)

    print("Invertovan index kreiran")

if __name__ == "__main__":
    run_indexing()