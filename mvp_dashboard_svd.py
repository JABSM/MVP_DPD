#!/usr/bin/env python3
"""
mvp_dashboard_svd.py
---------------------
Dashboard interactivo para el MVP de recomendación & moderación


Autor: Jon Asier Barria San Martín
"""

"""
Todas las referencias utilizadas, excepto omisión por error, se encuentran en este docuemnto
Si no se hace referencia a ellas se considera que el código generado es propio y por medio de los conocimientos del autor.
Podrán utilizarse partes de proyectos anteriores o similares, realizados durante el grado, siempre que sean del autor, sino se indicara su autor.

Referencias

https://streamlit.io/
https://github.com/streamlit/streamlit
https://www.datacamp.com/tutorial/streamlit
https://www.datacamp.com/tutorial/streamlit
https://www.geeksforgeeks.org/streamlit-introduction-and-setup/

https://www.geeksforgeeks.org/generating-word-cloud-python/
https://amueller.github.io/word_cloud/

"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from surprise import SVD

"""
Carga de datasets "limpios" y modelo svd

"""

RAW_DIR = Path("data/processed")
MODEL_PATH = Path("models/svd_model_v1.2.pkl")  # modificar a la última versión disponible

@st.cache_data(show_spinner="Cargando datasets…")
def load_data():
    users = pd.read_csv(RAW_DIR / "users_clean.csv")
    restaurants = pd.read_csv(RAW_DIR / "restaurants_clean.csv")
    reviews = pd.read_csv(RAW_DIR / "reviews_clean.csv")
    return users, restaurants, reviews


@st.cache_resource(show_spinner="Cargando modelo SVD…")
def load_model() -> SVD:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

"""
Muestra de los 5 KPI's del producto
 
"""

def compute_kpis(u: pd.DataFrame, r: pd.DataFrame, rv: pd.DataFrame):
    k = {
        "Usuarios": u["user_id"].nunique(),
        "Restaurantes": r["restaurant_id"].nunique(),
        "Reseñas": len(rv),
        "Rating medio": rv["rating"].mean().round(2),
        "Cobertura rec. (%)": round(100 * rv["restaurant_id"].nunique() / r["restaurant_id"].nunique(), 1),
    }
    if "is_toxic" in rv.columns:
        tox = int(rv["is_toxic"].sum())
        k["Reseñas tóxicas"] = tox
        k["Tóxicas (%)"] = round(100 * tox / len(rv), 1)
    return k

"""
Generación de recomendaciones   
Devuelve Top_n restaurantes no visitados por el usuario ordenados por el rating estimado

"""

def get_top_n(model: SVD, df_reviews: pd.DataFrame, user_id: int, n: int = 10):
    reviewed = set(df_reviews.loc[df_reviews.user_id == user_id, "restaurant_id"])
    candidates = df_reviews["restaurant_id"].unique()
    preds = [(iid, model.predict(user_id, iid).est) for iid in candidates if iid not in reviewed]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]

"""
Explicabilidad “casera” SVD, ya que no existe soporte SHAP directo para este
La descomposición de la predicción en sesgo global + sesgo usuario + sesgo restaurante + producto de factores latentes, 
es la forma “nativa” de explicar un SVD
Devuelve los componentes de la predicción SVD, μ + b_u + b_i + q_i·p_u, usando los IDs internos de Surprise

"""

def explain_svd(model: SVD, uid_raw: int, iid_raw: int):
    tr = model.trainset
    uid = tr.to_inner_uid(uid_raw)
    iid = tr.to_inner_iid(iid_raw)
    mu = tr.global_mean
    bu = model.bu[uid] if uid < len(model.bu) else 0.0
    bi = model.bi[iid] if iid < len(model.bi) else 0.0
    dot = float(np.dot(model.pu[uid], model.qi[iid]))
    return {
        "SESGO GLOBAL (μ)": mu,
        "SESGO USUARIO (b_u)": bu,
        "SESGO REST. (b_i)": bi,
        "FACTORES (q·p)": dot,
        "PREDICCIÓN FINAL": mu + bu + bi + dot,
    }

"""
Visualización de los KPI's
"""

def avg_rating_x_cocina(rv: pd.DataFrame, rest: pd.DataFrame):
    merged = rv.merge(rest, on="restaurant_id")
    stats = merged.groupby("cocina")["rating"].mean().sort_values(ascending=False)
    st.subheader("⭐ Rating medio por tipo de cocina")
    fig, ax = plt.subplots()
    sns.barplot(x=stats.values, y=stats.index, ax=ax)
    ax.set_xlabel("Rating medio")
    st.pyplot(fig)


def rating_distri(rv: pd.DataFrame):
    st.subheader("Distribución de ratings")
    fig, ax = plt.subplots()
    sns.histplot(rv["rating"], bins=5, kde=True, ax=ax)
    ax.set_xlabel("Rating")
    st.pyplot(fig)


def reviews_x_city(rv: pd.DataFrame, rest: pd.DataFrame):
    merged = rv.merge(rest, on="restaurant_id")
    city_count = merged["city"].value_counts()
    st.subheader("Nº de reseñas por ciudad")
    fig, ax = plt.subplots()
    city_count.plot(kind="bar", ax=ax)
    ax.set_ylabel("Nº reseñas")
    st.pyplot(fig)


def wordcloud_x_rest(rv: pd.DataFrame, restaurant_id: int, restaurant_name: str):
    subset = (
        rv.loc[rv["restaurant_id"] == restaurant_id, "review_text"]
        .dropna()
        .astype(str)
    )
    if subset.empty:
        st.info("No hay reseñas suficientes para generar la nube de palabras.")
        return
    # Construye texto y filtra palabras muy cortas
    text = " ".join([w for w in " ".join(subset).split() if len(w) > 3])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    st.subheader(f"Nube de palabras – {restaurant_name}")
    st.image(wc.to_array())

"""
Main para ejecutar el programa

"""

def main():
    st.title("APP de recomendación_MVP_demo")

    users, restaurants, reviews = load_data()
    model = load_model()

    # KPI's
    for col, (k, v) in zip(st.columns(len(compute_kpis(users, restaurants, reviews))), compute_kpis(users, restaurants, reviews).items()):
        col.metric(k, f"{v:,.0f}" if isinstance(v, (int, float)) else v)

    st.divider()

    # Experiencia de usuario demo
    st.header("Demo de recomendación personalizada")
    uid = st.selectbox("Selecciona un usuario", users["user_id"].unique())
    top_n = get_top_n(model, reviews, uid)

    if not top_n:
        st.warning("Este usuario ya ha valorado todos los restaurantes. No hay recomendaciones nuevas.")
        return

    # bloque para evitar errores en el visualizador por falta de datos
    id2name = restaurants.set_index("restaurant_id")["restaurant_name"].to_dict()

    df_top = pd.DataFrame([
        {
            "Ranking": i + 1,
            "RestaurantID": iid,
            "Restaurante": id2name.get(iid, f"Rest {iid}"),
            "Predicted Rating": f"{est:.2f}",
        }
        for i, (iid, est) in enumerate(top_n)
    ])
    st.dataframe(df_top, use_container_width=True)

    # Explicación de la predicción del #n
    st.subheader("Explicación de la primera recomendación")
    best_iid, _ = top_n[0]
    explanation = explain_svd(model, uid, best_iid)
    fig, ax = plt.subplots()
    parts = {k: v for k, v in explanation.items() if k != "PREDICCIÓN FINAL"}
    ax.barh(list(parts.keys()), list(parts.values()))
    ax.set_xlabel("Contribución a la predicción")
    st.pyplot(fig)
    st.caption(f"Predicción final: **{explanation['PREDICCIÓN FINAL']:.2f}**")

    # Nube de palabras del restaurante Top_n
    wordcloud_x_rest(reviews, best_iid, id2name.get(best_iid, str(best_iid)))

    st.divider()

    # Panel de análisis
    st.header("Analytics")
    avg_rating_x_cocina(reviews, restaurants)
    reviews_x_city(reviews, restaurants)
    rating_distri(reviews)


if __name__ == "__main__":
    main()