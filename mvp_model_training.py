#!/usr/bin/env python3
"""
mvp_model_training.py
---------------------
Entrenamiento y evaluación de dos modelos para recomendación
1. SVD x factorización de matriz   – De la libreria Surprise
2. KNNBasic basado en el coseno    – De la libreria Surprise

Métricas definidas
  • RMSE de predicción de rating
  • Hit Rate @10 (HR@10)  Al menos una recomendación del Top_10 tiene que coincidir con un ítem verdaderamente relevante en el test

Se registraran los tiempos de entrenamiento y predicción para el analisis

Autor: Jon Asier Barria San Martín
"""

"""
Todas las referencias utilizadas, excepto omisión por error, se encuentran en este docuemnto
Si no se hace referencia a ellas se considera que el código generado es propio y por medio de los conocimientos del autor.
Podrán utilizarse partes de proyectos anteriores o similares, realizados durante el grado, siempre que sean del autor, sino se indicara su autor.

Referencias

https://surpriselib.com/
https://surprise.readthedocs.io/en/stable/getting_started.html
https://medium.com/@wadrianrisso/surprise-una-librer%C3%ADa-de-python-para-el-desarrollo-de-tu-sistema-de-recomendaci%C3%B3n-441306172196
https://medium.com/@andres_frojas/explorando-la-biblioteca-surprise-en-python-para-sistemas-de-recomendaci%C3%B3n-cf4a744fbe8c
https://www.geeksforgeeks.org/svd-in-recommendation-systems/
https://www.researchgate.net/publication/343462569_Surprise_A_Python_library_for_recommender_systems

"""

# Importación de librerias necesarias
import argparse, time, re, collections, itertools, random, pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split

"""
Carga las reseñas limpias
Crea df para SVD con la libreria Surprise
Calcula HR@k == % de usuarios cuyo Top_k incluye al menos un ítem con rating real >= min_rating en el set de test
Calcula el tiempo de entrenamiento en s
Entrena, predice, calcula métricas y muestra los resultados

Además, si la columna "is_toxic" está presente, **NO** se elimina, se quiere
mantener estas filas para obtener métricas. De todos modos, esas columnas no
se utilizan para entrenar, sólo para informar del porcentaje de toxicidad en
datos de entrada.
"""

def load_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_toxic" in df.columns:
        toxic_ratio = df["is_toxic"].mean()
        print(f"Reseñas tóxicas en el dataset: {toxic_ratio:.2%} "
              f"({df['is_toxic'].sum()} de {len(df)})")
    needed = {"user_id", "restaurant_id", "rating"}
    if not needed.issubset(df.columns):
        raise ValueError(f"El CSV debe tener columnas {needed}")
    return df[list(needed)].astype(int)


def surprise_data(df: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df, reader)
    
def hit_rate_k(predictions, k: int = 10, min_rating: int = 4) -> float:
    truth: Dict[int, set] = {}
    ranked: Dict[int, List[Tuple[int, float]]] = {}

    for uid, iid, true_r, est, _ in predictions:
        if true_r >= min_rating:
            truth.setdefault(uid, set()).add(iid)
        ranked.setdefault(uid, []).append((iid, est))

    hits = 0
    for uid, ests in ranked.items():
        topk = {iid for iid, _ in sorted(ests, key=lambda x: x[1], reverse=True)[:k]}
        if truth.get(uid, set()) & topk:
            hits += 1
    return hits / len(ranked) if ranked else 0.0


def time_it(func, *args, **kwargs):
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    return res, time.perf_counter() - t0


def evaluate(
    algo,
    label: str,
    base_name: str,
    trainset,
    testset,
    model_dir: Path,):
    _, t_train = time_it(algo.fit, trainset)
    preds, t_pred = time_it(algo.test, testset)

    rmse = accuracy.rmse(preds, verbose=False)
    hr10 = hit_rate_k(preds, k=10)

    model_path = save_model(algo, model_dir, base_name)

    print(f"\n=== {label} ===")
    print(f"Guardado en          : {model_path}")
    print(f"Tiempo entrenamiento : {t_train:7.3f} s")
    print(f"Tiempo predicción    : {t_pred:7.3f} s")
    print(f"RMSE                 : {rmse:7.4f}")
    print(f"Hit Rate @10         : {hr10:7.4f}")
    
"""
Guardado de los modelos con versionado para seguimeinto
Calcula MAJOR.MINOR siguiente según los ficheros existentes
"""

VER_RE = re.compile(r"_v(\d+)\.(\d+)\.pkl$")


def _next_version(model_dir: Path, base_name: str) -> str:
    majors, minors = 0, 0
    for f in model_dir.glob(f"{base_name}_v*.pkl"):
        m = VER_RE.search(f.name)
        if m:
            maj, min_ = map(int, m.groups())
            if maj > majors or (maj == majors and min_ > minors):
                majors, minors = maj, min_
    return f"{majors}.{minors + 1}" if (majors or minors) else "1.0"


def save_model(algo, model_dir: Path, base_name: str) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    ver = _next_version(model_dir, base_name)
    path = model_dir / f"{base_name}_v{ver}.pkl"
    with open(path, "wb") as f:
        pickle.dump(algo, f)
    return path

"""
Main para ejecutar el programa
"""

def main():
    p = argparse.ArgumentParser("Entrena, evalúa y versiona los modelos MVP")
    p.add_argument("--reviews", type=Path,
                   default="data/processed/reviews_clean.csv")
    p.add_argument("--models-dir", type=Path, default="models")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = load_reviews(args.reviews)
    data = surprise_data(df)
    train, test = train_test_split(data, test_size=args.test_size, random_state=args.seed)

    svd = SVD(random_state=args.seed)
    knn = KNNBasic(sim_options={"name": "cosine", "user_based": False})

    evaluate(
        svd, "SVD x factorización", "svd_model", train, test, args.models_dir)
        
    evaluate(
        knn, "KNN x coseno", "knn_item_model", train, test, args.models_dir)


if __name__ == "__main__":
    main()
