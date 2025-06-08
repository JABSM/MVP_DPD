#!/usr/bin/env python3
"""
Todo el desarrollo para ejecutarse en local
mvp_data_pipeline.py
--------------------
Pipeline para el MVP de la plataforma de recomendación de restaurantes
xxxx. Genera datos sintéticos con PyDBGen y los guarda en CSV
1. Como la versión de PyDBGen daba problemascon la ultima de Faker, utilizo directamente Faker para genrar los datos sinteticos
2. Carga datasets .csv, .json o .txt
3. Aplica limpieza / listado de palabras ofensivas-toxicas++ / normalización básica
4. Guardar la versión procesada

Autor: Jon Asier Barria San Martín
"""

"""
Todas las referencias utilizadas, excepto omisión por error, se encuentran en este docuemnto
Si no se hace referencia a ellas se considera que el código generado es propio y por medio de los conocimientos del autor.
Podrán utilizarse partes de proyectos anteriores o similares, realizados durante el grado, siempre que sean del autor, sino se indicara su autor.

Referencias

https://github.com/tirthajyoti/pydbgen
https://opensource.com/article/18/11/pydbgen-random-database-table-generator
https://pydbgen.readthedocs.io/en/latest/

https://docs.python.org/3/library/typing.html
https://stackoverflow.com/questions/39458193/using-list-tuple-etc-from-typing-vs-directly-referring-type-as-list-tuple-etc
https://www.w3schools.com/python/python_tuples.asp

https://docs.python.org/3/library/argparse.html

https://faker.readthedocs.io/en/master/locales/es_ES.html
https://www.geeksforgeeks.org/python-faker-library/
https://github.com/faker-js/faker

"""

# Importación de librerias necesarias
import argparse
import re
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
#from pydbgen import pydbgen
from faker import Faker

"""
Funciones para generación de datos sinteticos
Aqui introducimos el tipo de cocina que ofrece cada restaurante
Genera los datos de los usuarios con pydbgen
La localización de los mismos
Las calificaciones de los usuarios para cada restaurante
Las reseñas de los usuarios para cada restaurante
Y se guardan los datos en CSV

!!! PyDbGen da problemas con la llamda a Faker, así que como usa Faker, uso directamente esta libreria sin
usar PyDbGen como se solicita¡¡¡

"""

# Listado de tipos de cocinas
tipo_cocina = [
    "italiana", "japonesa", "mexicana", "india", "española",
    "vegana", "mediterránea", "americana", "francesa", "tailandesa"
]
ciudades = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Vitoria"]

# Función para crear usuarios con Faker directamente
def _generate_users(n: int, seed: int) -> pd.DataFrame:
    Faker.seed(seed)
    fake = Faker("es_ES")
    data = [{
        "user_id": uid,
        "full_name": fake.name(),
        "email": fake.unique.email().lower(),
        "phone_number": fake.phone_number(), }
        for uid in range(1, n + 1)
    ]
    return pd.DataFrame(data)

# Función para generar los restaurantes y su ubicación
def _generate_restaurants(n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    data = [{
        "restaurant_id": rid,
        "restaurant_name": f"Restaurante {chr(64 + rid)}",
        "cocina": random.choice(tipo_cocina),
        "city": random.choice(ciudades), }
        for rid in range(1, n + 1)
    ]
    return pd.DataFrame(data)

# Función para generar la reseñas
def _generate_reviews(
    n_reviews: int, n_users: int, n_restaurants: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    sample_comments = [
        "¡Buenísimo!",
        "Servicio lento pero comida excelente",
        "Precio/calidad correcto",
        "No vuelvo, muy ruidoso",
        "El mejor sushi de la ciudad",
        "Demasiado hijos putA·$·& para mi gusto",
        "Así, así, caro y LA comida no eS para tanto",
        "$&%&/$%/%/",
        "09986 ~# ",
        "Trato nefasto, deberian cerrar",
        "Excelente los dueños maravillosos",
        "Una puta mierda, no he ido pero son uno cerdos y el sitio da por cuulo"
    ]
    data = [{
        "review_id": rid,
        "user_id": np.random.randint(1, n_users + 1),
        "restaurant_id": np.random.randint(1, n_restaurants + 1),
        "rating": np.random.randint(1, 6),
        "review_text": random.choice(sample_comments), }
        for rid in range(1, n_reviews + 1)
    ]
    return pd.DataFrame(data)

# Función para generar el dataset con todos los datos creados...
def generate_synthetic_data(
    out_dir: Path,
    n_users: int = 50,
    n_restaurants: int = 20,
    n_reviews: int = 300,
    seed: int = 42,) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    users = _generate_users(n_users, seed)
    restaurants = _generate_restaurants(n_restaurants, seed)
    reviews = _generate_reviews(n_reviews, n_users, n_restaurants, seed)

    f_users = out_dir / "users.csv" # Los guardo en CSV, podria ser en otro formarto
    f_restaurants = out_dir / "restaurants.csv"
    f_reviews = out_dir / "reviews.csv"

    users.to_csv(f_users, index=False)
    restaurants.to_csv(f_restaurants, index=False)
    reviews.to_csv(f_reviews, index=False)

    print(f"Datasets guardados en: {out_dir.resolve()}")
    return f_users, f_restaurants, f_reviews

"""
Función para cargar los datos
Podríamos cargar cualquier tipo de dataset csv, json, o txt
Si no lo tenemows reconocido, muestra mensaje de error

"""

def load_data(file_path: Path) -> pd.DataFrame:
  suffix = file_path.suffix.lower()
  if suffix == ".csv":
      df = pd.read_csv(file_path)
  elif suffix == ".json":
      df = pd.read_json(file_path, lines=True)
  elif suffix == ".txt":
      df = pd.read_csv(file_path, sep="\t")
  else:
      raise ValueError(f"Extensión no soportada: {suffix}")
  print(f"Cargado {file_path.name}: {df.shape[0]:,} registros × {df.shape[1]} atributos")
  return df

"""
Funciones de limpieza y normalización basicas
Eliminamos las palabrotas, ya que no tenemos conexión al ser para MVP las introducimos namualmente
Si encuentra palabrotas, las enmascara y devuelve, "texto_limpio" o "is_toxic"
Eliminamos los caracteres no alfanumericos
Eliminamos espacios duplicados
Pasamos todo a minusculas
Y elimina los duplicados

"""

OFFENSIVE_WORDS = {
    "gilipollas", "idiota", "imbécil", "estúpido", "puta", "puto", "hijo puta", "hija puta", "cerdo", "cerda"
    "cabron", "cabrona", "mierda", "joder", "coño", "polla", }

OFF_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, OFFENSIVE_WORDS)) + r")\b",
    flags=re.IGNORECASE, )

def _mask_offensive(text: str) -> tuple[str, bool]:
    def _mask(match):
        token = match.group(0)
        return token[0] + "*" * (len(token) - 1)   # g****, m*****
    cleaned, n_subs = OFF_PATTERN.subn(_mask, text)
    return cleaned, bool(n_subs)


NO_ALPHANUM = re.compile(r"[^0-9a-záéíóúüñ ]+", flags=re.IGNORECASE)

def _normalize_text(series: pd.Series) -> pd.Series:
  return (
      series.fillna("")
      .str.lower()
      .str.replace(NO_ALPHANUM, " ", regex=True)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip())


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["email"] = df["email"].str.lower().str.strip()
  df = df.drop_duplicates(subset="email")
  return df


def clean_restaurants(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["restaurant_name"] = df["restaurant_name"].str.title().str.strip()
  df["cocina"] = df["cocina"].str.lower().str.strip()
  return df.drop_duplicates(subset=["restaurant_name", "city"])


def clean_reviews(df: pd.DataFrame, drop_toxic: bool = False) -> pd.DataFrame:
  df = df.copy()
  df["review_text"] = _normalize_text(df["review_text"])
  results = df["review_text"].apply(_mask_offensive)
  df["review_text"] = results.str[0]
  df["is_toxic"] = results.str[1]
  if drop_toxic:
      df = df.loc[~df["is_toxic"]]
  df["rating"] = df["rating"].clip(1, 5).astype(int)
  df = df.dropna(subset=["user_id", "restaurant_id", "rating"])
  df = df.drop_duplicates(subset="review_id")
  return df

"""
Guardo los datos procesados en CSV

"""

def save_processed(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Guardado del archivo procesado: {out_path}")
    print(f"Tamaño del archivo: {out_path.stat().st_size / (1024 * 1024):.2f} MB")

"""
Main para ejecutar el programa

"""

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Pipeline MVP datos restauración")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Genera los datasets sintéticos")
    g.add_argument("--out-dir", type=Path, default="data/raw")
    g.add_argument("--n-users", type=int, default=50)
    g.add_argument("--n-restaurants", type=int, default=20)
    g.add_argument("--n-reviews", type=int, default=300)

    c = sub.add_parser("clean", help="Limpia un dataset, son tres")
    c.add_argument("in_file", type=Path)
    c.add_argument("--out-file", type=Path)

    return p.parse_args()


def main() -> None:
    args = _cli()

    if args.cmd == "generate":
        generate_synthetic_data(
            args.out_dir, args.n_users, args.n_restaurants, args.n_reviews
        )

    if args.cmd == "clean":
        df = load_data(args.in_file)
        stem = args.in_file.stem
        if "users" in stem:
            df = clean_users(df)
        elif "restaurants" in stem:
            df = clean_restaurants(df)
        elif "reviews" in stem:
            df = clean_reviews(df)
        else:
            df = df.drop_duplicates().dropna()  # si es un dataset cualquiera realizo una limpieza genérica
        out = args.out_file or Path("data/processed") / f"{stem}_clean.csv"
        save_processed(df, out)


if __name__ == "__main__":
    main()