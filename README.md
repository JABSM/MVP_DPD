# MVP_DPD Plataforma inteligente de recomendación y moderación para restauración  
*(MVP – Jon Asier Barria San Martín)*

## 1. Descripción
Este repositorio contiene el código y los datos necesarios para ejecutar un **MVP 100% local** que:
xx. Genera datasets sintéticos (usuarios, restaurantes y reseñas) con **PyDBGen**.
1. Eliminado **PyDBGen** por problemas con la llamada a **Faker** así que se utiliza directamente
2. Carga de archivos planos .csv, .json, .txt
3. Limpieza, filtrado de palabras ofensivas-toxicas y normalización de los datos para garantizar calidad y consistencia
4. Deja los datos listos para las próximas fases modelos de moderación y recomendación
5. Genera los modelos SVD y KNN para su comparación
6. Muestra un dashboard interactivo con las metricas más significativas

> **Estado Actividad 2**  
> Actividad 2.1 completada
> Actividad 2.2 completada
> Actividad 2.3 completada

## 2. Estructura de carpetas

```
.
|── data
|   |── raw/          # Datos originales o sintéticos sin procesar
|   |── processed/    # Datos limpios listos para ML
|── models
|── mvp_data_pipeline.py
|── mvp_model_training.py
|── mvp_dashboard_svd.py
|── requirements.txt
|── README.md
```

## 3. Instalación

### 3.1 Crear y activar un entorno virtual es opcional pero recomendado

```bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
```

### 3.2 Instalar dependencias mínimas

```bash
pip install -r requirements.txt
```

> Probado con **Python 3.10 / 3.11** Windows 10.

## 4. Uso rápido

### 4.1 Generar datos sintéticos

```bash
python mvp_data_pipeline.py generate \
       --out-dir data/raw \
       --n-users 100 \
       --n-restaurants 30 \
       --n-reviews 500
```

> Se crearán tres archivos CSV en "data/raw/"  
> "users.csv", "restaurants.csv", "reviews.csv"

### 4.2 Limpiar un dataset, se tiene que realizar para los tres, reviews, restaurants y users

En esta limpieza mantenemos las reseñas ofensivas pero las identificamos como tal, así podemos realizar analisis con ellas

```bash
python mvp_data_pipeline.py clean data/raw/reviews.csv
python mvp_data_pipeline.py clean data/raw/restaurants.csv
python mvp_data_pipeline.py clean data/raw/users.csv
# Salidas
# data/processed/reviews_clean.csv
# data/processed/restaurants_clean.csv
# data/processed/users_clean.csv
```

> Si queremos eliminar las reseñas ofensivas podemos ejecutar

```bash
python mvp_data_pipeline_v2.py clean data/raw/reviews.csv --drop-toxic
```

> Se puede indicar un nombre de salida personalizado si se quiere

```bash
python mvp_data_pipeline.py clean \
       data/raw/users.csv \
       --out-file data/processed/users_clean_v1.0.csv
```

### 4.3 Entrenamientoy evaluación de los modelos para recomendación

```bash
python mvp_model_training.py \
      --reviews data/processed/reviews_clean.csv \
      --models-dir models \
      --test-size 0.2 \
      --seed 42
```

### 4.4 Dashboard interactivo para svd

```bash
streamlit run mvp_dashboard_svd.py
```

