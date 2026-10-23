# Análisis de precios de diamantes con regresión lineal

Repositorio de análisis reproducible que examina la relación entre las características físicas y de calidad de un diamante y su precio. Combina preparación de datos, regresión lineal múltiple, comparación de modelos y diagnóstico de supuestos en una canalización ejecutable de extremo a extremo.

## Objetivo

Construir una referencia compacta de análisis de regresión aplicada: partir de datos tabulares, ajustar un modelo interpretable y dejar disponibles los artefactos necesarios para evaluar tanto su capacidad explicativa como sus limitaciones.

El análisis responde tres preguntas:

- ¿Qué características del diamante muestran mayor asociación con el precio?
- ¿Qué par de variables predictoras ofrece el mejor ajuste según AIC y R²?
- ¿Qué revelan los residuos sobre los supuestos del modelo lineal?

## Datos y enfoque

**Fuente:** [Diamonds — Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds?resource=download). El archivo `diamonds.csv` se incorporó al proyecto desde el archivo de datos descargado para este análisis y está disponible en `data/raw/`.

El dataset contiene 53,940 observaciones, siete variables numéricas (`carat`, `depth`, `table`, `price`, `x`, `y`, `z`) y tres categóricas (`cut`, `color`, `clarity`).

La variable dependiente es `price` (precio en USD). El modelo completo usa los seis predictores numéricos restantes y las tres variables categóricas codificadas mediante indicadores:

```text
price ~ carat + depth + table + x + y + z + C(cut) + C(color) + C(clarity)
```

Antes del ajuste se excluyen únicamente registros con alguna dimensión `x`, `y` o `z` menor o igual a cero, pues no representan una geometría físicamente válida. El script registra cuántas filas se excluyeron; no descarta otros valores extremos de manera arbitraria.

## Estructura

```text
data/raw/diamonds.csv              # dataset extraído del ZIP proporcionado
notebooks/01_regresion_diamantes.ipynb
src/analysis.py                    # canalización reproducible
src/validate_statsmodels.py        # validación independiente del ajuste OLS
tests/test_pipeline.py              # prueba de integración del proyecto
outputs/tables/                    # resultados CSV, resumen OLS y prueba BP
outputs/figures/                   # figuras de exploración y diagnóstico
requirements.txt
```

## Instalación y ejecución

Se recomienda Python 3.11 o superior en un entorno virtual.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.analysis
python -m unittest tests.test_pipeline
python -m src.validate_statsmodels
jupyter lab
```

Abra después `notebooks/01_regresion_diamantes.ipynb` desde la raíz del proyecto. El notebook ejecuta la misma canalización y permite revisar las tablas y figuras generadas.

## Resultados que genera

| Resultado | Artefacto |
| --- | --- |
| Exploración de datos | `data_dictionary.csv`, auditoría de limpieza y gráfica de precio frente a quilates. |
| Modelo completo | `full_model_summary.txt`, `full_model_coefficients.csv` y `full_model_metrics.csv`. |
| Significancia e incertidumbre | `significant_terms_5pct.csv` separa términos significativos al 5 %; `categorical_variables_joint_significance.csv` evalúa cada variable categórica como grupo; los IC están en `full_model_coefficients.csv`. |
| Diagnóstico | Métricas de normalidad, Durbin-Watson, número de condición, gráfica de residuos y gráfica Q-Q. |
| Comparación de modelos | `two_predictor_models.csv` compara los 36 pares posibles de las nueve variables predictoras por AIC, R² y R² ajustada. |
| Predictores principales | `top_two_numeric_predictors_by_abs_t.csv` identifica los dos predictores numéricos con mayor `|t|`; `selected_predictor_ci_95.csv` muestra el IC 95 % del primero. |
| Varianza de residuos | `breusch_pagan.csv` contiene la prueba de Breusch-Pagan. |
| Validación externa | `statsmodels_validation.csv`, generado por `python -m src.validate_statsmodels`, contrasta el ajuste contra `statsmodels`. |

La comparación de pares usa las nueve variables predictoras originales. Cada variable categórica se codifica mediante indicadores con una categoría de referencia; por ello, la tabla incluye `n_parameters` para mostrar cuántos parámetros estima cada modelo.

## Reproducibilidad

Los archivos de `outputs/` se excluyen de Git porque se regeneran al ejecutar la canalización. Los datos de entrada, el código y el notebook permanecen versionados para facilitar la revisión, adaptación y repetición del análisis.
