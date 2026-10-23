"""Canalización reproducible para explorar precios de diamantes con OLS.

Ejecutar desde la raíz del repositorio:
    python -m src.analysis

Genera tablas y gráficas en outputs/ para explorar el ajuste y sus supuestos.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "diamonds.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

TARGET = "price"
NUMERIC_PREDICTORS = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL_PREDICTORS = ["cut", "color", "clarity"]
REQUIRED_COLUMNS = [TARGET, *NUMERIC_PREDICTORS, *CATEGORICAL_PREDICTORS]

# Los órdenes fijan niveles de referencia interpretable en el modelo OLS.
CATEGORY_ORDERS = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

FULL_FORMULA = (
    "price ~ carat + depth + table + x + y + z + "
    "C(cut) + C(color) + C(clarity)"
)


@dataclass
class OLSResult:
    """Resultados necesarios de un ajuste por mínimos cuadrados ordinarios."""

    term_names: list[str]
    params: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    conf_int: np.ndarray
    fittedvalues: np.ndarray
    resid: np.ndarray
    exog: np.ndarray
    nobs: int
    df_resid: int
    rsquared: float
    rsquared_adj: float
    mse_resid: float
    fvalue: float
    f_pvalue: float
    llf: float
    aic: float
    bic: float


def ensure_output_dirs() -> None:
    """Crea los directorios de artefactos si aún no existen."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el CSV y valida el esquema que se usará en el análisis."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")

    data = pd.read_csv(path)
    # El archivo proveniente de Kaggle conserva un índice serializado sin nombre.
    unnamed = [column for column in data.columns if column.lower().startswith("unnamed")]
    data = data.drop(columns=unnamed, errors="ignore")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")

    for column, order in CATEGORY_ORDERS.items():
        data[column] = pd.Categorical(data[column], categories=order, ordered=True)
    return data


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Elimina filas sin datos requeridos y dimensiones físicamente imposibles.

    Los registros con x, y o z menores o iguales a cero no describen un diamante
    con volumen válido y afectarían la interpretación de predictores geométricos.
    No se eliminan valores extremos adicionales: se preserva el dataset fuente.
    """
    before = len(data)
    cleaned = data.dropna(subset=REQUIRED_COLUMNS).copy()
    after_missing = len(cleaned)
    positive_dimensions = (cleaned[["x", "y", "z"]] > 0).all(axis=1)
    cleaned = cleaned.loc[positive_dimensions].copy()
    audit = {
        "rows_original": before,
        "rows_removed_missing": before - after_missing,
        "rows_removed_nonpositive_dimensions": after_missing - len(cleaned),
        "rows_used": len(cleaned),
    }
    return cleaned, audit


def data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un diccionario de variables específico del conjunto usado."""
    descriptions = {
        "price": "Precio del diamante en dólares estadounidenses; variable dependiente.",
        "carat": "Peso del diamante en quilates.",
        "cut": "Calidad del corte, de Fair a Ideal.",
        "color": "Calidad del color, de J a D.",
        "clarity": "Claridad del diamante, de I1 a IF.",
        "depth": "Profundidad total porcentual: z / media(x, y) × 100.",
        "table": "Ancho de la parte superior del diamante como porcentaje del diámetro mayor.",
        "x": "Longitud en milímetros.",
        "y": "Anchura en milímetros.",
        "z": "Profundidad en milímetros.",
    }
    roles = {
        TARGET: "Dependiente numérica",
        **{column: "Independiente numérica" for column in NUMERIC_PREDICTORS},
        **{column: "Independiente categórica" for column in CATEGORICAL_PREDICTORS},
    }
    return pd.DataFrame(
        {
            "variable": data.columns,
            "tipo": [str(data[column].dtype) for column in data.columns],
            "rol_en_modelo": [roles.get(column, "No usada") for column in data.columns],
            "descripcion": [descriptions.get(column, "") for column in data.columns],
            "valores_faltantes": [int(data[column].isna().sum()) for column in data.columns],
            "niveles": [
                int(data[column].nunique()) if column in CATEGORICAL_PREDICTORS else pd.NA
                for column in data.columns
            ],
        }
    )


def design_matrix(data: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, list[str]]:
    """Construye una matriz con intercepto y dummies con una categoría de referencia."""
    columns = [pd.Series(1.0, index=data.index, name="Intercept")]
    names = ["Intercept"]
    for predictor in predictors:
        if predictor in NUMERIC_PREDICTORS:
            columns.append(pd.to_numeric(data[predictor], errors="raise").astype(float))
            names.append(predictor)
        elif predictor in CATEGORICAL_PREDICTORS:
            categories = list(data[predictor].cat.categories)
            reference = categories[0]
            for category in categories[1:]:
                columns.append((data[predictor] == category).astype(float))
                names.append(f"C({predictor})[T.{category}]")
            if reference not in set(data[predictor].dropna().unique()):
                raise ValueError(f"No hay observaciones para la categoría de referencia {reference}.")
        else:
            raise ValueError(f"Predictor no reconocido: {predictor}")
    return pd.concat(columns, axis=1).to_numpy(dtype=float), names


def fit_ols(data: pd.DataFrame, predictors: list[str]) -> OLSResult:
    """Ajusta OLS y calcula inferencia bajo los supuestos clásicos del modelo lineal."""
    exog, names = design_matrix(data, predictors)
    response = data[TARGET].to_numpy(dtype=float)
    params, _, rank, _ = np.linalg.lstsq(exog, response, rcond=None)
    fitted = exog @ params
    residuals = response - fitted
    nobs, n_parameters = exog.shape
    df_resid = nobs - rank
    if df_resid <= 0:
        raise ValueError("No hay grados de libertad residuales suficientes para OLS.")
    sse = float(np.dot(residuals, residuals))
    centered_sst = float(np.dot(response - response.mean(), response - response.mean()))
    mse_resid = sse / df_resid
    covariance = mse_resid * np.linalg.pinv(exog.T @ exog)
    bse = np.sqrt(np.diag(covariance))
    tvalues = params / bse
    pvalues = 2 * stats.t.sf(np.abs(tvalues), df_resid)
    critical_t = stats.t.ppf(0.975, df_resid)
    confidence_intervals = np.column_stack((params - critical_t * bse, params + critical_t * bse))
    r_squared = 1 - sse / centered_sst
    r_squared_adjusted = 1 - (1 - r_squared) * (nobs - 1) / df_resid
    model_df = rank - 1
    fvalue = ((centered_sst - sse) / model_df) / mse_resid
    f_pvalue = stats.f.sf(fvalue, model_df, df_resid)
    log_likelihood = -0.5 * nobs * (np.log(2 * np.pi) + 1 + np.log(sse / nobs))
    aic = -2 * log_likelihood + 2 * rank
    bic = -2 * log_likelihood + np.log(nobs) * rank
    return OLSResult(
        term_names=names,
        params=params,
        bse=bse,
        tvalues=tvalues,
        pvalues=pvalues,
        conf_int=confidence_intervals,
        fittedvalues=fitted,
        resid=residuals,
        exog=exog,
        nobs=nobs,
        df_resid=df_resid,
        rsquared=r_squared,
        rsquared_adj=r_squared_adjusted,
        mse_resid=mse_resid,
        fvalue=fvalue,
        f_pvalue=f_pvalue,
        llf=log_likelihood,
        aic=aic,
        bic=bic,
    )


def fit_full_model(data: pd.DataFrame) -> OLSResult:
    """Ajusta el modelo OLS con seis predictores numéricos y tres categóricos."""
    return fit_ols(data, [*NUMERIC_PREDICTORS, *CATEGORICAL_PREDICTORS])


def model_metrics(model: OLSResult) -> pd.DataFrame:
    """Extrae métricas habituales para evaluar el ajuste lineal."""
    omnibus = stats.normaltest(model.resid)
    jarque_bera = stats.jarque_bera(model.resid)
    omnibus_stat, omnibus_pvalue = float(omnibus.statistic), float(omnibus.pvalue)
    jb_stat, jb_pvalue = float(jarque_bera.statistic), float(jarque_bera.pvalue)
    skew = float(stats.skew(model.resid, bias=False))
    kurtosis = float(stats.kurtosis(model.resid, fisher=False, bias=False))
    durbin_watson = float(np.sum(np.diff(model.resid) ** 2) / np.sum(model.resid**2))
    metrics = {
        "r_squared": model.rsquared,
        "r_squared_adjusted": model.rsquared_adj,
        "residual_variance_mse": model.mse_resid,
        "f_statistic": model.fvalue,
        "f_statistic_pvalue": model.f_pvalue,
        "log_likelihood": model.llf,
        "aic": model.aic,
        "bic": model.bic,
        "omnibus": omnibus_stat,
        "omnibus_pvalue": omnibus_pvalue,
        "durbin_watson": durbin_watson,
        "jarque_bera": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "skew": skew,
        "kurtosis": kurtosis,
        "condition_number": np.linalg.cond(model.exog),
        "observations": int(model.nobs),
    }
    return pd.DataFrame(metrics.items(), columns=["metrica", "valor"])


def coefficient_table(model: OLSResult) -> pd.DataFrame:
    """Construye una tabla de coeficientes, errores estándar, t, p e IC 95%."""
    table = pd.DataFrame(
        {
            "termino": model.term_names,
            "coeficiente": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
            "ci_95_inf": model.conf_int[:, 0],
            "ci_95_sup": model.conf_int[:, 1],
        }
    )
    table["significativo_5pct"] = table["p_value"] < 0.05
    return table


def select_two_numeric_predictor_models(data: pd.DataFrame) -> pd.DataFrame:
    """Compara los 15 pares de predictores numéricos por AIC y R².

    Se comparan variables numéricas originales para que todos los modelos tengan
    exactamente dos predictores. Las variables categóricas se dejan en el modelo
    completo; cada una genera varios indicadores y no es directamente comparable
    con un único predictor numérico en esta comparación.
    """
    records: list[dict[str, Any]] = []
    for first, second in itertools.combinations(NUMERIC_PREDICTORS, 2):
        model = fit_ols(data, [first, second])
        records.append(
            {
                "predictor_1": first,
                "predictor_2": second,
                "aic": model.aic,
                "bic": model.bic,
                "r_squared": model.rsquared,
                "r_squared_adjusted": model.rsquared_adj,
                "f_statistic": model.fvalue,
                "f_statistic_pvalue": model.f_pvalue,
            }
        )
    return pd.DataFrame(records).sort_values("aic", kind="stable").reset_index(drop=True)


def breusch_pagan_test(model: OLSResult) -> pd.DataFrame:
    """Aplica Breusch-Pagan; H0: homocedasticidad de los residuos."""
    # Regresión auxiliar de e_i² sobre el mismo conjunto de regresores.
    residual_squares = model.resid**2
    aux_params, _, _, _ = np.linalg.lstsq(model.exog, residual_squares, rcond=None)
    aux_fitted = model.exog @ aux_params
    aux_sse = float(np.sum((residual_squares - aux_fitted) ** 2))
    aux_sst = float(np.sum((residual_squares - residual_squares.mean()) ** 2))
    aux_r_squared = 1 - aux_sse / aux_sst
    test_df = model.exog.shape[1] - 1
    lm_stat = model.nobs * aux_r_squared
    lm_pvalue = stats.chi2.sf(lm_stat, test_df)
    f_stat = (aux_r_squared / test_df) / ((1 - aux_r_squared) / (model.nobs - model.exog.shape[1]))
    f_pvalue = stats.f.sf(f_stat, test_df, model.nobs - model.exog.shape[1])
    return pd.DataFrame(
        {
            "estadistico": ["LM", "LM p_value", "F", "F p_value"],
            "valor": [lm_stat, lm_pvalue, f_stat, f_pvalue],
        }
    )


def create_figures(data: pd.DataFrame, model: OLSResult) -> None:
    """Crea visualizaciones de exploración y diagnóstico del ajuste."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(data["carat"], data[TARGET], alpha=0.18, s=8, color="#1f77b4")
    axis.set(title="Precio frente a quilates", xlabel="Quilates", ylabel="Precio (USD)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_vs_carat.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(model.fittedvalues, model.resid, alpha=0.18, s=8, color="#2a9d8f")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        title="Residuos frente a valores ajustados",
        xlabel="Precio ajustado (USD)",
        ylabel="Residuo (USD)",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals_vs_fitted.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    stats.probplot(model.resid, dist="norm", plot=axis)
    axis.set_title("Gráfica Q-Q de residuos")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residuals_qq.png", dpi=180)
    plt.close(fig)


def run_pipeline() -> dict[str, Any]:
    """Ejecuta el análisis técnico completo y persiste resultados reproducibles."""
    ensure_output_dirs()
    raw_data = load_data()
    data, audit = clean_data(raw_data)
    dictionary = data_dictionary(data)
    model = fit_full_model(data)
    coefficients = coefficient_table(model)
    metrics = model_metrics(model)
    candidate_models = select_two_numeric_predictor_models(data)
    bp_results = breusch_pagan_test(model)

    dictionary.to_csv(TABLES_DIR / "data_dictionary.csv", index=False)
    pd.DataFrame([audit]).to_csv(TABLES_DIR / "data_cleaning_audit.csv", index=False)
    coefficients.to_csv(TABLES_DIR / "full_model_coefficients.csv", index=False)
    metrics.to_csv(TABLES_DIR / "full_model_metrics.csv", index=False)
    candidate_models.to_csv(TABLES_DIR / "two_numeric_predictor_models.csv", index=False)
    bp_results.to_csv(TABLES_DIR / "breusch_pagan.csv", index=False)
    summary_lines = [
        "Regresión lineal múltiple OLS",
        f"Fórmula: {FULL_FORMULA}",
        f"Observaciones: {model.nobs}",
        f"R²: {model.rsquared:.6f}",
        f"R² ajustada: {model.rsquared_adj:.6f}",
        f"F: {model.fvalue:.6f}; p(F): {model.f_pvalue:.6g}",
        f"Log-Likelihood: {model.llf:.6f}; AIC: {model.aic:.6f}; BIC: {model.bic:.6f}",
        "",
        coefficients.to_string(index=False),
    ]
    (TABLES_DIR / "full_model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    significant_numeric = (
        coefficients.loc[coefficients["termino"].isin(NUMERIC_PREDICTORS)]
        .assign(abs_t=lambda frame: frame["t"].abs())
        .sort_values("abs_t", ascending=False)
    )
    significant_numeric.to_csv(TABLES_DIR / "numeric_predictors_by_abs_t.csv", index=False)
    create_figures(data, model)

    result = {
        "rows_used": audit["rows_used"],
        "full_model_formula": FULL_FORMULA,
        "best_pair_by_aic": candidate_models.iloc[0][["predictor_1", "predictor_2"]].to_dict(),
        "best_pair_by_r_squared": candidate_models.sort_values(
            "r_squared", ascending=False, kind="stable"
        )
        .iloc[0][["predictor_1", "predictor_2"]]
        .to_dict(),
    }
    (TABLES_DIR / "run_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    summary = run_pipeline()
    print("Análisis completado.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
