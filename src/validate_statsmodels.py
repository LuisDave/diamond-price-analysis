"""Valida el ajuste OLS propio contra statsmodels.

Ejecutar después de instalar requirements.txt:
    python -m src.validate_statsmodels
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import FULL_FORMULA, TABLES_DIR, clean_data, fit_full_model, load_data


def validate() -> pd.DataFrame:
    """Compara coeficientes, inferencia y métricas con la implementación externa."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as error:
        raise RuntimeError(
            "statsmodels no está instalado. Ejecute `python -m pip install -r requirements.txt`."
        ) from error

    data, _ = clean_data(load_data())
    own_model = fit_full_model(data)
    reference_model = smf.ols(FULL_FORMULA, data=data).fit()

    reference_terms = reference_model.params.index.tolist()
    if set(own_model.term_names) != set(reference_terms):
        raise AssertionError(
            "Los conjuntos de términos no coinciden. Revise la codificación de categorías y sus referencias."
        )

    # Patsy coloca los términos categóricos antes de los numéricos; el ajuste
    # propio conserva el orden de entrada. Se alinean por nombre antes de comparar.
    own_positions = {term: position for position, term in enumerate(own_model.term_names)}
    own_order = np.array([own_positions[term] for term in reference_terms])

    reference_ci = reference_model.conf_int()
    own_metrics = {
        "coefficient": own_model.params[own_order],
        "std_err": own_model.bse[own_order],
        "t_value": own_model.tvalues[own_order],
        "p_value": own_model.pvalues[own_order],
        "ci_95_lower": own_model.conf_int[own_order, 0],
        "ci_95_upper": own_model.conf_int[own_order, 1],
        "r_squared": np.repeat(own_model.rsquared, len(reference_terms)),
        "mse_resid": np.repeat(own_model.mse_resid, len(reference_terms)),
        "f_statistic": np.repeat(own_model.fvalue, len(reference_terms)),
        "log_likelihood": np.repeat(own_model.llf, len(reference_terms)),
        "aic": np.repeat(own_model.aic, len(reference_terms)),
        "bic": np.repeat(own_model.bic, len(reference_terms)),
    }
    reference_metrics = {
        "coefficient": reference_model.params.to_numpy(),
        "std_err": reference_model.bse.to_numpy(),
        "t_value": reference_model.tvalues.to_numpy(),
        "p_value": reference_model.pvalues.to_numpy(),
        "ci_95_lower": reference_ci.iloc[:, 0].to_numpy(),
        "ci_95_upper": reference_ci.iloc[:, 1].to_numpy(),
        "r_squared": np.repeat(reference_model.rsquared, len(reference_terms)),
        "mse_resid": np.repeat(reference_model.mse_resid, len(reference_terms)),
        "f_statistic": np.repeat(reference_model.fvalue, len(reference_terms)),
        "log_likelihood": np.repeat(reference_model.llf, len(reference_terms)),
        "aic": np.repeat(reference_model.aic, len(reference_terms)),
        "bic": np.repeat(reference_model.bic, len(reference_terms)),
    }

    rows = []
    for metric, own_values in own_metrics.items():
        reference_values = reference_metrics[metric]
        np.testing.assert_allclose(own_values, reference_values, rtol=1e-8, atol=1e-8)
        rows.append(
            {
                "metric": metric,
                "max_absolute_difference": float(np.max(np.abs(own_values - reference_values))),
                "validation": "passed",
            }
        )

    report = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(TABLES_DIR / "statsmodels_validation.csv", index=False)
    return report


if __name__ == "__main__":
    validation_report = validate()
    print("Validación contra statsmodels completada.")
    print(validation_report.to_string(index=False))
