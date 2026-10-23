"""Pruebas de integración de los artefactos generados por la canalización."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src import analysis


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = analysis.run_pipeline()

    def test_all_expected_artifacts_are_generated(self) -> None:
        expected_tables = {
            "data_dictionary.csv",
            "data_cleaning_audit.csv",
            "full_model_coefficients.csv",
            "full_model_metrics.csv",
            "two_predictor_models.csv",
            "significant_terms_5pct.csv",
            "categorical_variables_joint_significance.csv",
            "top_two_numeric_predictors_by_abs_t.csv",
            "all_coefficients_by_abs_t.csv",
            "top_two_coefficients_by_abs_t.csv",
            "selected_predictor_ci_95.csv",
            "breusch_pagan.csv",
            "full_model_summary.txt",
            "run_summary.json",
        }
        expected_figures = {
            "price_vs_carat.png",
            "residuals_vs_fitted.png",
            "residuals_qq.png",
        }
        self.assertTrue(all((analysis.TABLES_DIR / name).is_file() for name in expected_tables))
        self.assertTrue(all((analysis.FIGURES_DIR / name).is_file() for name in expected_figures))

    def test_all_36_source_variable_pairs_are_compared(self) -> None:
        pairs = pd.read_csv(analysis.TABLES_DIR / "two_predictor_models.csv")
        self.assertEqual(len(pairs), 36)
        self.assertEqual(len(pairs[["predictor_1", "predictor_2"]].drop_duplicates()), 36)
        categorical = {"cut", "color", "clarity"}
        self.assertTrue(categorical.issubset(set(pairs["predictor_1"]).union(pairs["predictor_2"])))

    def test_selected_interval_matches_top_numeric_predictor(self) -> None:
        top_two = pd.read_csv(analysis.TABLES_DIR / "top_two_numeric_predictors_by_abs_t.csv")
        selected = pd.read_csv(analysis.TABLES_DIR / "selected_predictor_ci_95.csv")
        self.assertEqual(len(top_two), 2)
        self.assertEqual(selected.loc[0, "termino"], top_two.loc[0, "termino"])
        self.assertAlmostEqual(selected.loc[0, "ci_95_inf"], top_two.loc[0, "ci_95_inf"])
        self.assertAlmostEqual(selected.loc[0, "ci_95_sup"], top_two.loc[0, "ci_95_sup"])

    def test_global_coefficient_ranking_includes_categorical_indicators(self) -> None:
        top_two = pd.read_csv(analysis.TABLES_DIR / "top_two_coefficients_by_abs_t.csv")
        self.assertEqual(top_two.loc[0, "termino"], "carat")
        self.assertEqual(top_two.loc[1, "termino"], "C(clarity)[T.VVS2]")
        self.assertEqual(top_two.loc[1, "referencia"], "I1")

    def test_summary_identifies_best_pairs_from_generated_table(self) -> None:
        pairs = pd.read_csv(analysis.TABLES_DIR / "two_predictor_models.csv")
        with (analysis.TABLES_DIR / "run_summary.json").open(encoding="utf-8") as file:
            stored_summary = json.load(file)
        self.assertEqual(stored_summary, self.summary)
        self.assertEqual(stored_summary["best_pair_by_aic"], pairs.iloc[0][["predictor_1", "predictor_2"]].to_dict())
        best_r_squared = pairs.sort_values("r_squared", ascending=False, kind="stable").iloc[0]
        self.assertEqual(
            stored_summary["best_pair_by_r_squared"],
            best_r_squared[["predictor_1", "predictor_2"]].to_dict(),
        )


if __name__ == "__main__":
    unittest.main()
