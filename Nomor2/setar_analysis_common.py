"""Utilitas bersama bagian iv-vii; metode solver tetap di modul masing-masing."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import iii_normal_equation as normal_equation
import iv_qr_householder as householder
from i_formulate_overdetermined_matrix import create_matrix_SETAR, return_price


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "hasil"
PARAMETERS = ("alpha1", "phi11", "phi12", "alpha2", "phi21", "phi22")


@dataclass
class SetarData:
    train_dates: list[str]
    test_dates: list[str]
    train_prices: np.ndarray
    test_prices: np.ndarray
    train_returns: np.ndarray
    test_returns: np.ndarray
    A: np.ndarray
    b: np.ndarray
    train_target_dates: list[str]


@dataclass
class ModelResults:
    coefficients: dict[str, np.ndarray]
    train_predictions: dict[str, np.ndarray]
    test_predictions: dict[str, np.ndarray]
    residuals: dict[str, float]
    train_rmse: dict[str, float]
    test_rmse: dict[str, float]
    condition_A: float
    condition_AtA: float


def _read_prices(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"Date", "Close"}.issubset(rows[0]):
        raise ValueError(f"CSV harus memiliki kolom Date dan Close: {path}")
    dates = [row["Date"] for row in rows]
    prices = np.asarray([float(row["Close"]) for row in rows], dtype=float)
    if not np.all(np.isfinite(prices)) or np.any(prices == 0.0):
        raise ValueError(f"Harga tidak valid atau nol: {path}")
    return dates, prices


def load_data() -> SetarData:
    train_dates, train_prices = _read_prices(HERE / "stock_train.csv")
    test_dates, test_prices = _read_prices(HERE / "stock_test.csv")
    if len(train_prices) < 4 or len(test_prices) < 1:
        raise ValueError("Dataset train/test terlalu pendek untuk model SETAR lag 2")
    train_returns = return_price(train_prices)
    # Target pertama test adalah return lintas batas dari penutupan train ke test.
    boundary_return = (test_prices[0] - train_prices[-1]) / train_prices[-1]
    test_returns = np.concatenate(([boundary_return], return_price(test_prices)))
    A, b = create_matrix_SETAR(train_returns)
    # Baris pertama A memodelkan return indeks 2, yakni perubahan menuju harga ke-4.
    train_target_dates = train_dates[3:]
    if len(train_target_dates) != len(b):
        raise ValueError("Tanggal target train tidak selaras dengan matriks desain")
    if len(test_dates) != len(test_returns):
        raise ValueError("Tanggal target test tidak selaras dengan return lintas batas")
    return SetarData(
        train_dates, test_dates, train_prices, test_prices,
        train_returns, test_returns, A, b, train_target_dates,
    )


def predict_row(r1: float, r2: float, x: np.ndarray) -> float:
    if r1 >= 0.0:
        return float(x[0] + x[1] * r1 + x[2] * r2)
    return float(x[3] + x[4] * r1 + x[5] * r2)


def predict_train(returns: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.asarray([
        predict_row(returns[i - 1], returns[i - 2], x)
        for i in range(2, len(returns))
    ])


def predict_test(train_returns: np.ndarray, test_returns: np.ndarray, x: np.ndarray) -> np.ndarray:
    # One-step ahead: setiap lag setelah target pertama memakai return aktual sebelumnya.
    context = list(train_returns[-2:])
    predictions: list[float] = []
    for actual in test_returns:
        predictions.append(predict_row(context[-1], context[-2], x))
        context.append(float(actual))
    return np.asarray(predictions)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Vektor aktual dan prediksi harus sama ukuran dan tidak kosong")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def fit_models(data: SetarData) -> ModelResults:
    x_qr, _, _ = householder.householder_lstsq(data.A, data.b)
    x_normal = normal_equation.solve_normal_equations(data.A, data.b)
    coefficients = {"QR Householder": x_qr, "Persamaan Normal": x_normal}
    train_predictions = {
        name: predict_train(data.train_returns, x) for name, x in coefficients.items()
    }
    test_predictions = {
        name: predict_test(data.train_returns, data.test_returns, x)
        for name, x in coefficients.items()
    }
    residuals = {
        name: float(np.linalg.norm(data.A @ x - data.b))
        for name, x in coefficients.items()
    }
    train_rmse = {
        name: rmse(data.b, train_predictions[name]) for name in coefficients
    }
    test_rmse = {
        name: rmse(data.test_returns, test_predictions[name]) for name in coefficients
    }
    condition_A, condition_AtA = normal_equation.calculate_condition_numbers(data.A)
    return ModelResults(
        coefficients, train_predictions, test_predictions, residuals,
        train_rmse, test_rmse, float(condition_A), float(condition_AtA),
    )


def select_best_method(results: ModelResults) -> str:
    """Pilih RMSE test terendah; untuk hasil imbang numerik, utamakan QR."""
    qr = results.test_rmse["QR Householder"]
    normal = results.test_rmse["Persamaan Normal"]
    tolerance = 1e-12 * max(1.0, abs(qr), abs(normal))
    if abs(qr - normal) <= tolerance:
        return "QR Householder"
    return "QR Householder" if qr < normal else "Persamaan Normal"


def write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def coefficient_rows(results: ModelResults):
    return zip(
        PARAMETERS,
        results.coefficients["QR Householder"],
        results.coefficients["Persamaan Normal"],
    )
