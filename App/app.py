import io
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Minimum Variance Portfolio", layout="wide")

US_LOCALE = "American"
EU_LOCALE = "European"


# =========================================================
# Formatting helpers
# =========================================================
def format_number_full(x, style="American"):
    if pd.isna(x):
        return ""
    if isinstance(x, (np.integer, int)):
        s = str(int(x))
    else:
        s = np.format_float_positional(float(x), unique=False, precision=15, trim="-")
    if style == EU_LOCALE:
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def format_dataframe_for_display(df: pd.DataFrame, style: str) -> pd.DataFrame:
    out = df.copy().astype(object)
    for col in out.columns:
        out[col] = out[col].map(lambda x: format_number_full(x, style))
    return out


# =========================================================
# File reading / parsing
# =========================================================
def detect_csv_style(content: bytes) -> Tuple[str, str, str]:
    sample = content[:4000].decode("utf-8-sig", errors="ignore")
    first_lines = [line for line in sample.splitlines() if line.strip()][:5]
    joined = "\n".join(first_lines)

    semicolons = joined.count(";")
    commas = joined.count(",")

    if semicolons > commas:
        return ";", ",", EU_LOCALE
    return ",", ".", US_LOCALE


def read_uploaded_table(uploaded_file) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        raise ValueError("Please upload a CSV or Excel file.")

    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".csv"):
        sep, decimal, locale_style = detect_csv_style(content)
        df = pd.read_csv(
            io.BytesIO(content),
            sep=sep,
            decimal=decimal,
            encoding="utf-8-sig",
        )
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(content), sheet_name=0)
        locale_style = US_LOCALE
    else:
        raise ValueError("Unsupported file type. Please upload .csv, .xlsx, or .xls")

    if df.empty:
        raise ValueError("Uploaded file is empty.")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).set_index(date_col)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 3:
        raise ValueError("Need at least one risk-free column and at least two asset columns.")

    return df, locale_style


# =========================================================
# Portfolio math
# =========================================================
@dataclass
class OptimizationResult:
    weights: np.ndarray
    success: bool
    message: str
    portfolio_excess_return: float
    portfolio_variance: float
    portfolio_volatility: float
    covariance_matrix: pd.DataFrame
    lagrangian_matrix: pd.DataFrame


def infer_periods_per_year(freq: str) -> int:
    return {"Daily": 252, "Monthly": 12, "Yearly": 1}[freq]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def convert_risk_free_to_periodic(
    rf_series_input: pd.Series,
    periods_per_year: int,
    rf_input_mode: str,
) -> pd.Series:
    if rf_input_mode == "Annual percent (e.g. 5.24)":
        return rf_series_input / (100.0 * periods_per_year)
    elif rf_input_mode == "Annual decimal (e.g. 0.0524)":
        return rf_series_input / periods_per_year
    elif rf_input_mode == "Per-period decimal (e.g. 0.00437 monthly)":
        return rf_series_input.copy()
    else:
        raise ValueError("Unknown risk-free input mode.")


def portfolio_excess_return(weights: np.ndarray, mu_excess: np.ndarray) -> float:
    return float(weights @ mu_excess)


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(weights @ cov @ weights)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, cov)))


def parse_fixed_weights(text: str, asset_names: List[str]) -> Dict[str, float]:
    fixed = {}
    if not text or not text.strip():
        return fixed

    valid_assets = {a.upper(): a for a in asset_names}
    parts = [p.strip() for p in text.split(",") if p.strip()]

    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid fixed-weight entry: '{part}'. Use ASSET=0.10")
        asset_raw, weight_raw = part.split("=", 1)
        asset_key = asset_raw.strip().upper()

        if asset_key not in valid_assets:
            raise ValueError(f"Unknown asset in fixed weights: '{asset_raw.strip()}'")

        fixed[valid_assets[asset_key]] = float(weight_raw.strip())

    return fixed


def build_lagrangian_matrix(
    cov: pd.DataFrame,
    mu: pd.Series,
    include_target_return: bool,
    fixed_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    assets = list(cov.index)
    fixed_weights = fixed_weights or {}

    base = 2.0 * cov.values

    constraint_cols = []
    constraint_rows = []
    row_col_names = []

    # Sum of weights constraint
    constraint_cols.append(-np.ones((len(assets), 1)))
    constraint_rows.append(np.ones((1, len(assets))))
    row_col_names.append("sum_weights")

    # Target return constraint
    if include_target_return:
        constraint_cols.append(-mu.values.reshape(-1, 1))
        constraint_rows.append(mu.values.reshape(1, -1))
        row_col_names.append("target_return")

    # Fixed-weight constraints
    for asset_name in fixed_weights.keys():
        col = np.zeros((len(assets), 1))
        row = np.zeros((1, len(assets)))
        row[0, assets.index(asset_name)] = 1.0
        constraint_cols.append(col)
        constraint_rows.append(row)
        row_col_names.append(f"fix_{asset_name}")

    top_block = np.hstack([base] + constraint_cols)

    m = len(constraint_rows)
    bottom_left = np.vstack(constraint_rows)
    bottom_right = np.zeros((m, m))

    full = np.vstack([
        top_block,
        np.hstack([bottom_left, bottom_right])
    ])

    names = assets + row_col_names
    return pd.DataFrame(full, index=names, columns=names)


def solve_min_variance_with_constraints(
    mu_excess: pd.Series,
    cov: pd.DataFrame,
    target_excess_return: Optional[float] = None,
    fixed_weights: Optional[Dict[str, float]] = None,
    long_only: bool = True,
    use_bank_constraint: bool = False,
) -> OptimizationResult:
    assets = list(mu_excess.index)
    mu = mu_excess.values
    sigma = cov.values
    n = len(assets)

    fixed_weights = fixed_weights or {}
    fixed_sum = sum(fixed_weights.values())
    if fixed_sum > 1 + 1e-12:
        raise ValueError("Fixed weights sum to more than 1.")

    fixed_idx = {assets.index(k): v for k, v in fixed_weights.items()}

    x0 = np.repeat(1.0 / n, n)
    for idx, val in fixed_idx.items():
        x0[idx] = val

    free_idx = [i for i in range(n) if i not in fixed_idx]
    remaining = 1.0 - fixed_sum
    if free_idx:
        free_guess = remaining / len(free_idx)
        for i in free_idx:
            x0[i] = free_guess

    if long_only:
        upper_bound = 0.10 if use_bank_constraint else 1.0
        bounds = [(0.0, upper_bound) for _ in range(n)]
    else:
        upper_bound = 0.10 if use_bank_constraint else 1.0
        bounds = [(-1.0, upper_bound) for _ in range(n)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_excess_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w, tr=target_excess_return: float(w @ mu) - tr}
        )

    for idx, val in fixed_idx.items():
        constraints.append(
            {"type": "eq", "fun": lambda w, i=idx, v=val: w[i] - v}
        )

    result = minimize(
        fun=lambda w: portfolio_variance(w, sigma),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    w = result.x
    if long_only:
        w = np.clip(w, 0, None)

    if np.sum(w) == 0:
        raise ValueError("Optimization returned zero weights.")

    w = w / np.sum(w)

    lagrangian_matrix = build_lagrangian_matrix(
        cov=cov,
        mu=mu_excess,
        include_target_return=target_excess_return is not None,
        fixed_weights=fixed_weights,
    )

    return OptimizationResult(
        weights=w,
        success=bool(result.success),
        message=str(result.message),
        portfolio_excess_return=portfolio_excess_return(w, mu),
        portfolio_variance=portfolio_variance(w, sigma),
        portfolio_volatility=portfolio_volatility(w, sigma),
        covariance_matrix=pd.DataFrame(sigma, index=assets, columns=assets),
        lagrangian_matrix=lagrangian_matrix,
    )


# =========================================================
# Excel export with formulas
# =========================================================
def create_excel_download(
    raw_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    rf_series_input: pd.Series,
    rf_input_mode: str,
    periods_per_year: int,
    log_returns: pd.DataFrame,
    risk_premia: pd.DataFrame,
    avg_risk_premia: pd.Series,
    weights_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    lagrangian_df: pd.DataFrame,
    portfolio_excess_return: float,
    portfolio_variance_value: float,
    portfolio_volatility_value: float,
) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw_export = raw_df.copy()
        raw_export.index.name = "DATE"
        raw_export.to_excel(writer, sheet_name="Raw Data")

        prices_export = prices_df.copy()
        prices_export.index.name = "DATE"
        prices_export.to_excel(writer, sheet_name="Prices")

        weights_df.to_excel(writer, sheet_name="Weights", index=False)
        cov_df.to_excel(writer, sheet_name="Covariance Matrix")
        lagrangian_df.to_excel(writer, sheet_name="Optimized Lagrangian Matrix")

        summary_df = pd.DataFrame({
            "Metric": [
                "Portfolio Excess Return",
                "Portfolio Variance",
                "Portfolio Volatility",
                "Sum of Weights"
            ],
            "Value": [
                portfolio_excess_return,
                portfolio_variance_value,
                portfolio_volatility_value,
                weights_df["Weight"].sum()
            ],
        })
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        wb = writer.book
        ws_raw = wb["Raw Data"]
        ws_prices = wb["Prices"]

        ws_rf = wb.create_sheet("Risk Free Periodic")
        ws_rf["A1"] = "DATE"
        ws_rf["B1"] = "Risk-Free Per Period"

        n_rows = len(raw_df)
        raw_rf_col_idx = raw_df.columns.get_loc(rf_series_input.name) + 2
        raw_rf_col_letter = ws_raw.cell(row=1, column=raw_rf_col_idx).column_letter

        for i in range(n_rows):
            excel_row = i + 2
            ws_rf[f"A{excel_row}"] = f"='Raw Data'!A{excel_row}"

            if rf_input_mode == "Annual percent (e.g. 5.24)":
                ws_rf[f"B{excel_row}"] = f"='Raw Data'!{raw_rf_col_letter}{excel_row}/(100*{periods_per_year})"
            elif rf_input_mode == "Annual decimal (e.g. 0.0524)":
                ws_rf[f"B{excel_row}"] = f"='Raw Data'!{raw_rf_col_letter}{excel_row}/{periods_per_year}"
            else:
                ws_rf[f"B{excel_row}"] = f"='Raw Data'!{raw_rf_col_letter}{excel_row}"

        ws_lr = wb.create_sheet("Log Returns")
        ws_lr["A1"] = "DATE"

        asset_names = list(prices_df.columns)
        for j, asset in enumerate(asset_names, start=2):
            ws_lr.cell(row=1, column=j, value=asset)

        prices_col_letters = {}
        for j, asset in enumerate(asset_names, start=2):
            prices_col_letters[asset] = ws_prices.cell(row=1, column=j).column_letter

        for i in range(2, len(prices_df) + 1):
            price_row = i + 1
            ws_lr[f"A{i}"] = f"='Prices'!A{price_row}"

            for j, asset in enumerate(asset_names, start=2):
                col_letter = prices_col_letters[asset]
                ws_lr.cell(
                    row=i,
                    column=j,
                    value=f"=LN('Prices'!{col_letter}{price_row}/'Prices'!{col_letter}{price_row-1})"
                )

        ws_rp = wb.create_sheet("Risk Premia")
        ws_rp["A1"] = "DATE"
        for j, asset in enumerate(asset_names, start=2):
            ws_rp.cell(row=1, column=j, value=asset)

        for i in range(2, len(prices_df) + 1):
            ws_rp[f"A{i}"] = f"='Log Returns'!A{i}"
            for j, asset in enumerate(asset_names, start=2):
                lr_col_letter = ws_lr.cell(row=1, column=j).column_letter
                ws_rp.cell(
                    row=i,
                    column=j,
                    value=f"='Log Returns'!{lr_col_letter}{i}-'Risk Free Periodic'!B{i+1}"
                )

        ws_avg = wb.create_sheet("Average Risk Premia")
        ws_avg["A1"] = "Asset"
        ws_avg["B1"] = "Average Risk Premium"

        rp_last_row = len(prices_df)
        for i, asset in enumerate(asset_names, start=2):
            rp_col_letter = ws_rp.cell(row=1, column=i).column_letter
            ws_avg[f"A{i}"] = asset
            ws_avg[f"B{i}"] = f"=AVERAGE('Risk Premia'!{rp_col_letter}2:{rp_col_letter}{rp_last_row})"

        ws_summary = wb["Summary"]
        ws_summary["B5"] = f"=SUM(Weights!B2:B{len(weights_df)+1})"

    output.seek(0)
    return output.getvalue()


# =========================================================
# UI
# =========================================================
st.title("Minimum Variance Portfolio from Risk Premia")
st.caption("Upload price data, convert to log returns and excess returns, and solve a constrained minimum variance portfolio.")

with st.sidebar:
    st.header("Inputs")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    frequency = st.selectbox("Data frequency", ["Daily", "Monthly", "Yearly"], index=1)
    rf_input_mode = st.selectbox(
        "Risk-free input format",
        [
            "Annual percent (e.g. 5.24)",
            "Annual decimal (e.g. 0.0524)",
            "Per-period decimal (e.g. 0.00437 monthly)"
        ],
        index=0,
    )

if uploaded_file is not None:
    try:
        raw_df, detected_locale = read_uploaded_table(uploaded_file)

        st.sidebar.write(f"Detected format: {detected_locale}")

        all_cols = list(raw_df.columns)
        default_rf = all_cols[-1]
        default_assets = [c for c in all_cols if c != default_rf]

        risk_free_col = st.sidebar.selectbox("Risk-free column", all_cols, index=all_cols.index(default_rf))
        asset_cols = st.sidebar.multiselect("Asset columns", all_cols, default=default_assets)

        if risk_free_col in asset_cols:
            st.error("Risk-free column cannot also be an asset column.")
            st.stop()

        if len(asset_cols) < 2:
            st.error("Please select at least two assets.")
            st.stop()

        locale_style = st.sidebar.selectbox("Display format", [detected_locale, US_LOCALE, EU_LOCALE], index=0)

        periods = infer_periods_per_year(frequency)

        prices = raw_df[asset_cols].copy().dropna(how="all")
        rf_series_input = raw_df[risk_free_col].copy()

        asset_log_returns = compute_log_returns(prices)

        rf_periodic = convert_risk_free_to_periodic(
            rf_series_input=rf_series_input,
            periods_per_year=periods,
            rf_input_mode=rf_input_mode,
        )
        rf_periodic = rf_periodic.reindex(asset_log_returns.index)

        risk_premia = asset_log_returns.sub(rf_periodic, axis=0).dropna(how="any")

        mean_risk_premia = risk_premia.mean()
        covariance_matrix = risk_premia.cov()

        with st.sidebar:
            st.header("Constraints")
            use_target = st.checkbox("Set target portfolio excess return", value=False)
            target_excess_return = None
            if use_target:
                target_excess_return = st.number_input(
                    "Target excess return per period",
                    min_value=-1.0,
                    max_value=2.0,
                    value=float(mean_risk_premia.mean()),
                    step=0.000001,
                    format="%.15f",
                )

            use_fixed_weights = st.checkbox("Fix one or more stock weights", value=False)
            fixed_weights_text = ""
            if use_fixed_weights:
                st.caption("Format: APPLE=0.10, WALMART=0.15")
                fixed_weights_text = st.text_input("Fixed weights", value="")

            long_only = st.checkbox("Long-only (weights ≥ 0)", value=True)
            use_bank_constraint = st.checkbox("Bank constraint: max 10% per position", value=False)

        fixed_weights = parse_fixed_weights(fixed_weights_text, asset_cols) if use_fixed_weights else {}

        result = solve_min_variance_with_constraints(
            mu_excess=mean_risk_premia,
            cov=covariance_matrix,
            target_excess_return=target_excess_return if use_target else None,
            fixed_weights=fixed_weights,
            long_only=long_only,
            use_bank_constraint=use_bank_constraint,
        )

        weights_df = pd.DataFrame({
            "Asset": asset_cols,
            "Weight": result.weights,
            "Mean Risk Premium": mean_risk_premia.values,
        }).sort_values("Weight", ascending=False).reset_index(drop=True)

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Excess Return", format_number_full(result.portfolio_excess_return, locale_style))
        m2.metric("Variance", format_number_full(result.portfolio_variance, locale_style))
        m3.metric("Volatility", format_number_full(result.portfolio_volatility, locale_style))
        m4.metric("Sum of Weights", format_number_full(float(np.sum(result.weights)), locale_style))

        if result.success:
            st.success("Optimization solved successfully.")
        else:
            st.warning(f"Solver warning: {result.message}")

        chart_col, table_col = st.columns([1.1, 1])

        with chart_col:
            st.subheader("Portfolio Weights")
            st.bar_chart(pd.DataFrame({"Weight": result.weights}, index=asset_cols))

        with table_col:
            st.subheader("Weights Table")
            display_weights = weights_df.copy()
            display_weights["Weight"] = display_weights["Weight"].map(lambda x: format_number_full(x, locale_style))
            display_weights["Mean Risk Premium"] = display_weights["Mean Risk Premium"].map(lambda x: format_number_full(x, locale_style))
            st.dataframe(display_weights, use_container_width=True)

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Risk", "Data", "Download"])

        with tab1:
            st.subheader("Average Excess Returns")
            avg_df = pd.DataFrame({
                "Asset": mean_risk_premia.index,
                "Average Risk Premium": [format_number_full(x, locale_style) for x in mean_risk_premia.values]
            })
            st.dataframe(avg_df, use_container_width=True)

        with tab2:
            st.subheader("Covariance Matrix")
            cov_display = format_dataframe_for_display(result.covariance_matrix, locale_style)
            st.dataframe(cov_display, use_container_width=True)

            st.subheader("Optimized Lagrangian Matrix")
            lag_display = format_dataframe_for_display(result.lagrangian_matrix, locale_style)
            st.dataframe(lag_display, use_container_width=True)

        with tab3:
            st.subheader("Input Data")
            st.dataframe(raw_df, use_container_width=True)

            st.subheader("Converted Risk-Free Rate")
            rf_used_df = pd.DataFrame({"Risk-Free Per Period": rf_periodic})
            st.dataframe(rf_used_df, use_container_width=True)

            st.subheader("Log Returns")
            st.dataframe(asset_log_returns, use_container_width=True)

            st.subheader("Risk Premia")
            st.dataframe(risk_premia, use_container_width=True)

        with tab4:
            st.subheader("Download Results")
            st.write("Export the workbook with formulas and optimization outputs.")

            try:
                excel_data = create_excel_download(
                    raw_df=raw_df,
                    prices_df=prices,
                    rf_series_input=rf_series_input,
                    rf_input_mode=rf_input_mode,
                    periods_per_year=periods,
                    log_returns=asset_log_returns,
                    risk_premia=risk_premia,
                    avg_risk_premia=mean_risk_premia,
                    weights_df=weights_df,
                    cov_df=result.covariance_matrix,
                    lagrangian_df=result.lagrangian_matrix,
                    portfolio_excess_return=result.portfolio_excess_return,
                    portfolio_variance_value=result.portfolio_variance,
                    portfolio_volatility_value=result.portfolio_volatility,
                )

                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name="portfolio_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as export_error:
                st.error(
                    "Excel export failed. Most likely the Excel writer package is missing. "
                    "Run: py -m pip install openpyxl"
                )
                st.code(str(export_error))
    except Exception as e:
        st.error(f"Error: {e}")
