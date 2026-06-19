
import io
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize


# =========================================================
# PAGE CONFIGURATION
# =========================================================
# This controls the browser tab title and page layout.
# Wide layout is better for dashboards, charts, and large matrices.
st.set_page_config(
    page_title="Minimum Variance Portfolio | Piraeus Bank",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CONSTANTS
# =========================================================
# These are used for displaying numbers in either American or European format.
US_LOCALE = "American"
EU_LOCALE = "European"


# =========================================================
# PIRAEUS BANK STYLE / CUSTOM CSS
# =========================================================
# This makes the app look more professional and closer to
# a Piraeus Bank-style internal dashboard.
#
# IMPORTANT:
# The sidebar has a dark teal background, so general labels are white.
# However, input boxes, dropdowns, and number fields need black text
# and white backgrounds so they do not become invisible.
st.markdown(
    """
    <style>
    :root {
        --piraeus-teal: #002F30;
        --piraeus-teal-soft: #06484A;
        --piraeus-yellow: #FFD900;
        --piraeus-cream: #F7F5EF;
        --piraeus-card: #FFFFFF;
        --piraeus-muted: #6D7A7A;
        --piraeus-border: #E4E0D5;
    }

    .stApp {
        background: linear-gradient(135deg, #F7F5EF 0%, #EFEBDD 100%);
        color: var(--piraeus-teal);
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar old color scheme */
    section[data-testid="stSidebar"] {
        background: var(--piraeus-teal);
        border-right: 4px solid var(--piraeus-yellow);
    }

    /* Sidebar normal text stays white */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    /* Inputs are white boxes with black text */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Selectbox and dropdown text */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #000000 !important;
    }

    /* File uploader */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] section * {
        color: #000000 !important;
    }

    /* Multiselect selected pills */
    section[data-testid="stSidebar"] .stMultiSelect span {
        color: #000000 !important;
    }

    /* Checkbox labels stay white */
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCheckbox p {
        color: #FFFFFF !important;
    }

    .hero {
        background: linear-gradient(135deg, var(--piraeus-teal) 0%, var(--piraeus-teal-soft) 100%);
        border-radius: 24px;
        padding: 34px 38px;
        box-shadow: 0 18px 45px rgba(0, 47, 48, 0.22);
        border-bottom: 8px solid var(--piraeus-yellow);
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: var(--piraeus-yellow);
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #FFFFFF !important;
        font-size: 44px;
        line-height: 1.05;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #DDEAEA;
        font-size: 17px;
        max-width: 950px;
        line-height: 1.55;
    }

    .info-box {
        background: #FFFFFF;
        border-left: 6px solid var(--piraeus-yellow);
        border-radius: 16px;
        padding: 18px 20px;
        color: var(--piraeus-teal);
        box-shadow: 0 8px 22px rgba(0, 47, 48, 0.07);
        margin-bottom: 20px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--piraeus-border);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 8px 22px rgba(0, 47, 48, 0.07);
    }

    div[data-testid="stMetricValue"] {
        color: var(--piraeus-teal) !important;
        font-weight: 900;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: var(--piraeus-yellow) !important;
        color: var(--piraeus-teal) !important;
        border: 0 !important;
        border-radius: 14px !important;
        font-weight: 850 !important;
    }

    h1, h2, h3, h4 {
        color: var(--piraeus-teal) !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid var(--piraeus-border);
    }

    .small-muted {
        color: #DDEAEA;
        font-size: 13px;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# FORMATTING HELPERS
# =========================================================
# These functions control how numbers are displayed.
# European format changes decimal points into commas.
def format_number_full(x, style="American"):
    """Format a numeric value into a string according to `style`.

    - Returns empty string for NaN values.
    - Preserves integer formatting for integer-like inputs.
    - Uses European decimal/comma swapping when `style==EU_LOCALE`.
    """

    # Handle missing values first.
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
    """Return a copy of `df` with all values converted to formatted strings.

    This is useful for rendering numeric tables in the Streamlit UI
    using the selected localization style.
    """
    out = df.copy().astype(object)
    for col in out.columns:
        out[col] = out[col].map(lambda x: format_number_full(x, style))
    return out


def pct_fmt(x):
    return f"{x:.2%}"


def money_fmt(x):
    return f"{x:,.2f}"


# =========================================================
# FILE READING / PARSING
# =========================================================
# These functions read uploaded CSV or Excel files.
# The CSV detector tries to determine whether the user is using
# American formatting or European formatting.
def detect_csv_style(content: bytes) -> Tuple[str, str, str]:
    """Heuristically detect the CSV delimiter and decimal separator.

    Returns a tuple: (sep, decimal, locale_style) where `locale_style`
    is either the US or European constant used elsewhere in the app.
    """

    sample = content[:4000].decode("utf-8-sig", errors="ignore")
    first_lines = [line for line in sample.splitlines() if line.strip()][:5]
    joined = "\n".join(first_lines)

    semicolons = joined.count(";")
    commas = joined.count(",")

    # European CSV files often use semicolons as separators and commas as decimals.
    if semicolons > commas:
        return ";", ",", EU_LOCALE

    return ",", ".", US_LOCALE


def read_uploaded_table(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Read an uploaded CSV or Excel file into a cleaned DataFrame.

    Returns the dataframe and a detected locale style string.
    Expects the first column to be a date column and coerces other
    columns to numeric where possible.
    """

    if uploaded_file is None:
        raise ValueError("Please upload a CSV or Excel file.")

    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    # Read CSV or Excel depending on file extension.
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

    # Assumes the first column is the date column.
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[date_col]).set_index(date_col)

    # Convert every other column into numeric values.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove columns that are completely empty.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 3:
        raise ValueError("Need at least one risk-free column and at least two asset columns.")

    return df, locale_style


# =========================================================
# PORTFOLIO MATH
# =========================================================
@dataclass
class OptimizationResult:
    """Container for optimization results and diagnostic matrices.

    Holds the final weights, solver status/message, portfolio statistics,
    and the covariance / Lagrangian matrices produced for inspection.
    """
    weights: np.ndarray
    success: bool
    message: str
    portfolio_excess_return: float
    portfolio_variance: float
    portfolio_volatility: float
    covariance_matrix: pd.DataFrame
    lagrangian_matrix: pd.DataFrame


def infer_periods_per_year(freq: str) -> int:
    # Used to convert annual risk-free rate into per-period risk-free rate.
    return {"Daily": 252, "Monthly": 12, "Yearly": 1}[freq]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price dataframe.

    Uses period-over-period natural log differences and drops rows
    that are entirely NA (e.g. first observation).
    """

    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def convert_risk_free_to_periodic(
    rf_series_input: pd.Series,
    periods_per_year: int,
    rf_input_mode: str,
) -> pd.Series:
    # Converts the risk-free rate into the same frequency as the return data.
    """Convert various risk-free input formats into per-period decimals.

    Supports annual percent, annual decimal, and already per-period decimal.
    """
    if rf_input_mode == "Annual percent (e.g. 5.24)":
        return rf_series_input / (100.0 * periods_per_year)
    elif rf_input_mode == "Annual decimal (e.g. 0.0524)":
        return rf_series_input / periods_per_year
    elif rf_input_mode == "Per-period decimal (e.g. 0.00437 monthly)":
        return rf_series_input.copy()
    else:
        raise ValueError("Unknown risk-free input mode.")


def portfolio_excess_return(weights: np.ndarray, mu_excess: np.ndarray) -> float:
    # Portfolio expected excess return = weights dot average risk premia.
    return float(weights @ mu_excess)


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    # Portfolio variance = w' Σ w.
    return float(weights @ cov @ weights)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    # Portfolio volatility = square root of variance.
    return float(np.sqrt(portfolio_variance(weights, cov)))


def parse_fixed_weights(text: str, asset_names: List[str]) -> Dict[str, float]:
    # Allows the user to manually fix certain weights.
    # Example: AAPL=0.10, MSFT=0.15
    fixed = {}
    if not text or not text.strip(): return fixed

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
    """Construct the augmented Lagrangian matrix for transparency.

    This returns a square DataFrame where the upper-left block is
    2*Covariance, and the bottom/right blocks encode equality
    constraints (sum-to-one, optional target return, and fixed weights).
    """
    assets = list(cov.index)
    fixed_weights = fixed_weights or {}

    # The top-left part is 2 times the covariance matrix.
    base = 2.0 * cov.values

    constraint_cols = []
    constraint_rows = []
    row_col_names = []

    # Constraint: weights must sum to 1.
    constraint_cols.append(-np.ones((len(assets), 1)))
    constraint_rows.append(np.ones((1, len(assets))))
    row_col_names.append("sum_weights")

    # Optional constraint: portfolio must hit a target excess return.
    if include_target_return:
        constraint_cols.append(-mu.values.reshape(-1, 1))
        constraint_rows.append(mu.values.reshape(1, -1))
        row_col_names.append("target_return")

    # Optional fixed-weight constraints.
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
    """Solve the minimum-variance portfolio under provided constraints.

    Uses SLSQP to minimize portfolio variance with equality constraints
    for sum-to-one, optional target return, and optional fixed weights.
    Supports long-only and an optional per-position cap (bank constraint).
    Returns an `OptimizationResult` with diagnostics and matrices.
    """

    assets = list(mu_excess.index)
    mu = mu_excess.values
    sigma = cov.values
    n = len(assets)

    fixed_weights = fixed_weights or {}
    fixed_sum = sum(fixed_weights.values())

    if fixed_sum > 1 + 1e-12:
        raise ValueError("Fixed weights sum to more than 1.")

    fixed_idx = {assets.index(k): v for k, v in fixed_weights.items()}

    # Start with equal weights.
    x0 = np.repeat(1.0 / n, n)

    # Apply fixed weights to the initial guess.
    for idx, val in fixed_idx.items():
        x0[idx] = val

    # Distribute the remaining weight across free assets.
    free_idx = [i for i in range(n) if i not in fixed_idx]
    remaining = 1.0 - fixed_sum

    if free_idx:
        free_guess = remaining / len(free_idx)
        for i in free_idx:
            x0[i] = free_guess

    # Bounds control the minimum and maximum weight allowed for each asset.
    if long_only:
        upper_bound = 0.10 if use_bank_constraint else 1.0
        bounds = [(0.0, upper_bound) for _ in range(n)]
    else:
        upper_bound = 0.10 if use_bank_constraint else 1.0
        bounds = [(-1.0, upper_bound) for _ in range(n)]

    # Constraint: all weights must sum to 1.
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Optional target return constraint.
    if target_excess_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w, tr=target_excess_return: float(w @ mu) - tr}
        )

    # Optional fixed weight constraints.
    for idx, val in fixed_idx.items():
        constraints.append(
            {"type": "eq", "fun": lambda w, i=idx, v=val: w[i] - v}
        )

    # Run scipy's SLSQP optimizer.
    result = minimize(
        fun=lambda w: portfolio_variance(w, sigma),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    w = result.x

    # Clean up tiny negative values if long-only.
    if long_only:
        w = np.clip(w, 0, None)

    if np.sum(w) == 0:
        raise ValueError("Optimization returned zero weights.")

    # Normalize weights to ensure they sum to 1.
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
# EXCEL EXPORT WITH FORMULAS
# =========================================================
# This function creates an Excel workbook in memory.
# It exports raw data, prices, weights, covariance matrix,
# Lagrangian matrix, risk-free conversion, log returns,
# risk premia, and summary outputs.
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
    """Create an in-memory Excel workbook containing inputs, formulas,
    and results suitable for downloading from the Streamlit UI.

    Returns the workbook bytes.
    """

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

        # Risk-free conversion sheet.
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

        # Log returns sheet.
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

        # Risk premia sheet.
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

        # Average risk premia sheet.
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
# MAIN APP HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Piraeus Bank • Portfolio Optimization Tool</div>
        <div class="hero-title">Minimum Variance Portfolio Dashboard</div>
        <div class="hero-subtitle">
            Upload price data, convert prices into log returns and risk premia, then solve a constrained
            minimum variance portfolio with optional target return, fixed weights, long-only settings,
            and a 10% bank-style position limit.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR INPUTS
# =========================================================
with st.sidebar:
    st.markdown("### Data Inputs")
    st.markdown(
        "<p class='small-muted'>Upload a CSV or Excel file. The first column should be dates. Other columns should be asset prices and one risk-free rate column.</p>",
        unsafe_allow_html=True,
    )

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


# =========================================================
# EMPTY STATE
# =========================================================
if uploaded_file is None:
    st.markdown(
        """
        <div class="info-box">
            <b>Start here:</b> upload your CSV or Excel file from the sidebar.
            The app will calculate log returns, risk premia, covariance matrix, optimized weights,
            and an exportable Excel workbook.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Minimum Variance")
    c2.metric("Default Frequency", "Monthly")
    c3.metric("Export", "Excel Workbook")

    st.stop()


# =========================================================
# MAIN APP LOGIC
# =========================================================
try:
    # Read the uploaded file and detect number format.
    raw_df, detected_locale = read_uploaded_table(uploaded_file)

    st.markdown(
        "<span class='status-pill'>File uploaded successfully</span>",
        unsafe_allow_html=True,
    )

    st.sidebar.write(f"Detected format: {detected_locale}")

    # Column selection.
    all_cols = list(raw_df.columns)
    default_rf = all_cols[-1]
    default_assets = [c for c in all_cols if c != default_rf]

    risk_free_col = st.sidebar.selectbox(
        "Risk-free column",
        all_cols,
        index=all_cols.index(default_rf)
    )

    asset_cols = st.sidebar.multiselect(
        "Asset columns",
        all_cols,
        default=default_assets
    )

    if risk_free_col in asset_cols:
        st.error("Risk-free column cannot also be an asset column.")
        st.stop()

    if len(asset_cols) < 2:
        st.error("Please select at least two assets.")
        st.stop()

    locale_style = st.sidebar.selectbox(
        "Display format",
        [detected_locale, US_LOCALE, EU_LOCALE],
        index=0
    )

    periods = infer_periods_per_year(frequency)

    # Separate asset prices and risk-free column.
    prices = raw_df[asset_cols].copy().dropna(how="all")
    rf_series_input = raw_df[risk_free_col].copy()

    # Calculate log returns.
    asset_log_returns = compute_log_returns(prices)

    # Convert risk-free rate into per-period value.
    rf_periodic = convert_risk_free_to_periodic(
        rf_series_input=rf_series_input,
        periods_per_year=periods,
        rf_input_mode=rf_input_mode,
    )

    # Align risk-free rate with return dates.
    rf_periodic = rf_periodic.reindex(asset_log_returns.index)

    # Risk premia = asset log returns minus risk-free rate.
    risk_premia = asset_log_returns.sub(rf_periodic, axis=0).dropna(how="any")

    # Average risk premia and covariance matrix are the two main optimization inputs.
    mean_risk_premia = risk_premia.mean()
    covariance_matrix = risk_premia.cov()

    # Constraint controls appear only after the file is loaded.
    with st.sidebar:
        st.divider()
        st.markdown("### Constraints")

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

        investment_amount = st.number_input(
            "Investment Amount",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            format="%.2f",
        )

    # Convert fixed weights from text into dictionary.
    fixed_weights = parse_fixed_weights(fixed_weights_text, asset_cols) if use_fixed_weights else {}

    # Run optimization.
    result = solve_min_variance_with_constraints(
        mu_excess=mean_risk_premia,
        cov=covariance_matrix,
        target_excess_return=target_excess_return if use_target else None,
        fixed_weights=fixed_weights,
        long_only=long_only,
        use_bank_constraint=use_bank_constraint,
    )

    # Build final weights table.
    weights_df = pd.DataFrame({
        "Asset": asset_cols,
        "Weight": result.weights,
        "Allocation Amount": result.weights * investment_amount,
        "Mean Risk Premium": mean_risk_premia.values,
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    # =====================================================
    # EXECUTIVE SUMMARY METRICS
    # =====================================================
    st.markdown("### Executive Summary")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Excess Return", format_number_full(result.portfolio_excess_return, locale_style))
    m2.metric("Variance", format_number_full(result.portfolio_variance, locale_style))
    m3.metric("Volatility", format_number_full(result.portfolio_volatility, locale_style))
    m4.metric("Sum of Weights", format_number_full(float(np.sum(result.weights)), locale_style))

    if result.success:
        st.success("Optimization solved successfully.")
    else:
        st.warning(f"Solver warning: {result.message}")

    # =====================================================
    # MAIN WEIGHTS SECTION
    # =====================================================
    st.markdown("### Portfolio Allocation")

    chart_col, table_col = st.columns([1.1, 1])

    with chart_col:
        st.markdown(
            """
            <div class="section-title">Portfolio Weights</div>
            <div class="section-subtitle">Visual allocation across selected assets.</div>
            """,
            unsafe_allow_html=True,
        )
        chart_df = pd.DataFrame({"Weight": result.weights}, index=asset_cols)
        st.bar_chart(chart_df, height=360)

    with table_col:
        st.markdown(
            """
            <div class="section-title">Weights Table</div>
            <div class="section-subtitle">Optimized weight and investment amount per asset.</div>
            """,
            unsafe_allow_html=True,
        )

        display_weights = weights_df.copy()
        display_weights["Weight"] = display_weights["Weight"].map(lambda x: format_number_full(x, locale_style))
        display_weights["Allocation Amount"] = display_weights["Allocation Amount"].map(lambda x: format_number_full(x, locale_style))
        display_weights["Mean Risk Premium"] = display_weights["Mean Risk Premium"].map(lambda x: format_number_full(x, locale_style))

        st.dataframe(display_weights, use_container_width=True, height=360)

    # =====================================================
    # DETAILED OUTPUT TABS
    # =====================================================
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Risk", "Data", "Download"])

    with tab1:
        st.markdown(
            """
            <div class="section-title">Average Excess Returns</div>
            <div class="section-subtitle">Mean risk premium for each selected asset.</div>
            """,
            unsafe_allow_html=True,
        )

        avg_df = pd.DataFrame({
            "Asset": mean_risk_premia.index,
            "Average Risk Premium": [format_number_full(x, locale_style) for x in mean_risk_premia.values]
        })

        st.dataframe(avg_df, use_container_width=True)

    with tab2:
        st.markdown(
            """
            <div class="section-title">Covariance Matrix</div>
            <div class="section-subtitle">Measures how asset risk premia move together.</div>
            """,
            unsafe_allow_html=True,
        )

        cov_display = format_dataframe_for_display(result.covariance_matrix, locale_style)
        st.dataframe(cov_display, use_container_width=True)

        st.markdown(
            """
            <div class="section-title">Optimized Lagrangian Matrix</div>
            <div class="section-subtitle">Shows the constraint structure used in the optimization.</div>
            """,
            unsafe_allow_html=True,
        )

        lag_display = format_dataframe_for_display(result.lagrangian_matrix, locale_style)
        st.dataframe(lag_display, use_container_width=True)

    with tab3:
        st.markdown(
            """
            <div class="section-title">Input and Calculation Data</div>
            <div class="section-subtitle">Raw data, converted risk-free rate, log returns, and risk premia.</div>
            """,
            unsafe_allow_html=True,
        )

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
        st.markdown(
            """
            <div class="download-box">
                <h3>Download Portfolio Results</h3>
                <p>Export the workbook with formulas, risk premia, covariance matrix, Lagrangian matrix, and final optimization outputs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                use_container_width=True,
            )

        except Exception as export_error:
            st.error(
                "Excel export failed. Most likely the Excel writer package is missing. "
                "Run: py -m pip install openpyxl"
            )
            st.code(str(export_error))

except Exception as e:
    # Shows a clear error if the uploaded file or optimization settings cause a problem.
    st.error(f"Error: {e}")


# To run this app:
#  python -m streamlit run app.py



