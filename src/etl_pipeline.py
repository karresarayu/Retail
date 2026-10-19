
from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "online_retail_II.xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retail_sales_final.csv"
)


def load_raw_data():
    """Load and combine the two sheets from the raw Excel dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}"
        )

    sheet_2009 = pd.read_excel(
        RAW_FILE, sheet_name="Year 2009-2010"
    )

    sheet_2010 = pd.read_excel(
        RAW_FILE, sheet_name="Year 2010-2011"
    )

    df = pd.concat(
        [sheet_2009, sheet_2010],
        ignore_index=True
    )

    print("Raw data loaded successfully.")
    print("Combined dataset shape:", df.shape)

    return df





def clean_data(df):
    """Clean and prepare the combined retail dataset."""

    # Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Convert columns to appropriate data types
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )

    # Remove rows with invalid essential values
    df = df.dropna(
        subset=["Invoice", "StockCode", "InvoiceDate",
                "Quantity", "Price"]
    ).copy()

    # Create revenue and quality flags
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Is_Return"] = df["Quantity"] < 0
    df["Is_Zero_Price"] = df["Price"] == 0
    df["Missing_Description"] = df["Description"].isna()

    print("Cleaning completed.")
    print("Rows after cleaning:", len(df))
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing descriptions:", df["Description"].isna().sum())

    return df


def validate_data(df):
    """Validate the cleaned dataset before saving."""

    required_columns = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Revenue",
        "Is_Return",
        "Is_Zero_Price",
        "Missing_Description",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.duplicated().any():
        raise ValueError("Duplicate rows detected.")

    if df["InvoiceDate"].isna().any():
        raise ValueError("Invalid invoice dates detected.")

    if (df["Price"] < 0).any():
        raise ValueError("Negative prices detected.")

    # Verify quality flags
    assert (
        df["Is_Return"] == (df["Quantity"] < 0)
    ).all(), "Return flag mismatch."

    assert (
        df["Is_Zero_Price"] == (df["Price"] == 0)
    ).all(), "Zero-price flag mismatch."

    assert (
        df["Missing_Description"] == df["Description"].isna()
    ).all(), "Missing-description flag mismatch."

    print("All validation checks passed.")


def save_data(df):
    """Save the validated dataset to the processed folder."""

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print("Final shape:", df.shape)
