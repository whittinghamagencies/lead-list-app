"""
Lead List Prep - Streamlit proof of concept
===========================================

Step 1 (this version): upload a CSV, keep only the columns we care about,
and display the result so we can confirm the pipeline works.

Planned next steps (stubs are already in place below):
    Step 2 - Smarty address verification / standardization
    Step 3 - Grok (xAI) lead scoring

Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Lead List Prep"
APP_DESCRIPTION = (
    "Upload a raw lead export. The app keeps only the columns used for "
    "outreach and drops everything else, so you can check the file before "
    "address verification and scoring get added."
)

# The only columns we keep. Anything else in the upload is ignored.
# Order here is the order they will appear in the table.
KEEP_COLUMNS = [
    "Company Name",
    "Mailing Address",
    "Mailing City",
    "Mailing State",
    "Mailing Zip Code",
    "Location Employee Size Range",
    "Executive First Name",
    "Executive Last Name",
    "Executive Title",
    "Phone Number Combined",
    "Primary SIC Code",
    "Primary SIC Code Description",
    "Infogroup ID",
]

MAX_PREVIEW_ROWS = 1000  # keep the on-screen table responsive on big files


# ---------------------------------------------------------------------------
# Secrets (never hardcode keys - these live in .streamlit/secrets.toml)
# ---------------------------------------------------------------------------

def get_secret(name: str, default: str = "") -> str:
    """
    Read a value from st.secrets without crashing when it isn't set yet.

    Returns the default (empty string) if there is no secrets file or the
    key is missing, which is what we want while the API steps are stubs.
    """
    try:
        return st.secrets[name]
    except Exception:
        return default


# Placeholders for the integrations coming in steps 2 and 3.
SMARTY_AUTH_ID = get_secret("SMARTY_AUTH_ID")
SMARTY_AUTH_TOKEN = get_secret("SMARTY_AUTH_TOKEN")
XAI_API_KEY = get_secret("XAI_API_KEY")


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded file into a DataFrame.

    Raises ValueError with a message meant for the user if the file isn't a
    readable, non-empty CSV.
    """
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("That file isn't a CSV. Export the list as .csv and upload again.")

    try:
        df = pd.read_csv(uploaded_file, dtype=str)
    except UnicodeDecodeError:
        # Common with exports saved out of Excel on Windows.
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("That CSV is empty - there are no columns or rows to read.")
    except pd.errors.ParserError:
        raise ValueError("That CSV couldn't be parsed. Check for stray commas or quotes.")

    if df.empty:
        raise ValueError("That CSV has headers but no rows.")

    return df


def filter_columns(df: pd.DataFrame):
    """
    Keep only the columns in KEEP_COLUMNS.

    Matching ignores case and surrounding whitespace, since exports vary.
    Returns (filtered_df, found_columns, missing_columns).
    """
    # Map a normalized version of each incoming header back to the real one.
    normalized = {str(col).strip().lower(): col for col in df.columns}

    found, missing, actual_names = [], [], []
    for wanted in KEEP_COLUMNS:
        match = normalized.get(wanted.lower())
        if match is not None:
            found.append(wanted)
            actual_names.append(match)
        else:
            missing.append(wanted)

    filtered = df[actual_names].copy()
    filtered.columns = found  # standardize headers to our canonical spelling
    return filtered, found, missing


# ---------------------------------------------------------------------------
# Step 2 stub - Smarty address verification
# ---------------------------------------------------------------------------

def verify_addresses(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOT IMPLEMENTED YET.

    Will send Mailing Address / City / State / Zip Code to the Smarty US
    Street API and append verified columns (standardized address, DPV match
    code, county, lat/lon). Use SMARTY_AUTH_ID and SMARTY_AUTH_TOKEN above,
    batch the requests, and return a new DataFrame rather than mutating.
    """
    return df


# ---------------------------------------------------------------------------
# Step 3 stub - Grok (xAI) scoring
# ---------------------------------------------------------------------------

def score_leads(df: pd.DataFrame) -> pd.DataFrame:
    """
    NOT IMPLEMENTED YET.

    Will send company size, SIC description, and executive title to the xAI
    API and append a fit score plus a short rationale column. Use XAI_API_KEY
    above and cache results by Infogroup ID so re-runs don't re-bill.
    """
    return df


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    uploaded_file = st.file_uploader("Upload a lead list", type=["csv"])

    if uploaded_file is None:
        st.info("Choose a CSV to get started.")
        return

    # Load the file, showing any problem in plain language.
    try:
        raw_df = load_csv(uploaded_file)
    except ValueError as err:
        st.error(str(err))
        return
    except Exception as err:  # anything unexpected
        st.error(f"Couldn't read that file: {err}")
        return

    filtered_df, found, missing = filter_columns(raw_df)

    if not found:
        st.error(
            "None of the expected columns were found. Check that this is a raw "
            "lead export and that the headers haven't been renamed."
        )
        st.write("Columns in your file:", list(raw_df.columns))
        return

    # --- Pipeline steps 2 and 3 will slot in here ---
    # filtered_df = verify_addresses(filtered_df)
    # filtered_df = score_leads(filtered_df)

    # Summary of what happened.
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Rows", f"{len(filtered_df):,}")
    col_b.metric("Columns kept", f"{len(found)} of {len(KEEP_COLUMNS)}")
    col_c.metric("Columns dropped", f"{len(raw_df.columns) - len(found):,}")

    if missing:
        st.warning("Not in this file: " + ", ".join(missing))

    st.dataframe(filtered_df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    if len(filtered_df) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing the first {MAX_PREVIEW_ROWS:,} rows. The download has all of them.")

    st.download_button(
        "Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_{uploaded_file.name}",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
