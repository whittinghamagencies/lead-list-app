"""
Lead List Prep - Streamlit proof of concept
===========================================

Step 1  - upload a CSV, keep only the columns we care about
Step 1b - filter rows by employee size so small accounts drop out
Step 2  - verify mailing addresses with Smarty (live)

Planned next:
    Step 3 - Grok (xAI) lead scoring

Run with:  streamlit run app.py

Requires:  streamlit, pandas, smartystreets_python_sdk
"""

import pandas as pd
import streamlit as st

# The Smarty SDK is optional at import time so the rest of the app still
# runs if it hasn't been installed yet.
try:
    from smartystreets_python_sdk import ClientBuilder, StaticCredentials, Batch
    from smartystreets_python_sdk import us_street
    SMARTY_AVAILABLE = True
except ImportError:
    SMARTY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_TITLE = "Lead List Prep"
APP_DESCRIPTION = (
    "Upload a raw lead export. The app keeps only the columns used for "
    "outreach, filters out companies below your minimum employee count, "
    "then verifies the remaining mailing addresses against USPS data."
)

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

SIZE_COLUMN = "Location Employee Size Range"

# Address fields the Smarty step reads.
STREET_COLUMN = "Mailing Address"
CITY_COLUMN = "Mailing City"
STATE_COLUMN = "Mailing State"
ZIP_COLUMN = "Mailing Zip Code"

DEFAULT_MIN_EMPLOYEES = 5
MAX_PREVIEW_ROWS = 1000
SMARTY_BATCH_SIZE = 100  # Smarty's per-request maximum


# ---------------------------------------------------------------------------
# Secrets (never hardcode keys - these live in .streamlit/secrets.toml)
# ---------------------------------------------------------------------------

def get_secret(name: str, default: str = "") -> str:
    """Read st.secrets without crashing when the key isn't set yet."""
    try:
        return st.secrets[name]
    except Exception:
        return default


SMARTY_AUTH_ID = get_secret("SMARTY_AUTH_ID")
SMARTY_AUTH_TOKEN = get_secret("SMARTY_AUTH_TOKEN")
XAI_API_KEY = get_secret("XAI_API_KEY")


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame, or raise a readable error."""
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("That file isn't a CSV. Export the list as .csv and upload again.")

    try:
        df = pd.read_csv(uploaded_file, dtype=str)
    except UnicodeDecodeError:
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
    """Keep only KEEP_COLUMNS, matching headers case-insensitively."""
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
    filtered.columns = found
    return filtered, found, missing


# ---------------------------------------------------------------------------
# Employee size filtering
# ---------------------------------------------------------------------------

def parse_min_employees(size_range):
    """
    Pull the lower bound out of a size range string.

    "1 to 4" -> 1,  "1,000 to 4,999" -> 1000,  "500+" -> 500,  blank -> None
    """
    if not isinstance(size_range, str):
        return None

    digits = ""
    for char in size_range.replace(",", ""):
        if char.isdigit():
            digits += char
        elif digits:
            break

    return int(digits) if digits else None


def filter_by_size(df: pd.DataFrame, minimum: int, include_unknown: bool):
    """Drop rows below the minimum employee count. Returns (df, rows_removed)."""
    if SIZE_COLUMN not in df.columns:
        return df, 0

    lower_bounds = df[SIZE_COLUMN].map(parse_min_employees)

    keep = lower_bounds.notna() & (lower_bounds >= minimum)
    if include_unknown:
        keep = keep | lower_bounds.isna()

    return df[keep].copy(), int((~keep).sum())


# ---------------------------------------------------------------------------
# Step 2 - Smarty address verification (live)
# ---------------------------------------------------------------------------

def build_smarty_client(auth_id: str, auth_token: str):
    """Create a Smarty US Street API client from the stored credentials."""
    credentials = StaticCredentials(auth_id, auth_token)
    return ClientBuilder(credentials).build_us_street_api_client()


def blank_result() -> dict:
    """The shape of one verification result, all empty."""
    return {
        "Address Status": "Not found",
        "Verified Address": "",
        "Verified City": "",
        "Verified State": "",
        "Verified ZIP+4": "",
        "County": "",
        "Property Type": "",
        "Record Type": "",
        "Vacant": "",
        "DPV Match Code": "",
        "Latitude": "",
        "Longitude": "",
    }


# Smarty's single-letter record types, spelled out.
RECORD_TYPES = {
    "F": "Firm",
    "G": "General Delivery",
    "H": "Highrise",
    "P": "PO Box",
    "R": "Rural Route",
    "S": "Street",
}


def candidate_to_result(candidate, original_street: str) -> dict:
    """Turn one Smarty candidate into our result columns."""
    components = candidate.components
    metadata = candidate.metadata
    analysis = candidate.analysis

    zip_plus_four = components.zipcode or ""
    if components.plus4_code:
        zip_plus_four = f"{zip_plus_four}-{components.plus4_code}"

    # "Corrected" means Smarty found it but had to change what we sent.
    delivered = candidate.delivery_line_1 or ""
    status = "Verified"
    if delivered.strip().lower() != str(original_street or "").strip().lower():
        status = "Corrected"

    return {
        "Address Status": status,
        "Verified Address": delivered,
        "Verified City": components.city_name or "",
        "Verified State": components.state_abbreviation or "",
        "Verified ZIP+4": zip_plus_four,
        "County": metadata.county_name or "",
        # rdi comes back as "Residential" or "Commercial".
        "Property Type": metadata.rdi or "Unknown",
        "Record Type": RECORD_TYPES.get(metadata.record_type, metadata.record_type or ""),
        "Vacant": "Yes" if analysis.vacant == "Y" else "No",
        "DPV Match Code": analysis.dpv_match_code or "",
        "Latitude": metadata.latitude or "",
        "Longitude": metadata.longitude or "",
    }


def verify_addresses(df: pd.DataFrame, auth_id: str, auth_token: str, progress=None) -> pd.DataFrame:
    """
    Send each row's mailing address to Smarty and append the verified columns.

    Sends in batches of 100 (Smarty's maximum). Rows with no street address
    are skipped rather than sent, so we don't burn lookups on blanks.
    Returns a new DataFrame; the original is left alone.
    """
    client = build_smarty_client(auth_id, auth_token)

    # Start every row off as unverified, then fill in what comes back.
    results = {index: blank_result() for index in df.index}

    # Only rows that actually have a street address are worth sending.
    sendable = [
        index for index in df.index
        if str(df.at[index, STREET_COLUMN] or "").strip() not in ("", "None", "nan")
    ]

    for batch_start in range(0, len(sendable), SMARTY_BATCH_SIZE):
        chunk = sendable[batch_start:batch_start + SMARTY_BATCH_SIZE]

        batch = Batch()
        for index in chunk:
            lookup = us_street.Lookup()
            lookup.input_id = str(index)  # so we can map results back to rows
            lookup.street = str(df.at[index, STREET_COLUMN] or "")
            lookup.city = str(df.at[index, CITY_COLUMN] or "")
            lookup.state = str(df.at[index, STATE_COLUMN] or "")
            lookup.zipcode = str(df.at[index, ZIP_COLUMN] or "")
            lookup.candidates = 1  # we only want the best match
            batch.add(lookup)

        client.send_batch(batch)

        for lookup in batch.all_lookups:
            index = int(lookup.input_id)
            if lookup.result:
                results[index] = candidate_to_result(
                    lookup.result[0], df.at[index, STREET_COLUMN]
                )

        if progress is not None:
            progress.progress(min((batch_start + SMARTY_BATCH_SIZE) / len(sendable), 1.0))

    verified = pd.DataFrame.from_dict(results, orient="index")
    return df.join(verified)


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

def render_smarty_section(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Draw the Smarty controls and return either the verified DataFrame or the
    original. Verification runs only on button press, so reruns don't re-bill.
    """
    st.subheader("Verify addresses")

    if not SMARTY_AVAILABLE:
        st.info(
            "The Smarty SDK isn't installed. Add smartystreets_python_sdk to "
            "requirements.txt and redeploy."
        )
        return working_df

    if not SMARTY_AUTH_ID or not SMARTY_AUTH_TOKEN:
        st.info(
            "No Smarty credentials found. Add SMARTY_AUTH_ID and "
            "SMARTY_AUTH_TOKEN in the app's Secrets settings."
        )
        return working_df

    if STREET_COLUMN not in working_df.columns:
        st.info(f"No '{STREET_COLUMN}' column in this file, so there's nothing to verify.")
        return working_df

    # Results are cached per uploaded file + row count so switching the size
    # filter doesn't silently reuse a stale verification.
    cache_key = f"verified::{file_key}::{len(working_df)}"

    has_addresses = int(
        working_df[STREET_COLUMN].fillna("").astype(str).str.strip().ne("").sum()
    )

    st.caption(
        f"{has_addresses:,} of {len(working_df):,} rows have a street address. "
        "Each one uses a Smarty lookup."
    )

    if st.button(f"Verify {has_addresses:,} addresses", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = verify_addresses(
                working_df, SMARTY_AUTH_ID, SMARTY_AUTH_TOKEN, progress
            )
        except Exception as err:
            st.error(f"Smarty verification failed: {err}")
            st.caption("A 401 usually means the key is wrong; a 402 means the lookup balance is empty.")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    verified_df = st.session_state[cache_key]

    counts = verified_df["Address Status"].value_counts()
    property_counts = verified_df["Property Type"].value_counts()

    stat_a, stat_b, stat_c, stat_d = st.columns(4)
    stat_a.metric("Verified as sent", f"{counts.get('Verified', 0):,}")
    stat_b.metric("Corrected", f"{counts.get('Corrected', 0):,}")
    stat_c.metric("Not found", f"{counts.get('Not found', 0):,}")
    stat_d.metric("Commercial", f"{property_counts.get('Commercial', 0):,}")

    # Post-verification filters. Each one narrows toward addresses a setter
    # could actually show up at.
    filter_a, filter_b, filter_c = st.columns(3)
    mailable_only = filter_a.checkbox("Mailable only", value=False,
                                      help="Drops addresses Smarty couldn't match.")
    commercial_only = filter_b.checkbox("Commercial only", value=False,
                                        help="Drops addresses flagged residential.")
    drop_po_boxes = filter_c.checkbox("No PO boxes", value=False,
                                      help="Drops PO box and general delivery records.")

    if mailable_only:
        verified_df = verified_df[verified_df["Address Status"] != "Not found"].copy()
    if commercial_only:
        verified_df = verified_df[verified_df["Property Type"] == "Commercial"].copy()
    if drop_po_boxes:
        verified_df = verified_df[
            ~verified_df["Record Type"].isin(["PO Box", "General Delivery"])
        ].copy()

    if mailable_only or commercial_only or drop_po_boxes:
        st.caption(f"{len(verified_df):,} leads match the address filters.")

    # A dedicated export with residential addresses stripped out. This keeps
    # anything Smarty couldn't classify - only rows explicitly flagged
    # Residential are dropped, so unmatched addresses aren't silently lost.
    non_residential = verified_df[verified_df["Property Type"] != "Residential"].copy()
    residential_count = len(verified_df) - len(non_residential)

    st.download_button(
        f"Download non-residential list ({len(non_residential):,} leads)",
        data=non_residential.to_csv(index=False).encode("utf-8"),
        file_name=f"non_residential_{file_key}",
        mime="text/csv",
        type="primary",
    )
    st.caption(f"Excludes {residential_count:,} addresses flagged residential.")

    return verified_df


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    uploaded_file = st.file_uploader("Upload a lead list", type=["csv"])

    if uploaded_file is None:
        st.info("Choose a CSV to get started.")
        return

    try:
        raw_df = load_csv(uploaded_file)
    except ValueError as err:
        st.error(str(err))
        return
    except Exception as err:
        st.error(f"Couldn't read that file: {err}")
        return

    column_df, found, missing = filter_columns(raw_df)

    if not found:
        st.error(
            "None of the expected columns were found. Check that this is a raw "
            "lead export and that the headers haven't been renamed."
        )
        st.write("Columns in your file:", list(raw_df.columns))
        return

    if missing:
        st.warning("Not in this file: " + ", ".join(missing))

    # --- Size filter ---
    st.subheader("Filter by company size")

    if SIZE_COLUMN not in column_df.columns:
        st.info(f"No '{SIZE_COLUMN}' column in this file, so size filtering is off.")
        working_df, removed = column_df, 0
    else:
        control_left, control_right = st.columns([1, 2])
        minimum = control_left.number_input(
            "Minimum employees",
            min_value=0,
            value=DEFAULT_MIN_EMPLOYEES,
            step=1,
            help="Keeps any size band starting at or above this number.",
        )
        include_unknown = control_right.checkbox(
            "Keep companies with no size listed",
            value=False,
        )
        working_df, removed = filter_by_size(column_df, minimum, include_unknown)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Leads kept", f"{len(working_df):,}")
    col_b.metric("Below minimum", f"{removed:,}")
    col_c.metric("Columns kept", f"{len(found)} of {len(KEEP_COLUMNS)}")

    if working_df.empty:
        st.info("Nothing left after filtering. Try a lower minimum.")
        return

    if SIZE_COLUMN in working_df.columns:
        with st.expander("Size breakdown of the kept leads"):
            breakdown = (
                working_df[SIZE_COLUMN]
                .value_counts(dropna=False)
                .rename_axis("Employee size range")
                .reset_index(name="Companies")
            )
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

    # --- Smarty verification ---
    working_df = render_smarty_section(working_df, uploaded_file.name)

    # --- Step 3 will slot in here ---
    # working_df = score_leads(working_df)

    st.subheader("Leads")
    st.dataframe(working_df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    if len(working_df) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing the first {MAX_PREVIEW_ROWS:,} rows. The download has all of them.")

    st.download_button(
        "Download filtered CSV",
        data=working_df.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_{uploaded_file.name}",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
