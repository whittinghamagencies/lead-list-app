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

import json

import pandas as pd
import requests
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

XAI_BASE_URL = "https://api.x.ai/v1/chat/completions"
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"  # agentic endpoint, has web_search
XAI_MODEL = "grok-4.5"
SCORING_BATCH_SIZE = 15  # leads per API call - keeps prompts small and cheap

# The default yardstick. Editable in the UI so it can be tuned without a
# code change.
DEFAULT_SCORING_CRITERIA = """We sell voluntary worksite benefits to employers,
enrolling their employees on site. A strong lead is a business with enough
employees to make an on-site enrollment worth the trip, in an industry with a
stable W-2 workforce rather than heavy seasonal or contract labor, and with a
named contact senior enough to approve a benefits decision (owner, president,
CEO, HR director, office manager at a small company).

Weaker leads: very small headcount, franchises or branches that cannot decide
locally, industries with mostly part-time or transient staff, and contacts with
no decision-making authority."""

# Contact checking is one web-searching call per lead, so it is slow and
# costs far more than scoring. Capped by default.
DEFAULT_CONTACT_CHECK_LIMIT = 25

# Apollo bulk enrichment: ten people per call, one credit per record matched.
APOLLO_BULK_URL = "https://api.apollo.io/api/v1/people/bulk_match"
APOLLO_BATCH_SIZE = 10
DEFAULT_APOLLO_LIMIT = 50

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
APOLLO_API_KEY = get_secret("APOLLO_API_KEY")


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

def lead_to_summary(row) -> str:
    """Condense one row into the few fields Grok needs to judge fit."""
    name = row.get("Company Name", "")
    size = row.get(SIZE_COLUMN, "unknown size")
    industry = row.get("Primary SIC Code Description", "unknown industry")
    title = row.get("Executive Title", "no contact title")
    city = row.get("Mailing City", "")
    state = row.get("Mailing State", "")

    return f"{name} | {size} employees | {industry} | contact: {title} | {city}, {state}"


def call_xai(prompt: str, api_key: str) -> str:
    """POST one prompt to the xAI chat completions endpoint, return the text."""
    response = requests.post(
        XAI_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": XAI_MODEL,
            "temperature": 0,  # scoring should be repeatable
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You score B2B sales leads. Reply with a JSON array only - "
                        "no prose, no markdown fences. Each element must be "
                        '{"id": <number>, "score": <1-10>, "reason": "<12 words max>"}.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_scores(raw: str) -> dict:
    """Turn the model's JSON reply into {id: (score, reason)}."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip a markdown fence if one slipped through.
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    parsed = {}
    for item in json.loads(cleaned):
        try:
            parsed[int(item["id"])] = (int(item["score"]), str(item.get("reason", "")))
        except (KeyError, ValueError, TypeError):
            continue  # skip malformed entries rather than failing the batch
    return parsed


def score_leads(df: pd.DataFrame, api_key: str, criteria: str, progress=None) -> pd.DataFrame:
    """
    Score each lead 1-10 for fit and append the score plus a short reason.

    Sends SCORING_BATCH_SIZE leads per call to keep token use down. Any batch
    that fails leaves its rows unscored rather than killing the whole run.
    Returns a new DataFrame.
    """
    scores = {index: (None, "") for index in df.index}
    indexes = list(df.index)

    for batch_start in range(0, len(indexes), SCORING_BATCH_SIZE):
        chunk = indexes[batch_start:batch_start + SCORING_BATCH_SIZE]

        lines = [f"{index}: {lead_to_summary(df.loc[index])}" for index in chunk]
        prompt = (
            f"Scoring criteria:\n{criteria}\n\n"
            f"Score each lead below from 1 (poor fit) to 10 (excellent fit). "
            f"Use the id number shown at the start of each line.\n\n"
            + "\n".join(lines)
        )

        try:
            scores.update(parse_scores(call_xai(prompt, api_key)))
        except Exception:
            pass  # leave this batch unscored; the UI reports the gap

        if progress is not None:
            progress.progress(min((batch_start + SCORING_BATCH_SIZE) / len(indexes), 1.0))

    scored = pd.DataFrame(
        [{"Fit Score": scores[i][0], "Score Reason": scores[i][1]} for i in df.index],
        index=df.index,
    )
    return df.join(scored)


# ---------------------------------------------------------------------------
# Contact verification - is this decision maker still there?
# ---------------------------------------------------------------------------

CONTACT_STATUSES = [
    "Confirmed current",   # found listed at the company now
    "Likely departed",     # found evidence they moved on, or a different person in the role
    "No evidence found",   # nothing either way - NOT the same as departed
    "Company not found",   # no usable web presence for the business at all
]


def extract_response_text(payload: dict) -> str:
    """
    Pull the assistant text out of a /v1/responses payload.

    The output is a list of items (tool calls, reasoning, messages); we want
    the text from the message items, so we walk it tolerantly rather than
    assuming a fixed position.
    """
    chunks = []
    for item in payload.get("output", []):
        for part in item.get("content", []) or []:
            text = part.get("text")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def check_one_contact(row, api_key: str) -> dict:
    """Ask Grok, with web search on, whether one contact is still at the company."""
    first = str(row.get("Executive First Name", "") or "").strip()
    last = str(row.get("Executive Last Name", "") or "").strip()
    company = str(row.get("Company Name", "") or "").strip()
    title = str(row.get("Executive Title", "") or "").strip()
    city = str(row.get("Mailing City", "") or "").strip()
    state = str(row.get("Mailing State", "") or "").strip()

    if not first and not last:
        return {"Contact Status": "No name listed", "Contact Evidence": "", "Contact Source": ""}

    question = (
        f"Is {first} {last} currently associated with {company} in {city}, {state}"
        f"{f' as {title}' if title else ''}? "
        "Search the company's own website first, then other public sources. "
        "Do not guess from the name alone.\n\n"
        "Reply with JSON only, no fences:\n"
        '{"status": "Confirmed current" | "Likely departed" | "No evidence found" '
        '| "Company not found", "evidence": "<15 words max>", "source": "<one URL or empty>"}\n\n'
        "Use 'Likely departed' only with actual evidence - a current staff listing "
        "naming someone else in the role, an announcement, or an obituary. "
        "Absence of information is 'No evidence found', never 'Likely departed'."
    )

    response = requests.post(
        XAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": XAI_MODEL,
            "input": [{"role": "user", "content": question}],
            "tools": [{"type": "web_search"}],
        },
        timeout=180,  # agentic calls run several searches, so allow time
    )
    response.raise_for_status()

    raw = extract_response_text(response.json())
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    parsed = json.loads(cleaned)
    status = parsed.get("status", "")
    return {
        "Contact Status": status if status in CONTACT_STATUSES else "No evidence found",
        "Contact Evidence": str(parsed.get("evidence", ""))[:120],
        "Contact Source": str(parsed.get("source", "")),
    }


def verify_contacts(df: pd.DataFrame, api_key: str, limit: int, progress=None) -> pd.DataFrame:
    """
    Check the listed decision maker for the first `limit` rows.

    Rows past the limit come back blank rather than unchecked-looking, so it
    is obvious which ones were actually looked at. One row failing doesn't
    stop the run.
    """
    blank = {"Contact Status": "", "Contact Evidence": "", "Contact Source": ""}
    results = {index: dict(blank) for index in df.index}

    targets = list(df.index)[:limit]

    for position, index in enumerate(targets):
        try:
            results[index] = check_one_contact(df.loc[index], api_key)
        except Exception as err:
            results[index] = {
                "Contact Status": "Check failed",
                "Contact Evidence": str(err)[:120],
                "Contact Source": "",
            }
        if progress is not None:
            progress.progress((position + 1) / len(targets))

    checked = pd.DataFrame.from_dict(results, orient="index")
    return df.join(checked)


# ---------------------------------------------------------------------------
# Apollo - check our executive names against Apollo's records
# ---------------------------------------------------------------------------

# Words that don't help decide whether two company names are the same business.
COMPANY_NOISE = {
    "inc", "llc", "ltd", "co", "corp", "corporation", "company", "the",
    "svc", "svcs", "service", "services", "group", "holdings", "and", "of",
}


def normalize_company(name: str) -> set:
    """Reduce a company name to its meaningful words for comparison."""
    if not isinstance(name, str):
        return set()

    cleaned = "".join(char if char.isalnum() else " " for char in name.lower())
    return {word for word in cleaned.split() if word and word not in COMPANY_NOISE}


def same_company(ours: str, theirs: str) -> bool:
    """
    Decide whether two company names refer to the same business.

    Compares meaningful words rather than exact strings, so "Aska USA" and
    "Aska USA Inc." match, but "Apollo Aviation" and "Apollo Roofing" don't.
    """
    ours_words, theirs_words = normalize_company(ours), normalize_company(theirs)
    if not ours_words or not theirs_words:
        return False

    overlap = len(ours_words & theirs_words)
    return overlap / min(len(ours_words), len(theirs_words)) >= 0.6


def blank_apollo() -> dict:
    """The shape of one Apollo result, all empty."""
    return {
        "Apollo Status": "",
        "Apollo Current Employer": "",
        "Apollo Current Title": "",
        "Apollo LinkedIn": "",
    }


def apollo_status(row, person: dict) -> dict:
    """
    Compare one Apollo person record against what our lead list claims.

    "Still listed"  - Apollo has them at the same company
    "Moved on"      - Apollo has them somewhere else now
    "Title changed" - same company, different title from our record
    "Not in Apollo" - no record found for that name at that company
    """
    if not person:
        return {**blank_apollo(), "Apollo Status": "Not in Apollo"}

    organization = person.get("organization") or {}
    their_company = organization.get("name") or ""
    their_title = person.get("title") or ""
    our_company = row.get("Company Name", "")
    our_title = str(row.get("Executive Title", "") or "").strip().lower()

    if not their_company:
        status = "Not in Apollo"
    elif same_company(our_company, their_company):
        # Same employer. Flag a title change only when both titles are known.
        if our_title and their_title and our_title not in their_title.lower():
            status = "Title changed"
        else:
            status = "Still listed"
    else:
        status = "Moved on"

    return {
        "Apollo Status": status,
        "Apollo Current Employer": their_company,
        "Apollo Current Title": their_title,
        "Apollo LinkedIn": person.get("linkedin_url") or "",
    }


def apollo_bulk_match(df: pd.DataFrame, api_key: str, limit: int, progress=None) -> pd.DataFrame:
    """
    Send executive names to Apollo in batches of ten and append what Apollo
    currently has on file for each one.

    Only rows with a first and last name are sent, since Apollo can't match
    on a company name alone and a blank lookup still costs a call.
    """
    results = {index: blank_apollo() for index in df.index}

    targets = [
        index for index in list(df.index)[:limit]
        if str(df.at[index, "Executive First Name"] or "").strip() not in ("", "None", "nan")
        and str(df.at[index, "Executive Last Name"] or "").strip() not in ("", "None", "nan")
    ]

    for batch_start in range(0, len(targets), APOLLO_BATCH_SIZE):
        chunk = targets[batch_start:batch_start + APOLLO_BATCH_SIZE]

        details = [
            {
                "first_name": str(df.at[index, "Executive First Name"]),
                "last_name": str(df.at[index, "Executive Last Name"]),
                "organization_name": str(df.at[index, "Company Name"] or ""),
            }
            for index in chunk
        ]

        try:
            response = requests.post(
                APOLLO_BULK_URL,
                headers={
                    "x-api-key": api_key,          # header auth only
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache",
                },
                json={"details": details},
                timeout=90,
            )
            response.raise_for_status()
            matches = response.json().get("matches", [])
        except Exception as err:
            for index in chunk:
                results[index] = {**blank_apollo(), "Apollo Status": f"Lookup failed: {str(err)[:60]}"}
            matches = []

        # Apollo returns matches in the order submitted, with nulls for misses.
        for position, index in enumerate(chunk):
            person = matches[position] if position < len(matches) else None
            results[index] = apollo_status(df.loc[index], person)

        if progress is not None:
            progress.progress(min((batch_start + APOLLO_BATCH_SIZE) / len(targets), 1.0))

    enriched = pd.DataFrame.from_dict(results, orient="index")
    return df.join(enriched)


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


def render_scoring_section(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Draw the Grok scoring controls. Like verification, this runs only on
    button press so reruns don't re-bill.
    """
    st.subheader("Score leads")

    if not XAI_API_KEY:
        st.info("No xAI key found. Add XAI_API_KEY in the app's Secrets settings.")
        return working_df

    criteria = st.text_area(
        "What makes a good lead?",
        value=DEFAULT_SCORING_CRITERIA,
        height=180,
        help="Edit this to change how leads are judged. Changes apply on the next run.",
    )

    cache_key = f"scored::{file_key}::{len(working_df)}"
    call_count = -(-len(working_df) // SCORING_BATCH_SIZE)  # ceiling division

    st.caption(f"{len(working_df):,} leads, {call_count} API calls at {SCORING_BATCH_SIZE} per call.")

    if st.button(f"Score {len(working_df):,} leads", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = score_leads(
                working_df, XAI_API_KEY, criteria, progress
            )
        except Exception as err:
            st.error(f"Scoring failed: {err}")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    scored_df = st.session_state[cache_key]
    unscored = int(scored_df["Fit Score"].isna().sum())

    if unscored:
        st.warning(f"{unscored:,} leads came back unscored. Run again to retry those batches.")

    minimum_score = st.slider("Minimum fit score", 1, 10, 1)
    if minimum_score > 1:
        scored_df = scored_df[scored_df["Fit Score"].fillna(0) >= minimum_score].copy()
        st.caption(f"{len(scored_df):,} leads at score {minimum_score} or above.")

    # Best prospects first.
    return scored_df.sort_values("Fit Score", ascending=False, na_position="last")


def render_contact_section(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """
    Draw the contact verification controls. Slow and costly per lead, so it
    runs on a capped subset and only on button press.
    """
    st.subheader("Check decision makers")

    if not XAI_API_KEY:
        return working_df

    st.caption(
        "Searches the open web for whether each listed contact is still with the "
        "company. Company sites are the main source; LinkedIn is often unreachable "
        "and ZoomInfo is not searchable at all."
    )

    limit = st.slider(
        "How many leads to check",
        min_value=5,
        max_value=min(100, max(5, len(working_df))),
        value=min(DEFAULT_CONTACT_CHECK_LIMIT, len(working_df)),
        help="Runs top-down on the current list order. One web search per lead, so this is slow.",
    )

    cache_key = f"contacts::{file_key}::{len(working_df)}::{limit}"

    if st.button(f"Check {limit} decision makers"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = verify_contacts(
                working_df, XAI_API_KEY, limit, progress
            )
        except Exception as err:
            st.error(f"Contact check failed: {err}")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    checked_df = st.session_state[cache_key]
    status_counts = checked_df["Contact Status"].value_counts()

    stat_a, stat_b, stat_c = st.columns(3)
    stat_a.metric("Confirmed current", f"{status_counts.get('Confirmed current', 0):,}")
    stat_b.metric("Likely departed", f"{status_counts.get('Likely departed', 0):,}")
    stat_c.metric("No evidence", f"{status_counts.get('No evidence found', 0):,}")

    stale = checked_df[checked_df["Contact Status"] == "Likely departed"]
    if not stale.empty:
        st.warning(f"{len(stale):,} contacts look out of date. Worth a call before a visit.")
        st.download_button(
            f"Download the {len(stale):,} stale contacts",
            data=stale.to_csv(index=False).encode("utf-8"),
            file_name=f"stale_contacts_{file_key}",
            mime="text/csv",
        )

    return checked_df


def render_apollo_section(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """Draw the Apollo cross-check controls."""
    st.subheader("Cross-check contacts against Apollo")

    if not APOLLO_API_KEY:
        st.info(
            "No Apollo key found. Generate one in Apollo under Settings > "
            "Integrations > API, then add APOLLO_API_KEY in the app's Secrets."
        )
        return working_df

    named = int(
        working_df["Executive Last Name"].fillna("").astype(str).str.strip().ne("").sum()
    ) if "Executive Last Name" in working_df.columns else 0

    if not named:
        st.info("No executive names in this list to check.")
        return working_df

    limit = st.slider(
        "How many contacts to check",
        min_value=10,
        max_value=min(500, max(10, len(working_df))),
        value=min(DEFAULT_APOLLO_LIMIT, len(working_df)),
        step=10,
        help="Apollo charges one credit per record it matches.",
    )

    st.caption(
        f"{named:,} rows have a name. Checking {limit} of them costs up to "
        f"{limit} Apollo credits and takes about {-(-limit // APOLLO_BATCH_SIZE)} calls."
    )

    cache_key = f"apollo::{file_key}::{len(working_df)}::{limit}"

    if st.button(f"Check {limit} contacts against Apollo"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = apollo_bulk_match(
                working_df, APOLLO_API_KEY, limit, progress
            )
        except Exception as err:
            st.error(f"Apollo lookup failed: {err}")
            st.caption("A 401 means the key is wrong; a 403 usually means the plan lacks API access.")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    apollo_df = st.session_state[cache_key]
    counts = apollo_df["Apollo Status"].value_counts()

    stat_a, stat_b, stat_c, stat_d = st.columns(4)
    stat_a.metric("Still listed", f"{counts.get('Still listed', 0):,}")
    stat_b.metric("Moved on", f"{counts.get('Moved on', 0):,}")
    stat_c.metric("Title changed", f"{counts.get('Title changed', 0):,}")
    stat_d.metric("Not in Apollo", f"{counts.get('Not in Apollo', 0):,}")

    departed = apollo_df[apollo_df["Apollo Status"] == "Moved on"]
    if not departed.empty:
        with st.expander(f"See the {len(departed):,} contacts Apollo places elsewhere"):
            st.dataframe(
                departed[[
                    "Company Name", "Executive First Name", "Executive Last Name",
                    "Executive Title", "Apollo Current Employer", "Apollo Current Title",
                ]],
                use_container_width=True,
                hide_index=True,
            )
        st.download_button(
            f"Download the {len(departed):,} out-of-date contacts",
            data=departed.to_csv(index=False).encode("utf-8"),
            file_name=f"outdated_contacts_{file_key}",
            mime="text/csv",
        )

    if st.checkbox("Show only contacts Apollo still places at the company", value=False):
        apollo_df = apollo_df[
            apollo_df["Apollo Status"].isin(["Still listed", "Title changed"])
        ].copy()
        st.caption(f"{len(apollo_df):,} leads with a contact Apollo confirms.")

    return apollo_df


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

    # --- Grok scoring ---
    working_df = render_scoring_section(working_df, uploaded_file.name)

    # --- Contact verification ---
    working_df = render_contact_section(working_df, uploaded_file.name)

    # --- Apollo cross-check ---
    working_df = render_apollo_section(working_df, uploaded_file.name)

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
