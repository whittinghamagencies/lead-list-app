"""
Lead List Prep - internal prospecting pipeline
==============================================

Four stages, run in order. Each one narrows the list:

    1. Size      - drop companies below a headcount threshold
    2. Address   - verify mailing addresses, keep commercial ones
    3. Contacts  - confirm / correct / flag the decision maker
                     green  = same person, still there
                     yellow = corrected with new data from Apollo
                     red    = nobody found, set aside
    4. Score     - rank the remaining leads, red ones excluded

Run with:  streamlit run app.py
Requires:  streamlit, pandas, requests, smartystreets_python_sdk
"""

import json

import pandas as pd
import streamlit as st

try:
    from smartystreets_python_sdk import ClientBuilder, StaticCredentials, Batch
    from smartystreets_python_sdk import us_street
    SMARTY_AVAILABLE = True
except ImportError:
    SMARTY_AVAILABLE = False

import requests


# ===========================================================================
# Configuration
# ===========================================================================

APP_TITLE = "Lead List Prep"

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
STREET_COLUMN = "Mailing Address"
CITY_COLUMN = "Mailing City"
STATE_COLUMN = "Mailing State"
ZIP_COLUMN = "Mailing Zip Code"

DEFAULT_MIN_EMPLOYEES = 5
MAX_PREVIEW_ROWS = 500
SMARTY_BATCH_SIZE = 100

# Apollo endpoints. Organization Search costs 1 credit per page; People
# Search costs nothing, which is why the contact stage leans on it.
APOLLO_ORG_SEARCH_URL = "https://api.apollo.io/api/v1/mixed_companies/search"
APOLLO_PEOPLE_SEARCH_URL = "https://api.apollo.io/api/v1/mixed_people/api_search"

# Who counts as a decision maker for a worksite benefits conversation.
DM_SENIORITIES = ["owner", "founder", "c_suite", "vp", "head", "director", "manager"]
DM_TITLES = [
    "owner", "president", "chief executive officer", "general manager",
    "human resources director", "human resources manager", "office manager",
    "controller", "chief financial officer", "vice president",
]

# Stage 3 statuses and their colors.
STATUS_CONFIRMED = "Confirmed"
STATUS_CORRECTED = "Corrected"
STATUS_MISSING = "Missing"

STATUS_COLORS = {
    STATUS_CONFIRMED: "#d4f4dd",   # green
    STATUS_CORRECTED: "#fdf3c8",   # yellow
    STATUS_MISSING: "#fbd5d5",     # red
}

XAI_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL = "grok-4.5"
SCORING_BATCH_SIZE = 15

DEFAULT_SCORING_CRITERIA = """We sell voluntary worksite benefits to employers,
enrolling their employees on site. A strong lead has enough employees to make an
on-site enrollment worth the trip, a stable W-2 workforce rather than seasonal or
contract labor, and a contact senior enough to approve a benefits decision.

Weaker leads: very small headcount, branches that cannot decide locally,
industries with mostly part-time or transient staff."""


# ===========================================================================
# Secrets - never hardcode keys
# ===========================================================================

def get_secret(name: str, default: str = "") -> str:
    """Read st.secrets without crashing when the key isn't set."""
    try:
        return st.secrets[name]
    except Exception:
        return default


SMARTY_AUTH_ID = get_secret("SMARTY_AUTH_ID")
SMARTY_AUTH_TOKEN = get_secret("SMARTY_AUTH_TOKEN")
APOLLO_API_KEY = get_secret("APOLLO_API_KEY")
XAI_API_KEY = get_secret("XAI_API_KEY")


# ===========================================================================
# Loading
# ===========================================================================

def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV, or raise a readable error."""
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("That file isn't a CSV. Export the list as .csv and upload again.")

    try:
        df = pd.read_csv(uploaded_file, dtype=str)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str, encoding="latin-1")
    except pd.errors.EmptyDataError:
        raise ValueError("That CSV is empty - no columns or rows to read.")
    except pd.errors.ParserError:
        raise ValueError("That CSV couldn't be parsed. Check for stray commas or quotes.")

    if df.empty:
        raise ValueError("That CSV has headers but no rows.")

    return df


def filter_columns(df: pd.DataFrame):
    """Keep only KEEP_COLUMNS, matching headers case-insensitively."""
    normalized = {str(col).strip().lower(): col for col in df.columns}

    found, missing, actual = [], [], []
    for wanted in KEEP_COLUMNS:
        match = normalized.get(wanted.lower())
        if match is not None:
            found.append(wanted)
            actual.append(match)
        else:
            missing.append(wanted)

    filtered = df[actual].copy()
    filtered.columns = found
    return filtered, found, missing


# ===========================================================================
# Stage 1 - Employee size
# ===========================================================================

def parse_min_employees(size_range):
    """Lower bound of a size band: "1 to 4" -> 1, "1,000 to 4,999" -> 1000."""
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
    """
    Drop companies below the threshold and sort largest first.

    Sorting by size means the later stages - which are capped - spend their
    budget on the biggest employers rather than whatever happened to be at
    the top of the export.
    """
    if SIZE_COLUMN not in df.columns:
        return df, 0

    bounds = df[SIZE_COLUMN].map(parse_min_employees)

    keep = bounds.notna() & (bounds >= minimum)
    if include_unknown:
        keep = keep | bounds.isna()

    result = df[keep].copy()
    result["_size_sort"] = bounds[keep]
    result = result.sort_values("_size_sort", ascending=False, na_position="last")
    result = result.drop(columns=["_size_sort"])

    return result, int((~keep).sum())


# ===========================================================================
# Stage 2 - Address verification (Smarty)
# ===========================================================================

RECORD_TYPES = {
    "F": "Firm", "G": "General Delivery", "H": "Highrise",
    "P": "PO Box", "R": "Rural Route", "S": "Street",
}


def blank_address() -> dict:
    return {
        "Address Status": "Not found",
        "Verified Address": "",
        "Verified City": "",
        "Verified State": "",
        "Verified ZIP+4": "",
        "Property Type": "",
        "Record Type": "",
        "County": "",
    }


def candidate_to_address(candidate, original_street: str) -> dict:
    """Turn one Smarty candidate into our address columns."""
    components, metadata = candidate.components, candidate.metadata

    zip_plus_four = components.zipcode or ""
    if components.plus4_code:
        zip_plus_four = f"{zip_plus_four}-{components.plus4_code}"

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
        # rdi is USPS delivery classification: Residential or Commercial.
        "Property Type": metadata.rdi or "Unknown",
        "Record Type": RECORD_TYPES.get(metadata.record_type, metadata.record_type or ""),
        "County": metadata.county_name or "",
    }


def verify_addresses(df: pd.DataFrame, auth_id: str, auth_token: str, progress=None) -> pd.DataFrame:
    """Send mailing addresses to Smarty in batches of 100 and append results."""
    client = ClientBuilder(StaticCredentials(auth_id, auth_token)).build_us_street_api_client()

    results = {index: blank_address() for index in df.index}

    sendable = [
        index for index in df.index
        if str(df.at[index, STREET_COLUMN] or "").strip() not in ("", "None", "nan")
    ]

    for start in range(0, len(sendable), SMARTY_BATCH_SIZE):
        chunk = sendable[start:start + SMARTY_BATCH_SIZE]

        batch = Batch()
        for index in chunk:
            lookup = us_street.Lookup()
            lookup.input_id = str(index)
            lookup.street = str(df.at[index, STREET_COLUMN] or "")
            lookup.city = str(df.at[index, CITY_COLUMN] or "")
            lookup.state = str(df.at[index, STATE_COLUMN] or "")
            lookup.zipcode = str(df.at[index, ZIP_COLUMN] or "")
            lookup.candidates = 1
            batch.add(lookup)

        client.send_batch(batch)

        for lookup in batch.all_lookups:
            index = int(lookup.input_id)
            if lookup.result:
                results[index] = candidate_to_address(
                    lookup.result[0], df.at[index, STREET_COLUMN]
                )

        if progress is not None:
            progress.progress(min((start + SMARTY_BATCH_SIZE) / len(sendable), 1.0))

    return df.join(pd.DataFrame.from_dict(results, orient="index"))


def keep_commercial(df: pd.DataFrame, drop_po_boxes: bool = True) -> pd.DataFrame:
    """
    Narrow to addresses a setter could actually visit.

    Keeps anything not explicitly flagged Residential - an unmatched address
    is unknown, not residential, and dropping it would lose real prospects.
    """
    result = df[df["Property Type"] != "Residential"].copy()
    if drop_po_boxes:
        result = result[~result["Record Type"].isin(["PO Box", "General Delivery"])].copy()
    return result


# ===========================================================================
# Stage 3 - Decision maker resolution (Apollo)
# ===========================================================================

COMPANY_NOISE = {
    "inc", "llc", "ltd", "co", "corp", "corporation", "company", "the",
    "svc", "svcs", "service", "services", "group", "holdings", "and", "of",
}


def apollo_post(url: str, payload: dict, api_key: str) -> dict:
    """POST to Apollo with header auth. Query-param auth stopped working in 2024."""
    response = requests.post(
        url,
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def find_organization(company: str, city: str, state: str, api_key: str) -> dict:
    """
    Look up a company in Apollo to get its id and domain.

    Costs 1 credit per call, so callers cache by company name. Location
    narrows the search so a same-named business elsewhere doesn't match.
    """
    payload = {"q_organization_name": company, "per_page": 1}
    if city and state:
        payload["organization_locations"] = [f"{city}, {state}"]

    body = apollo_post(APOLLO_ORG_SEARCH_URL, payload, api_key)
    candidates = body.get("organizations") or body.get("accounts") or []
    if not candidates:
        return {}

    org = candidates[0]
    domain = org.get("primary_domain") or ""
    if not domain and org.get("website_url"):
        domain = (
            org["website_url"].replace("https://", "").replace("http://", "")
            .replace("www.", "").split("/")[0]
        )

    return {"id": org.get("id") or org.get("organization_id") or "", "domain": domain,
            "name": org.get("name") or ""}


def find_decision_makers(organization: dict, api_key: str) -> list:
    """
    Pull current decision makers at a company. Costs 0 credits.

    Filters by seniority and title so we get people who could actually
    approve a benefits decision, not every employee on file.
    """
    payload = {
        "person_seniorities": DM_SENIORITIES,
        "person_titles": DM_TITLES,
        "include_similar_titles": True,
        "per_page": 10,
    }

    # Prefer the Apollo org id; fall back to the domain.
    if organization.get("id"):
        payload["organization_ids"] = [organization["id"]]
    elif organization.get("domain"):
        payload["q_organization_domains_list"] = [organization["domain"]]
    else:
        return []

    body = apollo_post(APOLLO_PEOPLE_SEARCH_URL, payload, api_key)
    return body.get("people") or body.get("contacts") or []


def names_match(our_first: str, our_last: str, their_first: str, their_last: str) -> bool:
    """
    Same person? Last name must match; first name matches on first initial.

    The initial-only rule catches Mike/Michael and Bob/Robert, which show up
    constantly between a purchased list and a maintained database.
    """
    our_last, their_last = our_last.strip().lower(), their_last.strip().lower()
    if not our_last or our_last != their_last:
        return False

    our_first, their_first = our_first.strip().lower(), their_first.strip().lower()
    if not our_first or not their_first:
        return True  # last name matched and one first name is blank

    return our_first[0] == their_first[0]


def rank_decision_maker(person: dict) -> int:
    """
    Sort key for choosing which decision maker to suggest.

    Owners and presidents outrank HR, which outranks a generic manager.
    """
    title = (person.get("title") or "").lower()
    for rank, keyword in enumerate([
        "owner", "president", "chief executive", "ceo",
        "human resources", "hr ", "general manager", "controller",
        "chief financial", "vice president", "director", "manager",
    ]):
        if keyword in title:
            return rank
    return 99


def blank_contact() -> dict:
    return {
        "Contact Status": "",
        "Contact Note": "",
        "Current First Name": "",
        "Current Last Name": "",
        "Current Title": "",
        "Current LinkedIn": "",
        "Apollo Domain": "",
    }


def resolve_one_contact(row, organization: dict, people: list) -> dict:
    """
    Decide the contact status for one lead.

    Confirmed - our listed person is among the company's current decision makers
    Corrected - our person isn't, but Apollo has someone else who is
    Missing   - Apollo has nobody, so there is no one to ask for by name
    """
    our_first = str(row.get("Executive First Name", "") or "").strip()
    our_last = str(row.get("Executive Last Name", "") or "").strip()
    our_title = str(row.get("Executive Title", "") or "").strip()

    result = blank_contact()
    result["Apollo Domain"] = organization.get("domain", "")

    if not people:
        result["Contact Status"] = STATUS_MISSING
        result["Contact Note"] = (
            "No decision maker on file in Apollo"
            if organization else "Company not found in Apollo"
        )
        return result

    # Is our person still listed?
    for person in people:
        if names_match(our_first, our_last,
                       person.get("first_name", "") or "",
                       person.get("last_name", "") or ""):
            their_title = person.get("title") or ""
            result.update({
                "Contact Status": STATUS_CONFIRMED,
                "Current First Name": person.get("first_name") or "",
                "Current Last Name": person.get("last_name") or "",
                "Current Title": their_title,
                "Current LinkedIn": person.get("linkedin_url") or "",
            })
            # Same person, different title - still confirmed, but say so.
            if our_title and their_title and our_title.lower() not in their_title.lower():
                result["Contact Note"] = f"Title now {their_title} (list said {our_title})"
            else:
                result["Contact Note"] = "Listed contact confirmed"
            return result

    # Our person isn't there. Offer the best decision maker Apollo does have.
    best = sorted(people, key=rank_decision_maker)[0]
    had_name = bool(our_first or our_last)

    result.update({
        "Contact Status": STATUS_CORRECTED,
        "Current First Name": best.get("first_name") or "",
        "Current Last Name": best.get("last_name") or "",
        "Current Title": best.get("title") or "",
        "Current LinkedIn": best.get("linkedin_url") or "",
        "Contact Note": (
            f"Replaces {our_first} {our_last}".strip() if had_name
            else "New contact - list had no name"
        ),
    })
    return result


def resolve_contacts(df: pd.DataFrame, api_key: str, limit: int, progress=None):
    """
    Resolve decision makers for the first `limit` rows.

    Two calls per company: one paid org lookup (cached by company name) and
    one free people search. Rows past the limit come back blank.
    """
    results = {index: blank_contact() for index in df.index}
    diagnostics = {"orgs_found": 0, "orgs_tried": 0, "people_found": 0, "errors": []}

    org_cache = st.session_state.setdefault("apollo_org_cache", {})
    targets = list(df.index)[:limit]

    for position, index in enumerate(targets):
        company = str(df.at[index, "Company Name"] or "").strip()
        if not company:
            results[index]["Contact Status"] = STATUS_MISSING
            results[index]["Contact Note"] = "No company name"
            continue

        try:
            if company not in org_cache:
                diagnostics["orgs_tried"] += 1
                organization = find_organization(
                    company,
                    str(df.at[index, CITY_COLUMN] or "").strip(),
                    str(df.at[index, STATE_COLUMN] or "").strip(),
                    api_key,
                )
                org_cache[company] = {
                    "organization": organization,
                    "people": find_decision_makers(organization, api_key) if organization else [],
                }

            cached = org_cache[company]
            if cached["organization"]:
                diagnostics["orgs_found"] += 1
            diagnostics["people_found"] += len(cached["people"])

            results[index] = resolve_one_contact(
                df.loc[index], cached["organization"], cached["people"]
            )
        except Exception as err:
            results[index]["Contact Status"] = STATUS_MISSING
            results[index]["Contact Note"] = f"Lookup failed: {str(err)[:60]}"
            if len(diagnostics["errors"]) < 3:
                diagnostics["errors"].append(str(err)[:100])

        if progress is not None:
            progress.progress((position + 1) / max(len(targets), 1))

    return df.join(pd.DataFrame.from_dict(results, orient="index")), diagnostics


def color_status(df: pd.DataFrame):
    """Green / yellow / red on the Contact Status column."""
    def shade(column):
        return [
            f"background-color: {STATUS_COLORS.get(value, '')}" if value in STATUS_COLORS else ""
            for value in column
        ]

    if "Contact Status" not in df.columns:
        return df
    return df.style.apply(shade, subset=["Contact Status"])


# ===========================================================================
# Stage 4 - Scoring (xAI)
# ===========================================================================

def lead_to_summary(row) -> str:
    """Condense one lead into the fields that matter for judging fit."""
    contact = f"{row.get('Current First Name', '')} {row.get('Current Last Name', '')}".strip()
    title = row.get("Current Title") or row.get("Executive Title") or "no title"

    return (
        f"{row.get('Company Name', '')} | {row.get(SIZE_COLUMN, 'unknown')} employees | "
        f"{row.get('Primary SIC Code Description', 'unknown industry')} | "
        f"contact: {contact or 'unnamed'}, {title} | "
        f"{row.get('Mailing City', '')}, {row.get('Mailing State', '')}"
    )


def parse_scores(raw: str) -> dict:
    """Turn the model's JSON reply into {row_id: (score, reason)}."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    parsed = {}
    for item in json.loads(cleaned):
        try:
            parsed[int(item["id"])] = (int(item["score"]), str(item.get("reason", "")))
        except (KeyError, ValueError, TypeError):
            continue
    return parsed


def score_leads(df: pd.DataFrame, api_key: str, criteria: str, progress=None) -> pd.DataFrame:
    """Score each lead 1-10 for fit, in batches, and append score plus reason."""
    scores = {index: (None, "") for index in df.index}
    indexes = list(df.index)

    for start in range(0, len(indexes), SCORING_BATCH_SIZE):
        chunk = indexes[start:start + SCORING_BATCH_SIZE]
        lines = [f"{index}: {lead_to_summary(df.loc[index])}" for index in chunk]

        try:
            response = requests.post(
                XAI_URL,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": XAI_MODEL,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content":
                            "You score B2B sales leads. Reply with a JSON array only - no "
                            'prose, no fences. Each element: {"id": <number>, '
                            '"score": <1-10>, "reason": "<12 words max>"}.'},
                        {"role": "user", "content":
                            f"Scoring criteria:\\n{criteria}\\n\\nScore each lead 1 (poor) "
                            f"to 10 (excellent). Use the id at the start of each line.\\n\\n"
                            + "\\n".join(lines)},
                    ],
                },
                timeout=120,
            )
            response.raise_for_status()
            scores.update(parse_scores(response.json()["choices"][0]["message"]["content"]))
        except Exception:
            pass  # leave this batch unscored; the UI reports the gap

        if progress is not None:
            progress.progress(min((start + SCORING_BATCH_SIZE) / len(indexes), 1.0))

    scored = pd.DataFrame(
        [{"Fit Score": scores[i][0], "Score Reason": scores[i][1]} for i in df.index],
        index=df.index,
    )
    return df.join(scored).sort_values("Fit Score", ascending=False, na_position="last")


# ===========================================================================
# UI
# ===========================================================================

def download_row(df: pd.DataFrame, label: str, filename: str, primary: bool = False) -> None:
    """One download button, sized to the frame it's given."""
    st.download_button(
        f"{label} ({len(df):,})",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        type="primary" if primary else "secondary",
    )


def stage_one_size(column_df: pd.DataFrame):
    """Stage 1: headcount threshold."""
    st.subheader("1. Company size")

    if SIZE_COLUMN not in column_df.columns:
        st.info(f"No '{SIZE_COLUMN}' column, so size filtering is off.")
        return column_df

    left, right = st.columns([1, 2])
    minimum = left.number_input("Minimum employees", min_value=0,
                                value=DEFAULT_MIN_EMPLOYEES, step=1)
    include_unknown = right.checkbox("Keep companies with no size listed", value=False)

    result, removed = filter_by_size(column_df, minimum, include_unknown)

    a, b = st.columns(2)
    a.metric("Leads at or above threshold", f"{len(result):,}")
    b.metric("Dropped", f"{removed:,}")

    if result.empty:
        st.info("Nothing left. Try a lower minimum.")
        return result

    with st.expander("Size breakdown"):
        st.dataframe(
            result[SIZE_COLUMN].value_counts(dropna=False)
            .rename_axis("Employee size range").reset_index(name="Companies"),
            use_container_width=True, hide_index=True,
        )

    return result


def stage_two_address(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """Stage 2: verify addresses, keep commercial."""
    st.subheader("2. Mailable commercial address")

    if not SMARTY_AVAILABLE:
        st.info("Smarty SDK not installed. Add smartystreets_python_sdk to requirements.txt.")
        return working_df
    if not SMARTY_AUTH_ID or not SMARTY_AUTH_TOKEN:
        st.info("No Smarty credentials. Add SMARTY_AUTH_ID and SMARTY_AUTH_TOKEN to Secrets.")
        return working_df
    if STREET_COLUMN not in working_df.columns:
        st.info(f"No '{STREET_COLUMN}' column, so there's nothing to verify.")
        return working_df

    has_street = int(
        working_df[STREET_COLUMN].fillna("").astype(str).str.strip().ne("").sum()
    )
    cache_key = f"addr::{file_key}::{len(working_df)}"

    st.caption(f"{has_street:,} rows have a street address. One Smarty lookup each.")

    if st.button(f"Verify {has_street:,} addresses", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = verify_addresses(
                working_df, SMARTY_AUTH_ID, SMARTY_AUTH_TOKEN, progress
            )
        except Exception as err:
            st.error(f"Verification failed: {err}")
            st.caption("401 means the key is wrong; 402 means the lookup balance is empty.")
            return working_df
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df

    verified = st.session_state[cache_key]
    status = verified["Address Status"].value_counts()
    property_type = verified["Property Type"].value_counts()

    a, b, c, d = st.columns(4)
    a.metric("Verified", f"{status.get('Verified', 0):,}")
    b.metric("Corrected", f"{status.get('Corrected', 0):,}")
    c.metric("Commercial", f"{property_type.get('Commercial', 0):,}")
    d.metric("Residential", f"{property_type.get('Residential', 0):,}")

    drop_po = st.checkbox("Also drop PO boxes and general delivery", value=True)
    commercial = keep_commercial(verified, drop_po)

    st.caption(
        f"{len(commercial):,} leads carried forward. "
        f"Excludes {property_type.get('Residential', 0):,} residential"
        + (" and any PO boxes." if drop_po else ".")
    )
    download_row(commercial, "Download non-residential list", f"commercial_{file_key}")

    return commercial


def stage_three_contacts(working_df: pd.DataFrame, file_key: str):
    """Stage 3: confirm, correct, or flag the decision maker."""
    st.subheader("3. Decision maker")

    if not APOLLO_API_KEY:
        st.info(
            "No Apollo key. Generate one under Settings > Integrations > API "
            "with the mixed_companies/search and mixed_people/api_search scopes, "
            "then add APOLLO_API_KEY to Secrets."
        )
        return working_df, None

    limit = st.slider(
        "How many companies to resolve", min_value=10,
        max_value=min(500, max(10, len(working_df))),
        value=min(50, len(working_df)), step=10,
        help="Largest employers first. One paid company lookup each; the "
             "decision maker search itself is free.",
    )

    cache_key = f"contacts::{file_key}::{len(working_df)}::{limit}"

    if st.button(f"Resolve {limit} decision makers", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state[cache_key] = resolve_contacts(
                working_df, APOLLO_API_KEY, limit, progress
            )
        except Exception as err:
            st.error(f"Apollo lookup failed: {err}")
            st.caption("401 means the key is wrong; 403 usually means a missing scope or plan.")
            return working_df, None
        finally:
            progress.empty()

    if cache_key not in st.session_state:
        return working_df, None

    resolved, diagnostics = st.session_state[cache_key]
    counts = resolved["Contact Status"].value_counts()

    a, b, c = st.columns(3)
    a.metric("Confirmed", f"{counts.get(STATUS_CONFIRMED, 0):,}", help="Listed contact still there")
    b.metric("Corrected", f"{counts.get(STATUS_CORRECTED, 0):,}", help="New name from Apollo")
    c.metric("Missing", f"{counts.get(STATUS_MISSING, 0):,}", help="No decision maker found")

    with st.expander("What Apollo reported"):
        st.write({
            "Companies looked up": diagnostics["orgs_tried"],
            "Companies found in Apollo": diagnostics["orgs_found"],
            "Decision makers returned": diagnostics["people_found"],
        })
        for message in diagnostics["errors"]:
            st.error(message)
        if diagnostics["orgs_tried"] and not diagnostics["orgs_found"]:
            st.warning(
                "No companies matched. Either they aren't in Apollo, or the key "
                "lacks the mixed_companies/search scope."
            )

    corrected = resolved[resolved["Contact Status"] == STATUS_CORRECTED]
    if not corrected.empty:
        with st.expander(f"Review the {len(corrected):,} corrected contacts"):
            st.dataframe(
                corrected[[
                    "Company Name", "Executive First Name", "Executive Last Name",
                    "Executive Title", "Current First Name", "Current Last Name",
                    "Current Title", "Contact Note",
                ]],
                use_container_width=True, hide_index=True,
            )

    # Red rows go no further - no name means no one to ask for.
    missing = resolved[resolved["Contact Status"] == STATUS_MISSING]
    actionable = resolved[resolved["Contact Status"] != STATUS_MISSING].copy()

    if not missing.empty:
        download_row(missing, "Download set-aside leads (no contact)", f"no_contact_{file_key}")

    return actionable, missing


def stage_four_score(working_df: pd.DataFrame, file_key: str) -> pd.DataFrame:
    """Stage 4: score what's left."""
    st.subheader("4. Score the remaining leads")

    if not XAI_API_KEY:
        st.info("No xAI key. Add XAI_API_KEY to Secrets to enable scoring.")
        return working_df

    if not st.toggle("Enable scoring", value=False,
                     help="Off by default so it can't spend tokens unattended."):
        return working_df

    criteria = st.text_area("What makes a good lead?",
                            value=DEFAULT_SCORING_CRITERIA, height=160)

    calls = -(-len(working_df) // SCORING_BATCH_SIZE)
    st.caption(f"{len(working_df):,} leads, about {calls} API calls.")

    cache_key = f"scored::{file_key}::{len(working_df)}"

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

    scored = st.session_state[cache_key]
    unscored = int(scored["Fit Score"].isna().sum())
    if unscored:
        st.warning(f"{unscored:,} leads came back unscored. Run again to retry.")

    floor = st.slider("Minimum fit score", 1, 10, 1)
    if floor > 1:
        scored = scored[scored["Fit Score"].fillna(0) >= floor].copy()
        st.caption(f"{len(scored):,} leads at {floor} or above.")

    return scored


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.write(
        "Four stages, in order: filter by size, verify the address is a mailable "
        "commercial location, confirm or correct the decision maker, then score "
        "what's left."
    )

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

    column_df, found, missing_columns = filter_columns(raw_df)
    if not found:
        st.error("None of the expected columns were found. Is this a raw lead export?")
        st.write("Columns in your file:", list(raw_df.columns))
        return
    if missing_columns:
        st.warning("Not in this file: " + ", ".join(missing_columns))

    st.caption(f"{len(raw_df):,} rows in, {len(found)} of {len(KEEP_COLUMNS)} columns kept.")
    st.divider()

    working_df = stage_one_size(column_df)
    if working_df.empty:
        return

    st.divider()
    working_df = stage_two_address(working_df, uploaded_file.name)

    st.divider()
    working_df, set_aside = stage_three_contacts(working_df, uploaded_file.name)

    st.divider()
    working_df = stage_four_score(working_df, uploaded_file.name)

    # --- Final list ---
    st.divider()
    st.subheader("Working list")

    if set_aside is not None and not set_aside.empty:
        st.caption(
            f"{len(working_df):,} leads with a usable contact. "
            f"{len(set_aside):,} set aside with no decision maker."
        )

    st.dataframe(color_status(working_df.head(MAX_PREVIEW_ROWS)), use_container_width=True)

    if len(working_df) > MAX_PREVIEW_ROWS:
        st.caption(f"Showing {MAX_PREVIEW_ROWS:,} rows. The download has all of them.")

    download_row(working_df, "Download working list", f"leads_{uploaded_file.name}", primary=True)


if __name__ == "__main__":
    main()
