#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import csv
import json
import hashlib
import io
from typing import Dict, Any, List, Optional, Tuple

import requests
import streamlit as st
import pandas as pd
import re

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "").strip()
REFERER = "https://localhost"
TITLE = "Metaculus Comment Scorer - Streamlit"

PREFERRED_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]

API2 = "https://www.metaculus.com/api2"
API = "https://www.metaculus.com/api"
UA = {"User-Agent": "metaculus-comments-llm-scorer/0.7 (+python-requests)"}
HTTP = requests.Session()

def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = HTTP.get(url, params=params or {}, headers=UA, timeout=30)
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", "1") or 1)
        time.sleep(min(wait, 10))
        r = HTTP.get(url, params=params or {}, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_recent_questions(n_subjects: int = 10, page_limit: int = 80) -> List[Dict[str, Any]]:
    data = _get(f"{API2}/questions/", {"status": "open", "limit": page_limit})
    results = data.get("results") or data.get("data") or []
    def ts(q):
        return q.get("open_time") or q.get("created_at") or q.get("scheduled_close_time") or ""
    results.sort(key=ts, reverse=True)
    out = []
    for q in results[:n_subjects]:
        qid = q.get("id")
        if not qid:
            continue
        out.append(
            {
                "id": qid,
                "title": q.get("title", ""),
                "url": q.get("page_url")
                or q.get("url")
                or f"https://www.metaculus.com/questions/{qid}/",
            }
        )
    return out

def fetch_question_by_id(qid: int) -> Optional[Dict[str, Any]]:
    try:
        q = _get(f"{API2}/questions/{qid}/")
        if not q or "id" not in q:
            return None
        return {
            "id": q["id"],
            "title": q.get("title", f"Question {qid}"),
            "url": q.get("page_url")
            or q.get("url")
            or f"https://www.metaculus.com/questions/{qid}/",
        }
    except Exception as e:
        print(f"[metaculus] could not fetch question {qid}: {e!r}")
        return None

def fetch_comments_for_post(post_id: int, page_limit: int = 120) -> List[Dict[str, Any]]:
    base = f"{API}/comments/"
    params = {
        "post": post_id,
        "limit": page_limit,
        "offset": 0,
        "sort": "-created_at",
        "is_private": "false",
    }
    out, url = [], base
    while url:
        data = _get(url, params if url == base else None)
        batch = data.get("results") or []
        out += batch
        nxt = data.get("next")
        if nxt:
            url = nxt
            time.sleep(0.2)
        else:
            if batch:
                params["offset"] = params.get("offset", 0) + params.get("limit", page_limit)
                url = base
                time.sleep(0.2)
            else:
                break
    return out

def _ascii(s: str) -> str:
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)

def or_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY.")
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": _ascii(REFERER),
        "X-Title": _ascii(TITLE),
        "User-Agent": _ascii("metaculus-comments-llm-scorer/0.7"),
    }

def list_models_raw() -> List[Dict[str, Any]]:
    r = requests.get(OPENROUTER_MODELS, headers=or_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("data") or data.get("models") or []

def list_models_clean() -> List[Dict[str, Any]]:
    try:
        ms = list_models_raw()
    except Exception as e:
        print("[openrouter] could not list models:", repr(e))
        return []
    out = []
    for m in ms:
        out.append(
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "context_length": m.get("context_length") or m.get("max_context_length"),
                "pricing": m.get("pricing") or {},
                "tags": m.get("tags") or [],
                "arch": m.get("architecture"),
            }
        )
    return out

def pick_model() -> str:
    global OPENROUTER_MODEL
    if OPENROUTER_MODEL:
        return OPENROUTER_MODEL
    ms = list_models_clean()
    if ms:
        ids = {m.get("id"): m for m in ms if m.get("id")}
        for mid in PREFERRED_MODELS:
            if mid in ids:
                return mid
        best_id, best_price = None, 1e9
        for m in ms:
            id_ = (m.get("id") or "").lower()
            tags = " ".join((m.get("tags") or [])).lower()
            arch = (m.get("arch") or "").lower()
            if ("instruct" in id_) or ("instruct" in tags) or ("instruct" in arch):
                pr = m.get("pricing") or {}
                p = pr.get("prompt") or pr.get("input") or 0.0
                try:
                    p = float(p) if p else 0.0
                except Exception:
                    p = 0.0
                if p < best_price:
                    best_price, best_id = p, (m.get("id") or "")
        if best_id:
            OPENROUTER_MODEL = best_id
            return best_id
    return PREFERRED_MODELS[0]

SYSTEM_PROMPT = (
    "You are a strict rater for a forecasting forum.\n\n"
    "Return ONLY valid JSON with keys:\n"
    "- score (integer 1..5)\n"
    "- rationale (string, <=180 chars)\n"
    "- flags (object with booleans: off_topic, toxicity, low_effort, has_evidence, likely_ai)\n"
    "- evidence_urls (array of http/https URLs; <=5, deduplicated; MUST be [] if has_evidence=false)\n\n"
    "Core principle: A good comment ties facts/arguments to how to update the forecast "
    "(priors/base rates, mechanisms, scenarios, timelines, probabilities). Listing facts without linking them "
    "to the forecast deserves a low score.\n\n"
    "Rubric:\n"
    "1 = Toxic/off-topic or irrelevant factual claims.\n"
    "2 = Low effort: generic, facts with no explicit update logic; unclear stance.\n"
    "3 = Adequate: takes a stance and makes at least one explicit link from facts to forecast, but shallow.\n"
    "4 = Good: clear reasoning that updates the forecast; uses base rates/mechanisms/scenarios; cites data.\n"
    "5 = Excellent: structured, novel insight; quantifies impact; multiple credible sources; transparent uncertainty.\n\n"
    "Flags:\n"
    "- off_topic: not about the forecast.\n"
    "- toxicity: insults/harassment/slurs.\n"
    "- low_effort: <50 words or fact list with no explicit forecast linkage.\n"
    "- has_evidence: true only if concrete sources/data are cited (prefer URLs).\n"
    "- likely_ai: style suggests AI; this does NOT cap the score.\n\n"
    "Evidence:\n"
    "- If sources are cited, set has_evidence=true and include up to 5 URLs in evidence_urls.\n"
    "- If no concrete source, has_evidence=false and evidence_urls=[].\n\n"
    "Edge rules:\n"
    "- Be conservative if uncertain.\n"
    "- Rationale must state how the forecast should be updated (direction/size/conditions) in <=180 chars.\n"
    "- Do NOT include any text outside the JSON.\n"
)

FEWSHOTS = [
    {"role": "user", "content": "TEXT: Thanks for sharing!"},
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "score": 1,
                "rationale": "Trivial acknowledgement only.",
                "flags": {
                    "off_topic": False,
                    "toxicity": False,
                    "low_effort": True,
                    "has_evidence": False,
                    "likely_ai": False,
                },
                "evidence_urls": [],
            }
        ),
    },
    {"role": "user", "content": "TEXT: Anyone who thinks this will happen is an idiot."},
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "score": 1,
                "rationale": "Toxic with no evidence.",
                "flags": {
                    "off_topic": False,
                    "toxicity": True,
                    "low_effort": True,
                    "has_evidence": False,
                    "likely_ai": False,
                },
                "evidence_urls": [],
            }
        ),
    },
    {
        "role": "user",
        "content": "TEXT: Turnout fell 3–5% vs 2020 in key counties (CSV). I estimate P(win)=0.56.",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "score": 4,
                "rationale": "Quantified comparison with evidence pointer.",
                "flags": {
                    "off_topic": False,
                    "toxicity": False,
                    "low_effort": False,
                    "has_evidence": True,
                    "likely_ai": False,
                },
                "evidence_urls": [],
            }
        ),
    },
]

def build_msgs(
    qtitle: str, qurl: str, text: str, cid: int, aid: Optional[int], votes: Optional[int]
) -> List[Dict[str, str]]:
    u = (
        "Rate this comment for quality.\n\n"
        f"QUESTION_TITLE: {qtitle}\nQUESTION_URL: {qurl}\n\n"
        f"COMMENT_ID: {cid}\nAUTHOR_ID: {aid}\nVOTE_SCORE: {votes}\nTEXT:\n{text}\n\n"
        'Return JSON: {"score":1|2|3|4|5,"rationale":"...",'
        '"flags":{"off_topic":bool,"toxicity":bool,'
        '"low_effort":bool,"has_evidence":bool,"likely_ai":bool},'
        '"evidence_urls":["..."]}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}] + FEWSHOTS + [
        {"role": "user", "content": u}
    ]

def parse_json_strict(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1:
            return json.loads(s[a : b + 1])
        raise

def call_openrouter(messages: List[Dict[str, str]], model: str, retries: int = 3) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }
    last = None
    for k in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=or_headers(), json=payload, timeout=60)
            if r.status_code == 404:
                raise RuntimeError("404 No endpoints for model")
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After", "2") or 2), 10))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            ch = data.get("choices") or []
            if not ch:
                raise RuntimeError("No choices in response")
            content = ch[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("Empty content")
            return parse_json_strict(content)
        except Exception as e:
            last = e
            time.sleep(0.6 * (k + 1))
    print("[openrouter] giving up:", repr(last))
    return {
        "score": 3,
        "rationale": "Fallback after OR errors.",
        "flags": {
            "off_topic": False,
            "toxicity": False,
            "low_effort": False,
            "has_evidence": False,
            "likely_ai": False,
        },
        "evidence_urls": [],
    }

_cache: Dict[str, Dict[str, Any]] = {}

def score_with_llm(qtitle: str, qurl: str, c: Dict[str, Any], model: str) -> int:
    text = (c.get("text") or "").strip()
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key not in _cache:
        msgs = build_msgs(
            qtitle,
            qurl,
            text,
            c.get("id"),
            (c.get("author") or {}).get("id"),
            c.get("vote_score"),
        )
        resp = call_openrouter(msgs, model)
        if resp.get("rationale", "").startswith("Fallback after"):
            alt = pick_model()
            if alt != model:
                resp = call_openrouter(msgs, alt)
        _cache[key] = resp
    score = _cache[key].get("score", 3)
    try:
        score = int(round(float(score)))
    except Exception:
        score = 3
    return max(1, min(5, score))

def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = ["poster_id", "comment_id", "market_id", "ai_score", "comment_text"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()

def score_recent_questions_streamlit(n: int, comments_limit: int) -> List[Dict[str, Any]]:
    model = pick_model()
    st.info(f"OpenRouter model: **{model}**")
    subjects = fetch_recent_questions(n_subjects=n, page_limit=80)
    rows: List[Dict[str, Any]] = []
    if not subjects:
        st.warning("No questions fetched from Metaculus.")
        return rows
    progress = st.progress(0.0)
    status = st.empty()
    total = len(subjects)
    for i, s in enumerate(subjects, 1):
        sid = s["id"]
        status.markdown(f"### Question {i}/{total} – [{sid}] {s['title']}\n{s['url']}")
        comments = fetch_comments_for_post(sid, page_limit=comments_limit)
        if not comments:
            st.write(f"- No public comments for `{sid}`.")
        else:
            for c in comments:
                text = " ".join((c.get("text") or "").split())
                if not text:
                    continue
                a = c.get("author") or {}
                score = score_with_llm(s["title"], s["url"], c, model)
                rows.append(
                    {
                        "poster_id": a.get("id"),
                        "comment_id": c.get("id"),
                        "market_id": sid,
                        "ai_score": score,
                        "comment_text": text,
                    }
                )
        progress.progress(i / total)
    status.markdown("✅ Done.")
    return rows

def score_specific_qids_streamlit(qids: List[int], comments_limit: int) -> List[Dict[str, Any]]:
    model = pick_model()
    st.info(f"OpenRouter model: **{model}**")
    rows: List[Dict[str, Any]] = []
    total = len(qids)
    progress = st.progress(0.0)
    status = st.empty()
    for i, qid in enumerate(qids, 1):
        status.markdown(f"### Question {i}/{total} – ID `{qid}`")
        subject = fetch_question_by_id(qid)
        if not subject:
            st.warning(f"- Question {qid} not found.")
            progress.progress(i / total)
            continue
        comments = fetch_comments_for_post(subject["id"], page_limit=comments_limit)
        if not comments:
            st.write(f"- No public comments for `{qid}`.")
            progress.progress(i / total)
            continue
        for c in comments:
            text = " ".join((c.get("text") or "").split())
            if not text:
                continue
            a = c.get("author") or {}
            score = score_with_llm(subject["title"], subject["url"], c, model)
            rows.append(
                {
                    "poster_id": a.get("id"),
                    "comment_id": c.get("id"),
                    "market_id": subject["id"],
                    "ai_score": score,
                    "comment_text": text,
                }
            )
        progress.progress(i / total)
    status.markdown("✅ Done.")
    return rows

author_aliases = [
    r"poster_id",
    r"author_id",
    r"user_id",
    r"commenter_id",
    r"account_id",
    r"poster",
    r"author",
    r"user",
]
score_aliases = [r"ai_score", r"score", r"rating", r"model_score"]

def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for p in patterns:
        for c in cols:
            if c.lower() == p.lower():
                return c
    for p in patterns:
        r = re.compile(p, re.I)
        for c in cols:
            if r.search(c):
                return c
    return None

def aggregate_author_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    auth_col = find_col(df, author_aliases)
    score_col = find_col(df, score_aliases)
    if not auth_col or not score_col:
        raise ValueError(
            f"Columns not found.\nAuthor column: {auth_col}\nScore column: {score_col}\nAvailable: {list(df.columns)}"
        )
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").clip(1, 5)
    agg = (
        df.dropna(subset=[auth_col, score_col])
        .groupby(auth_col, as_index=False)[score_col]
        .agg(total="sum", count="size", mean="mean")
        .sort_values("total", ascending=False)
    )
    return agg, auth_col, score_col

st.set_page_config(page_title="Metaculus Comment Scorer", layout="wide")
st.title("🔍 Metaculus Comment Scorer (OpenRouter + LLM)")

st.sidebar.header("Configuration")

default_key = OPENROUTER_API_KEY or ""
api_key = st.sidebar.text_input(
    "OpenRouter API key",
    type="password",
    value=default_key,
    help="You can also set it via the OPENROUTER_API_KEY environment variable.",
)
if api_key:
    OPENROUTER_API_KEY = api_key.strip()

mode = st.sidebar.radio(
    "Mode",
    ["List models", "Score recent questions", "Score specific IDs", "Aggregate author scores from CSV"],
)

st.sidebar.markdown("---")
comments_limit = st.sidebar.number_input(
    "Max comments per question", min_value=10, max_value=500, value=120, step=10
)

if mode == "List models":
    st.subheader("Accessible OpenRouter models")
    if not OPENROUTER_API_KEY:
        st.info("Enter your OpenRouter API key in the sidebar to list models.")
    else:
        try:
            models = list_models_clean()
            if not models:
                st.write("No models visible (invalid key or no access).")
            else:
                df = pd.DataFrame(
                    [
                        {
                            "id": x.get("id"),
                            "context_length": x.get("context_length"),
                            "pricing_in": (x.get("pricing") or {}).get("prompt")
                            or (x.get("pricing") or {}).get("input"),
                            "pricing_out": (x.get("pricing") or {}).get("completion")
                            or (x.get("pricing") or {}).get("output"),
                            "tags": ", ".join(x.get("tags") or []),
                        }
                        for x in models
                    ]
                )
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error while fetching models: {e}")

elif mode == "Score recent questions":
    if not OPENROUTER_API_KEY:
        st.warning("Enter your OpenRouter API key in the sidebar to use this mode.")
    else:
        st.subheader("Score recent Metaculus questions")
        n = st.number_input(
            "Number of recent questions",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
        )
        if st.button("Run scoring on recent questions"):
            with st.spinner("Fetching and scoring comments..."):
                try:
                    rows = score_recent_questions_streamlit(n=n, comments_limit=comments_limit)
                except Exception as e:
                    st.error(f"Error during scoring: {e}")
                    rows = []
            if rows:
                df = pd.DataFrame(rows)
                st.success(f"{len(rows)} comments scored.")
                st.dataframe(df, use_container_width=True)
                csv_data = rows_to_csv(rows)
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="metaculus_comment_scores_recent.csv",
                    mime="text/csv",
                )
            else:
                st.info("No comments were scored.")

elif mode == "Score specific IDs":
    if not OPENROUTER_API_KEY:
        st.warning("Enter your OpenRouter API key in the sidebar to use this mode.")
    else:
        st.subheader("Score specific Metaculus question IDs")
        qids_str = st.text_area(
            "Metaculus IDs (comma or space separated)",
            placeholder="Example: 12345, 67890, 13579",
        )
        qids: List[int] = []
        if qids_str.strip():
            for chunk in qids_str.replace(",", " ").split():
                try:
                    qids.append(int(chunk))
                except ValueError:
                    pass
        if st.button("Run scoring on these IDs"):
            if not qids:
                st.warning("Please enter at least one valid ID.")
            else:
                with st.spinner("Fetching and scoring comments..."):
                    try:
                        rows = score_specific_qids_streamlit(
                            qids=qids, comments_limit=comments_limit
                        )
                    except Exception as e:
                        st.error(f"Error during scoring: {e}")
                        rows = []
                if rows:
                    df = pd.DataFrame(rows)
                    st.success(f"{len(rows)} comments scored.")
                    st.dataframe(df, use_container_width=True)
                    csv_data = rows_to_csv(rows)
                    st.download_button(
                        "Download CSV",
                        data=csv_data,
                        file_name="metaculus_comment_scores_qids.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No comments were scored.")

elif mode == "Aggregate author scores from CSV":
    st.subheader("Aggregate author scores from a CSV of comments")
    uploaded = st.file_uploader(
        "Upload CSV with comment scores",
        type=["csv"],
        help="Can be a CSV produced by this app or an external one.",
    )
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_in = None
        if df_in is not None:
            st.markdown("Preview of input CSV:")
            st.dataframe(df_in.head(), use_container_width=True)
            try:
                agg_df, auth_col, score_col = aggregate_author_scores(df_in)
                st.success(f"Aggregated by author column `{auth_col}` with score column `{score_col}`.")
                st.dataframe(agg_df, use_container_width=True)
                csv_buf = io.StringIO()
                agg_df.to_csv(csv_buf, index=False)
                agg_bytes = csv_buf.getvalue().encode("utf-8")
                st.download_button(
                    "Download author ranking CSV",
                    data=agg_bytes,
                    file_name="author_totals.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))

