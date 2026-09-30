"""Daily market data only; no old V7 Edge, confidence or ranking functions.

Parsers inherited from V7.6.4 and kept strict. One download carries raw OHLCV
and Adj Close. All derived return estimates elsewhere are exploratory.
"""
from __future__ import annotations
import math, os, logging, time, re, json, gzip, sqlite3, hashlib, zlib, socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import numpy as np
import pandas as pd
import requests
from operational_tools import safe_error_text, runtime_inventory


def _load_yahoo_backend():
    """Native extension failures must not prevent the HTTP/cache-only app loading.

    This does not fix a pip install failure. No requests.Session is injected into
    older yfinance versions that explicitly require curl_cffi.
    """
    try:
        import importlib
        module = importlib.import_module("yfinance")
        return module, None
    except (ImportError, OSError) as exc:
        return None, f"{type(exc).__name__}: {safe_error_text(exc, 160)}"

def _backend_load_failure(exc):
    if isinstance(exc, (requests.RequestException, TimeoutError, ConnectionError, socket.gaierror)):
        return False
    return isinstance(exc, (ImportError, OSError))


def provider_runtime_status(store=None):
    status = runtime_inventory()
    status.update({"sqlite_runtime": sqlite3.sqlite_version,
                   "wal_fixed_by_version": sqlite_wal_fixed(),
                   "yahoo_backend_loaded": yf is not None,
                   "yahoo_import_diagnostic": safe_error_text(YAHOO_IMPORT_DIAGNOSTIC or "", 240),
                   "finmind_token_configured": bool(_finmind_token()),
                   "dataset_access_verified": False})
    if store is not None and hasattr(store, "health"):
        status["cache"] = store.health()
    return status


yf, YAHOO_IMPORT_DIAGNOSTIC = _load_yahoo_backend()
logger=logging.getLogger(__name__)
@dataclass(frozen=True)
class DataConfig:
    batch_size: int=32
    download_retries: int=1
    individual_fallback_limit: int=6
    finmind_workers: int=2
    finmind_timeout: int=12
CFG=DataConfig()
PARSER_VERSION="8.1-dual-v764"
OHLCV=["Open","High","Low","Close","Volume"]
INDUSTRY_MAP={"01":"Cement","02":"Food","03":"Plastics","04":"Textiles",
"05":"Electrical machinery","06":"Wires","07":"Chemicals/Biotech","08":"Glass",
"09":"Paper","10":"Steel","11":"Rubber","12":"Automobiles","14":"Construction",
"15":"Shipping","16":"Tourism","17":"Financials","18":"Retail","20":"Other",
"21":"Chemicals","22":"Biotech","23":"Energy","24":"Semiconductors","25":"Computers",
"26":"Optoelectronics","27":"Communications","28":"Electronic components",
"29":"Electronics distribution","30":"Information services","31":"Other electronics",
"32":"Culture","33":"Agriculture","34":"Ecommerce","35":"Green energy",
"36":"Digital services","37":"Sports","38":"Home","39":"Other industries"}
def _finmind_token():
    value=os.getenv("FINMIND_TOKEN","").strip()
    if value: return value
    try:
        import streamlit as st
        return str(st.secrets.get("FINMIND_TOKEN","")).strip()
    except Exception:
        return ""


def safe_float(v, default=np.nan):
    """Convert a scalar only; missing, non-finite and vector values stay missing."""
    if v is None or isinstance(v,(pd.Series,pd.DataFrame,np.ndarray,list,tuple,dict,set)):
        return default
    try:
        value=float(v)
    except (TypeError,ValueError,OverflowError):
        return default
    return value if math.isfinite(value) else default

def _first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm = {re.sub(r"\s+", "", str(c)).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", "", str(cand)).lower()
        if key in norm:
            return norm[key]
    return None

def _normalize_roster(df: pd.DataFrame, market: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker","name","industry","market"])
    code_col = _first_col(df,["公司代號","SecuritiesCompanyCode","公司代碼","Code"])
    name_col = _first_col(df,["公司簡稱","CompanyAbbreviation","公司名稱","CompanyName","Name"])
    ind_col = _first_col(df,["產業別","SecuritiesIndustryCode","產業代號","Industry"])
    if not code_col or not name_col:
        return pd.DataFrame(columns=["ticker","name","industry","market"])
    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    out["name"] = df[name_col].astype(str).str.strip()
    if ind_col:
        raw_ind = df[ind_col].astype(str).str.strip()
        def map_ind(v):
            vv = re.sub(r"\.0$", "", str(v)).zfill(2) if re.fullmatch(r"\d+(?:\.0)?", str(v)) else str(v)
            return INDUSTRY_MAP.get(vv, vv if vv and vv.lower() != "nan" else "其他")
        out["industry"] = raw_ind.map(map_ind)
    else:
        out["industry"] = "其他"
    out = out[out["code"].str.fullmatch(r"\d{4}", na=False)].copy()
    out = out[out["name"].ne("") & ~out["name"].str.contains("存託|特別股|權證", na=False)]
    suffix = ".TW" if market == "TWSE" else ".TWO"
    out["ticker"] = out["code"] + suffix
    out["market"] = market
    return out[["ticker","name","industry","market"]].drop_duplicates("ticker")

def _request_as_of(as_of: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Resolve now once; an explicitly invalid timestamp must never mean now."""
    if as_of is None:
        return _taipei_timestamp()
    parsed = _known_timestamp(as_of)
    if pd.isna(parsed):
        raise ValueError("as_of must be a valid timestamp (None means now)")
    return parsed

def _period_start_date(period: str, as_of: Optional[pd.Timestamp] = None) -> str:
    """Lookback anchored to the supplied Taipei timestamp, including warm-up.

    Calendar dates retain the app's midnight semantics. The downloader anchors
    this helper to the last eligible daily bar, not the wall clock of a worker.
    """
    base = _request_as_of(as_of).normalize()
    value = str(period).strip().lower()
    m = re.fullmatch(r"([1-9]\d*)(y|mo|wk|d)", value)
    if m:
        count, unit = int(m.group(1)), m.group(2)
        unit_name = {"y": "years", "mo": "months", "wk": "weeks", "d": "days"}[unit]
        padding = 20 if unit == "y" else 10
        offset = pd.DateOffset(**{unit_name: count})
        return (base - offset - pd.Timedelta(days=padding)).strftime("%Y-%m-%d")
    if value == "ytd":
        return (base.replace(month=1, day=1) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    if value == "max":
        return "1900-01-01"
    raise ValueError(f"Unsupported history period: {period!r}")

def _price_as_of_key(as_of: Optional[pd.Timestamp] = None) -> str:
    """Daily-price cache key; same eligible day shares data, never different days.

    16:00 is a canonical internal anchor, not an actual provider publication time.
    All source-event timestamps remain separate in the calibration snapshot.
    """
    cutoff = _daily_cutoff(_request_as_of(as_of))
    return (cutoff + pd.Timedelta(hours=16)).isoformat()

def _price_request_window(period: str, as_of: Optional[pd.Timestamp] = None) -> dict:
    """Explicit bounds: Yahoo end is exclusive; FinMind end_date is the cutoff."""
    resolved = _request_as_of(as_of)
    key = _price_as_of_key(resolved)
    cutoff = _daily_cutoff(key)
    start = _period_start_date(period, key)
    if pd.Timestamp(start) > cutoff:
        raise ValueError("History start must not be after the daily cutoff")
    return {
        "as_of": resolved.isoformat(),
        "price_as_of_key": key,
        "request_start": start,
        "data_cutoff": cutoff.strftime("%Y-%m-%d"),
        "yahoo_end_exclusive": (cutoff + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "finmind_end_date": cutoff.strftime("%Y-%m-%d"),
    }

def _trim_daily_request(frame: pd.DataFrame, window: dict) -> pd.DataFrame:
    """Enforce bounds locally too, even if a provider ignores start/end."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    try:
        d = _normalize_daily_index(frame)
        return d.loc[(d.index >= pd.Timestamp(window["request_start"]))
                     & (d.index <= pd.Timestamp(window["data_cutoff"]))].copy()
    except (PriceDataError, ValueError, TypeError, OverflowError) as exc:
        logger.warning("Price-date window rejected: %s", exc)
        return pd.DataFrame()

class PriceDataError(ValueError):
    """The provider response is ambiguous or cannot be interpreted safely."""

def _taipei_timestamp(value=None) -> pd.Timestamp:
    t=pd.Timestamp.now(tz="Asia/Taipei") if value is None else pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("Invalid as-of timestamp")
    if t.tzinfo is not None:
        t=t.tz_convert("Asia/Taipei").tz_localize(None)
    return t

def _daily_cutoff(value=None) -> pd.Timestamp:
    """Conservative daily-bar policy; never treat an unfinished bar as closed.

    14:00 Taipei is an application policy, NOT a promise that every provider
    has already published all exchange data at that time.
    """
    t=_taipei_timestamp(value)
    day=t.normalize()
    return day if t.hour>=14 else day-pd.Timedelta(days=1)

def _field_label(v) -> Optional[str]:
    key=re.sub(r"[\s_]","",str(v)).lower()
    return {"open":"Open","high":"High","low":"Low","close":"Close",
            "adjclose":"Adj Close","adjustedclose":"Adj Close","volume":"Volume"}.get(key)

def _flatten_price_columns(frame: pd.DataFrame,ticker: Optional[str]=None) -> pd.DataFrame:
    """Identify the field axis, select a symbol, then verify uniqueness.

    Never discard a variable symbol/source level: that can merge two stocks
    into duplicate Close columns. Identical duplicates may be collapsed;
    conflicting duplicate columns are rejected.
    """
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    d=frame.copy()
    if isinstance(d.columns,pd.MultiIndex):
        levels=[]
        for level in range(d.columns.nlevels):
            labels={_field_label(v) for v in d.columns.get_level_values(level)}
            if {"Open","High","Low","Close"}.issubset(labels):
                levels.append(level)
        if len(levels)!=1:
            raise PriceDataError("Cannot identify one unambiguous OHLC field level")
        field_level=levels[0]
        if ticker is not None:
            want=str(ticker).strip().upper()
            symbol_levels=[]
            for level in range(d.columns.nlevels):
                if level==field_level: continue
                vals=[str(v).strip().upper() for v in d.columns.get_level_values(level)]
                if want in vals: symbol_levels.append(level)
            if len(symbol_levels)!=1:
                raise PriceDataError(f"Requested symbol {want} missing or ambiguous")
            lv=symbol_levels[0]
            mask=[str(v).strip().upper()==want for v in d.columns.get_level_values(lv)]
            d=d.loc[:,mask]
        for level in range(d.columns.nlevels):
            if level!=field_level and d.columns.get_level_values(level).nunique(dropna=False)>1:
                raise PriceDataError("Unresolved variable level after symbol selection")
        labels=[_field_label(v) for v in d.columns.get_level_values(field_level)]
    else:
        labels=[_field_label(v) for v in d.columns]
    result=pd.DataFrame(index=d.index)
    for field in ["Open","High","Low","Close","Adj Close","Volume"]:
        positions=[i for i,label in enumerate(labels) if label==field]
        if not positions: continue
        first=pd.to_numeric(d.iloc[:,positions[0]],errors="coerce")
        for position in positions[1:]:
            other=pd.to_numeric(d.iloc[:,position],errors="coerce")
            if not np.allclose(first.to_numpy(dtype=float),other.to_numpy(dtype=float),equal_nan=True,rtol=0,atol=0):
                raise PriceDataError(f"Conflicting duplicate field: {field}")
        result[field]=first
    if not {"Open","High","Low","Close","Volume"}.issubset(result.columns):
        raise PriceDataError("Missing required OHLCV fields")
    if isinstance(result.columns,pd.MultiIndex) or not result.columns.is_unique:
        raise PriceDataError("Price columns must be flat and unique after symbol selection")
    if any(not isinstance(result[field],pd.Series) for field in result.columns):
        raise PriceDataError("A price field must resolve to exactly one Series")
    return result

def _canonical_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse already canonical numeric columns without rebuilding every Series.

    The extractor's flat output normally takes this path. MultiIndex, duplicate,
    alias, object or ambiguous fields still go through the strict parser. Never
    strip a variable symbol/source level or trust a caller-supplied marker.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    allowed = required | {"Adj Close"}
    if (isinstance(frame, pd.DataFrame) and not frame.empty
            and not isinstance(frame.columns, pd.MultiIndex)
            and frame.columns.is_unique
            and required.issubset(frame.columns)
            and set(frame.columns).issubset(allowed)
            and all(pd.api.types.is_numeric_dtype(dtype)
                    and not pd.api.types.is_complex_dtype(dtype)
                    and not pd.api.types.is_bool_dtype(dtype)
                    for dtype in frame.dtypes)):
        # Downstream date normalization and validation make copies before edits.
        return frame
    return _flatten_price_columns(frame)

def _known_timestamp(value) -> pd.Timestamp:
    """Parse a supplied timestamp. Missing is NEVER an instruction to use now()."""
    if value is None or isinstance(value,(bool,int,float,np.integer,np.floating)):
        return pd.NaT
    if isinstance(value,(pd.Series,pd.DataFrame,np.ndarray,list,tuple,dict,set)):
        return pd.NaT
    try:
        if pd.isna(value) or (isinstance(value,str) and not value.strip()):
            return pd.NaT
        return _taipei_timestamp(value)
    except (ValueError,TypeError,OverflowError):
        return pd.NaT

def _normalize_daily_index(d: pd.DataFrame) -> pd.DataFrame:
    if isinstance(d.index,pd.RangeIndex) or pd.api.types.is_numeric_dtype(d.index.dtype):
        raise PriceDataError("A dated index is required")
    if isinstance(d.index,pd.DatetimeIndex):
        idx=d.index
        if idx.tz is not None: idx=idx.tz_convert("Asia/Taipei").tz_localize(None)
        idx=idx.normalize()
    else:
        def convert(v):
            t=_known_timestamp(v)
            return t.normalize() if pd.notna(t) else pd.NaT
        idx=pd.DatetimeIndex([convert(v) for v in d.index])
    keep=~idx.isna()
    out=d.loc[keep].copy(); out.index=idx[keep]
    return out.loc[~out.index.duplicated(keep="last")].sort_index()

def _align_price_pair(signal,execution,as_of=None):
    s=_normalize_ohlcv_frame(signal); e=_normalize_ohlcv_frame(execution)
    if as_of is not None:
        cutoff=_daily_cutoff(as_of)
        s=s.loc[s.index<=cutoff]; e=e.loc[e.index<=cutoff]
    if s.empty or e.empty or s.index[-1]!=e.index[-1]:
        return pd.DataFrame(),pd.DataFrame()
    dates=s.index.intersection(e.index)
    return s.loc[dates].copy(),e.loc[dates].copy()

def _slice_pair(pair: dict,as_of):
    if not isinstance(pair,dict): return {}
    s,e=_align_price_pair(pair.get("signal"),pair.get("exec"),as_of)
    if s.empty or e.empty: return {}
    return {**{k:v for k,v in pair.items() if k not in {"signal","exec"}},"signal":s,"exec":e}

def _normalize_ohlcv_frame(d: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(d,pd.DataFrame) or d.empty: return pd.DataFrame()
    try:
        x=_normalize_daily_index(_canonical_price_frame(d))
        x=x[["Open","High","Low","Close","Volume"]].replace([np.inf,-np.inf],np.nan)
        finite=np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
        tol=x["Close"].abs()*1e-10+1e-10
        valid=(finite & (x[["Open","High","Low","Close"]]>0).all(axis=1) & (x["Volume"]>=0)
               & (x["High"]+tol>=x[["Open","Close","Low"]].max(axis=1))
               & (x["Low"]-tol<=x[["Open","Close","High"]].min(axis=1)))
        out=x.loc[valid].copy()
        out.attrs["invalid_rows_removed"]=int((~valid).sum())
        return out
    except (PriceDataError,TypeError,ValueError) as exc:
        logger.warning("OHLCV rejected: %s",exc)
        return pd.DataFrame()

def _finmind_fetch_dataset(ticker: str, period: str, dataset: str, token: str = "", as_of: Optional[pd.Timestamp] = None) -> Tuple[str, pd.DataFrame, str]:
    """Fetch one explicit FinMind daily-price dataset inside the frozen window."""
    window = _price_request_window(period, as_of)
    code = str(ticker).split(".")[0].strip()
    if not code.isdigit():
        return ticker, pd.DataFrame(), "not a Taiwan numeric ticker"
    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"User-Agent": "TaiwanAlphaRadar/7.6"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {
        "dataset": dataset,
        "data_id": code,
        "start_date": window["request_start"],
        "end_date": window["finmind_end_date"],
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=CFG.finmind_timeout)
        if resp.status_code != 200:
            return ticker, pd.DataFrame(), f"HTTP {resp.status_code}"
        payload = resp.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            msg = payload.get("msg", "empty") if isinstance(payload, dict) else "empty"
            return ticker, pd.DataFrame(), safe_error_text(msg, 120, (token,))
        x = pd.DataFrame(rows)
        if "date" not in x.columns:
            return ticker, pd.DataFrame(), "missing date"
        cmap = {
            "open": "Open", "max": "High", "min": "Low", "close": "Close",
            "Trading_Volume": "Volume", "trading_volume": "Volume",
            "Trading_volume": "Volume", "volume": "Volume",
        }
        x = x.rename(columns={k:v for k,v in cmap.items() if k in x.columns})
        if not {"Open","High","Low","Close","Volume"}.issubset(x.columns):
            return ticker, pd.DataFrame(), "missing OHLCV columns"
        x.index = pd.to_datetime(x["date"], errors="coerce")
        d = _normalize_ohlcv_frame(x[["Open","High","Low","Close","Volume"]])
        d = _trim_daily_request(d, window)
        if len(d) < 60:
            return ticker, pd.DataFrame(), f"only {len(d)} rows"
        return ticker, d, dataset
    except Exception as e:
        return ticker, pd.DataFrame(), f"{type(e).__name__}: {safe_error_text(e, 120, (token,))}"

def _extract_yahoo_ticker_frame(raw: pd.DataFrame,ticker: str,single: bool=False) -> pd.DataFrame:
    """Select a symbol without conflating a field-only axis with a symbol axis.

    A one-level MultiIndex can contain only OHLCV field names. Accept that
    layout solely for a request known to contain one symbol. Multi-symbol
    responses still require an explicit, unambiguous symbol dimension.
    Do not blindly droplevel(0): the first level may be prices OR symbols.
    """
    if not isinstance(raw,pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    try:
        multi=isinstance(raw.columns,pd.MultiIndex)
        field_only=not multi or raw.columns.nlevels==1
        if field_only:
            if not single:
                raise PriceDataError("Multi-symbol response lacks a symbol dimension")
            selected=_flatten_price_columns(raw)
        else:
            selected=_flatten_price_columns(raw,ticker)
        if isinstance(selected.columns,pd.MultiIndex) or not selected.columns.is_unique:
            raise PriceDataError("Selected price columns must be flat and unique")
        return selected
    except (PriceDataError,ValueError,TypeError) as exc:
        logger.warning("Yahoo parse %s rejected: %s",ticker,exc)
        empty=pd.DataFrame(); empty.attrs["parse_error"]=str(exc)
        return empty

def _dual_from_yahoo_frame(d: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,bool]:
    """Align flat, single-symbol tracks without filling unknown adjustments."""
    if not isinstance(d,pd.DataFrame) or d.empty:
        return pd.DataFrame(),pd.DataFrame(),True
    try:
        flat=_normalize_daily_index(_canonical_price_frame(d))
        if isinstance(flat.columns,pd.MultiIndex) or not flat.columns.is_unique:
            raise PriceDataError("Ambiguous fields remain after flattening")
        execution=_normalize_ohlcv_frame(flat)
        if execution.empty:
            return pd.DataFrame(),pd.DataFrame(),True
        if "Adj Close" not in flat:
            return execution.copy(),execution,True
        # Uniqueness checks guarantee Series/Series arithmetic on one date index.
        adj=pd.to_numeric(flat["Adj Close"],errors="coerce").reindex(execution.index)
        raw_close=pd.to_numeric(execution["Close"],errors="coerce")
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            factor=(adj/raw_close.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
        valid=factor.notna() & (factor>0)
        if not valid.any():
            return execution.copy(),execution,True
        signal=execution.loc[valid].copy()
        with np.errstate(over="ignore",invalid="ignore"):
            for c in ["Open","High","Low","Close"]:
                signal[c]=signal[c]*factor.loc[valid]
        signal=_normalize_ohlcv_frame(signal)
        degraded=not bool(valid.all()) or len(signal)!=len(execution)
        return signal,execution,degraded
    except (PriceDataError,ValueError,TypeError,OverflowError) as exc:
        logger.warning("Dual-track response rejected: %s",exc)
        return pd.DataFrame(),pd.DataFrame(),True

def _download_live(tickers_tuple: tuple, period: str = "5y", as_of: Optional[pd.Timestamp] = None) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], dict]:
    """One-pass dual-price loader.

    Yahoo is requested once with auto_adjust=False; Adj Close is used to derive
    the corporate-action-consistent signal OHLC while raw OHLC remains the
    execution/chart track. Missing Taiwan symbols fall back to FinMind adjusted
    + raw datasets. This preserves the multi-source resilience without doing
    two full Yahoo downloads.
    """
    request_asof = _request_as_of(as_of)
    window = _price_request_window(period, request_asof)
    tickers = list(dict.fromkeys(str(t).strip() for t in tickers_tuple if str(t).strip()))
    result: Dict[str, Dict[str,pd.DataFrame]] = {}
    source_by_ticker: Dict[str,str] = {}
    diag = {
        "requested":len(tickers), "downloaded":0, "batch_recovered":0,
        "individual_recovered":0, "finmind_recovered":0, "failed_tickers":[],
        "errors":[], "provider":"Yahoo dual-track + FinMind fallback",
        "source_counts":{}, "source_by_ticker":source_by_ticker,
        "execution_degraded":[], "signal_adjustment_degraded":[],
        "request_window":window,
    }

    yahoo_available = yf is not None
    if not yahoo_available:
        diag["errors"].append("Yahoo backend unavailable: " + str(YAHOO_IMPORT_DIAGNOSTIC or "missing dependency"))
        diag["provider_degraded"] = True
    yahoo_tickers = tickers if yahoo_available else []
    for i in range(0, len(yahoo_tickers), CFG.batch_size):
        batch = yahoo_tickers[i:i+CFG.batch_size]
        raw = pd.DataFrame(); last_error = None
        for attempt in range(CFG.download_retries+1):
            try:
                kwargs = dict(tickers=batch, start=window["request_start"], end=window["yahoo_end_exclusive"], interval="1d", group_by="ticker",
                              auto_adjust=False, actions=False, threads=4, progress=False, timeout=25)
                try:
                    raw = yf.download(repair=False, **kwargs)
                except TypeError:
                    kwargs.pop("timeout", None); raw = yf.download(**kwargs)
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    break
            except Exception as e:
                last_error = e
                if _backend_load_failure(e):
                    # A native loader error is not a transient network exception.
                    yahoo_available = False
                    diag["provider_degraded"] = True
                    break
            if attempt < CFG.download_retries:
                time.sleep(.6*(attempt+1))
        if last_error is not None and (raw is None or raw.empty):
            diag["errors"].append(f"Yahoo batch {type(last_error).__name__}: {safe_error_text(last_error, 110)}")
        before = len(result)
        for t in batch:
            d = _extract_yahoo_ticker_frame(raw, t, single=(len(batch)==1))
            if d.empty and d.attrs.get("parse_error"):
                diag["errors"].append(f"{t}: {d.attrs['parse_error']}")
            d = _trim_daily_request(d, window)
            signal, execution, degraded = _dual_from_yahoo_frame(d)
            if not signal.empty and not execution.empty:
                result[t] = {"signal":signal, "exec":execution, "execution_verified":True, "signal_adjustment_verified":not degraded, "source":"Yahoo"}
                source_by_ticker[t] = "Yahoo-dual"
                if degraded:
                    diag["signal_adjustment_degraded"].append(t)
        diag["batch_recovered"] += len(result)-before
        if not yahoo_available:
            break

    missing = [t for t in tickers if t not in result]
    token = _finmind_token()
    # A missing Yahoo import must not kill independent FinMind HTTP recovery.
    fallback_limit = max(CFG.batch_size, CFG.individual_fallback_limit) if not yahoo_available else CFG.individual_fallback_limit
    finmind_targets = [t for t in missing if str(t).split(".")[0].isdigit()][:fallback_limit] if token else []
    if not yahoo_available and not token:
        diag["errors"].append("No independent live provider configured; existing verified disk cache remains usable. Set FINMIND_TOKEN with dataset access, or repair Yahoo installation.")
    if finmind_targets:
        def fetch_dual(t):
            _,adj,s1=_finmind_fetch_dataset(t,period,"TaiwanStockPriceAdj",token,as_of=request_asof)
            _,rawp,s2=_finmind_fetch_dataset(t,period,"TaiwanStockPrice",token,as_of=request_asof)
            if rawp.empty:
                return t,pd.DataFrame(),pd.DataFrame(),f"Raw price unavailable: {s2}"
            if adj.empty:
                return t,rawp.copy(),rawp,"signal_unadjusted"
            adj,rawp=_align_price_pair(adj,rawp,request_asof)
            return t,adj,rawp,"FinMind-dual"
        with ThreadPoolExecutor(max_workers=max(1, CFG.finmind_workers)) as ex:
            futs = {ex.submit(fetch_dual, t):t for t in finmind_targets}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    _, adj, rawp, src = fut.result()
                except Exception as e:
                    adj=pd.DataFrame(); rawp=pd.DataFrame(); src=f"{type(e).__name__}: {safe_error_text(e, 80, (token,))}"
                if not adj.empty and not rawp.empty:
                    result[t] = {"signal":adj, "exec":rawp, "execution_verified":True, "signal_adjustment_verified":src!="signal_unadjusted", "source":"FinMind"}
                    source_by_ticker[t] = "FinMind-dual"
                    diag["finmind_recovered"] += 1
                    if src == "signal_unadjusted":
                        diag["signal_adjustment_degraded"].append(t)
                else:
                    diag["errors"].append(f"{t}: FinMind {safe_error_text(src, 160, (token,))}")

    missing2 = [t for t in tickers if t not in result]
    for t in (missing2[:CFG.individual_fallback_limit] if yahoo_available else []):
        try:
            kwargs = dict(tickers=t, start=window["request_start"], end=window["yahoo_end_exclusive"], interval="1d", group_by="ticker",
                          auto_adjust=False, actions=False, threads=False, progress=False, timeout=20)
            try:
                raw = yf.download(repair=False, **kwargs)
            except TypeError:
                kwargs.pop("timeout",None); raw = yf.download(**kwargs)
            d = _extract_yahoo_ticker_frame(raw, t, single=True)
            d = _trim_daily_request(d, window)
            signal, execution, degraded = _dual_from_yahoo_frame(d)
            if not signal.empty and not execution.empty:
                result[t] = {"signal":signal, "exec":execution, "execution_verified":True, "signal_adjustment_verified":not degraded, "source":"Yahoo"}
                source_by_ticker[t] = "Yahoo-single-dual"
                diag["individual_recovered"] += 1
                if degraded:
                    diag["signal_adjustment_degraded"].append(t)
        except Exception as e:
            diag["errors"].append(f"{t}: Yahoo single {type(e).__name__}: {safe_error_text(e, 90, (token,))}")
            if _backend_load_failure(e):
                diag["provider_degraded"] = True
                break

    counts={}
    for src in source_by_ticker.values(): counts[src]=counts.get(src,0)+1
    diag["source_counts"] = counts
    diag["downloaded"] = len(result)
    diag["failed_tickers"] = [t for t in tickers if t not in result]
    diag["errors"] = diag["errors"][:12]
    return result, diag


def normalize_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """CSV contract; never truncate a 6-digit ETF into a 4-digit equity code."""
    if not isinstance(frame,pd.DataFrame) or not {'ticker','name','industry'}.issubset(frame.columns):
        raise PriceDataError('Universe requires ticker,name,industry')
    out=frame.copy()
    out['ticker']=out['ticker'].astype(str).str.strip().str.upper()
    out=out[out['ticker'].str.fullmatch(r'\d{4}\.(?:TW|TWO)',na=False)].copy()
    for col in ['name','industry']:
        out[col]=out[col].fillna('Unknown').astype(str)
    out['market']=np.where(out['ticker'].str.endswith('.TWO'),'TPEx','TWSE')
    return out[['ticker','name','industry','market']].drop_duplicates('ticker').sort_values('ticker').reset_index(drop=True)


def get_universe(data_dir: Path, refresh: bool=False) -> tuple[pd.DataFrame,dict]:
    """Current roster only. This function does not recreate historical constituents."""
    data_dir=Path(data_dir); data_dir.mkdir(parents=True,exist_ok=True)
    custom=data_dir/'universe.csv'
    cache=data_dir/'roster.json'
    if custom.exists():
        u=normalize_universe(pd.read_csv(custom,dtype={'ticker':str}))
        if u.empty: raise PriceDataError('The supplied universe.csv has no supported equities')
        return u,{'source':'custom_universe','full_market_verified':False,'total':len(u),'warnings':[]}
    if cache.exists() and not refresh:
        try:
            c=json.loads(cache.read_text())
            if time.time()-c['saved_at']<86400:
                return normalize_universe(pd.DataFrame(c['rows'])),c['meta']
        except (ValueError,KeyError,TypeError):
            pass
    errors=[]; pieces=[]
    endpoints=[('TWSE','https://openapi.twse.com.tw/v1/opendata/t187ap03_L'),
               ('TPEx','https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O')]
    for market,url in endpoints:
        try:
            response=requests.get(url,timeout=15,headers={'User-Agent':'AlphaRadarResearch/8.1'})
            response.raise_for_status()
            frame=_normalize_roster(pd.DataFrame(response.json()),market)
            if frame.empty: raise PriceDataError('Unrecognized roster format')
            pieces.append(frame)
        except Exception as exc:
            errors.append(f'{market}: {type(exc).__name__}: {str(exc)[:120]}')
    if not any((p['market']=='TPEx').any() for p in pieces):
        try:
            from io import StringIO
            response=requests.get('https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv',timeout=15)
            response.raise_for_status()
            p=_normalize_roster(pd.read_csv(StringIO(response.content.decode('utf-8-sig'))),'TPEx')
            if not p.empty: pieces.append(p)
        except Exception as exc: errors.append(f'TPEx CSV: {type(exc).__name__}')
    if pieces:
        u=normalize_universe(pd.concat(pieces,ignore_index=True))
        meta={'source':'official_current_rosters','full_market_verified':set(u['market'])=={'TWSE','TPEx'},
              'total':len(u),'warnings':errors,'fetched_at':_taipei_timestamp().isoformat()}
        cache.write_text(json.dumps({'saved_at':time.time(),'rows':u.to_dict('records'),'meta':meta},ensure_ascii=False))
        return u,meta
    if cache.exists():
        c=json.loads(cache.read_text()); meta=dict(c['meta'])
        meta['source']='stale_roster_cache'; meta['full_market_verified']=False; meta['warnings']=errors
        return normalize_universe(pd.DataFrame(c['rows'])),meta
    raise PriceDataError('No usable official/custom roster. '+ '; '.join(errors))


def resolve_ticker(code: str, universe: Optional[pd.DataFrame]=None) -> str:
    value=str(code).strip().upper()
    if re.fullmatch(r'\d{4}\.(TW|TWO)',value): return value
    if not re.fullmatch(r'\d{4}',value):
        raise PriceDataError('Use a four-digit code, optionally suffixed .TW or .TWO')
    if universe is not None and not universe.empty:
        hits=universe[universe['ticker'].str.split('.').str[0].eq(value)]
        if len(hits)==1: return str(hits.iloc[0]['ticker'])
    raise PriceDataError('Market is unknown. Supply the exact .TW or .TWO suffix')


def _frame_payload(df: pd.DataFrame) -> dict:
    d=_normalize_ohlcv_frame(df)
    return {'dates':d.index.strftime('%Y-%m-%d').tolist(),'values':d[OHLCV].to_numpy(dtype=float).tolist()}


def _payload_frame(data: dict) -> pd.DataFrame:
    if len(data.get('dates',[]))>15000: raise PriceDataError('Cached history exceeds row budget')
    d=pd.DataFrame(data['values'],columns=OHLCV,index=pd.to_datetime(data['dates']))
    return _normalize_ohlcv_frame(d)


def serialize_pair(pair: dict) -> bytes:
    d={k:pair[k] for k in ('source','execution_verified','signal_adjustment_verified') if k in pair}
    d['signal']=_frame_payload(pair['signal']); d['exec']=_frame_payload(pair['exec'])
    return gzip.compress(json.dumps(d,separators=(',',':'),allow_nan=False).encode(),compresslevel=3)


def deserialize_pair(blob: bytes) -> dict:
    import io
    if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob)>5_000_000:
        raise PriceDataError('Cache blob exceeds limit or has an invalid type')
    with gzip.GzipFile(fileobj=io.BytesIO(bytes(blob))) as stream:
        text=stream.read(15_000_001)
    if len(text)>15_000_000: raise PriceDataError('Inflated cache exceeds limit')
    obj=json.loads(text)
    if (not isinstance(obj,dict) or obj.get('execution_verified') is not True
            or obj.get('signal_adjustment_verified') is not True):
        raise PriceDataError('Cached tracks must have explicit verified flags')
    frames = {}
    for key in ('signal','exec'):
        payload=obj.get(key)
        if (not isinstance(payload,dict) or not isinstance(payload.get('dates'),list)
                or not isinstance(payload.get('values'),list)
                or len(payload['dates'])!=len(payload['values'])):
            raise PriceDataError('Invalid cached price table')
        frames[key]=_payload_frame(payload)
        if frames[key].empty:
            raise PriceDataError('Empty cached price track')
    if not frames['signal'].index.equals(frames['exec'].index):
        raise PriceDataError('Cached dual-track dates are not identical')
    return {**obj,**frames}


def sqlite_wal_fixed(version_info=None):
    """Conservative gate for SQLite's documented 2026 WAL-reset fix.

    Fixed upstream: >=3.51.3; maintained backports: 3.50.7 and 3.44.6.
    Unknown distro backports cannot be inferred from a version string, so use
    rollback journaling with bounded waits instead of claiming WAL is safe.
    """
    v = sqlite3.sqlite_version_info if version_info is None else version_info
    if (not isinstance(v, (tuple, list)) or len(v) != 3
            or any(type(n) is not int or n < 0 for n in v)):
        return False
    v = tuple(v)
    return v >= (3,51,3) or ((3,50,7) <= v < (3,51,0)) or ((3,44,6) <= v < (3,45,0))


class DailyPriceStore:
    """Optional bounded LOCAL cache, never the authoritative evidence database.

    Validated version-gated WAL (else DELETE), one connection per operation,
    bounded waits/retries, and a per-fetch circuit breaker avoid N repeated
    waits under a persistent lock. Expected storage failures become visible
    cache misses; malformed schemas/programming errors are not hidden. No file
    is deleted automatically. Live responses still need normal verification.
    """
    def __init__(self, path: Path, max_bytes: int=220_000_000,
                 lock_timeout: float=30.0, busy_retries: int=2):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.lock_timeout = float(lock_timeout)
        self.busy_retries = int(busy_retries)
        if self.max_bytes <= 0 or not math.isfinite(self.lock_timeout) or self.lock_timeout <= 0 or not 0 <= self.busy_retries <= 5:
            raise ValueError("Invalid cache budget/timeout/retry settings")
        self.last_cache_error = None
        self.last_failure_kind = None
        self.failure_sequence = 0
        self.actual_journal_mode = None
        self.journal_mode = 'wal' if sqlite_wal_fixed() else 'delete'
        self.cache_notice = (None if self.journal_mode == 'wal' else
            'SQLite ' + sqlite3.sqlite_version + ': using rollback journal; WAL requires upstream 3.51.3+ or fixed 3.50.7/3.44.6 branch')
        self.enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._run(self._initialize)
        except (sqlite3.DatabaseError, OSError) as exc:
            kind = self._storage_failure_kind(exc)
            if kind is None:
                raise
            self.enabled = False
            self._record_failure(kind, 'initialization', exc)

    @staticmethod
    def _busy(exc):
        code = getattr(exc, 'sqlite_errorcode', 0) or 0
        return (code & 255) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or any(
            text in str(exc).lower() for text in ('database is locked', 'database table is locked', 'database is busy'))

    @classmethod
    def _storage_failure_kind(cls, exc):
        if cls._busy(exc):
            return 'busy'
        code = (getattr(exc,'sqlite_errorcode',0) or 0) & 255
        codes = {sqlite3.SQLITE_CORRUPT:'corrupt', sqlite3.SQLITE_NOTADB:'not_database',
                 sqlite3.SQLITE_CANTOPEN:'cannot_open', sqlite3.SQLITE_FULL:'disk_full',
                 sqlite3.SQLITE_IOERR:'io_error', sqlite3.SQLITE_READONLY:'read_only',
                 sqlite3.SQLITE_PERM:'permission'}
        if code in codes:
            return codes[code]
        text = str(exc).lower()
        for fragment, kind in [('file is not a database','not_database'),
                               ('database disk image is malformed','corrupt'),
                               ('unable to open database file','cannot_open'),
                               ('database or disk is full','disk_full'),
                               ('disk i/o error','io_error'),('readonly database','read_only')]:
            if fragment in text:
                return kind
        if isinstance(exc,OSError):
            return 'filesystem'
        return None

    def _record_failure(self, kind, operation, exc=None):
        self.failure_sequence += 1
        self.last_failure_kind = kind
        detail = (': '+safe_error_text(exc,160)) if exc is not None else ''
        self.last_cache_error = f'Cache {operation} {kind}; verified live data may be used without this cache{detail}'
        logger.warning(self.last_cache_error)
        if kind in {'corrupt','not_database','cannot_open','io_error','filesystem','permission'}:
            self.enabled = False

    def health(self):
        return {'enabled':self.enabled,'sqlite_runtime':sqlite3.sqlite_version,
                'requested_journal_mode':self.journal_mode,'actual_journal_mode':self.actual_journal_mode,
                'busy_timeout_ms':int(self.lock_timeout*1000),'busy_retries':self.busy_retries,
                'max_compressed_payload_bytes':self.max_bytes,
                'cache_error_count':self.failure_sequence,'last_failure_kind':self.last_failure_kind,
                'last_error':safe_error_text(self.last_cache_error,300) if self.last_cache_error else None,
                'notice':self.cache_notice}

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=self.lock_timeout)
        try:
            db.execute(f'PRAGMA busy_timeout={int(self.lock_timeout * 1000)}')
            with db:
                yield db
        finally:
            db.close()

    def _run(self, operation, write=False):
        for attempt in range(self.busy_retries + 1):
            try:
                with self._connection() as db:
                    if write:
                        db.execute('BEGIN IMMEDIATE')
                    return operation(db)
            except sqlite3.OperationalError as exc:
                if not self._busy(exc) or attempt >= self.busy_retries:
                    raise
                time.sleep(.05 * (2 ** attempt))

    def _initialize(self, db):
        mode = str(db.execute('PRAGMA journal_mode').fetchone()[0]).lower()
        desired = self.journal_mode
        if mode != desired:
            mode = str(db.execute('PRAGMA journal_mode=' + desired.upper()).fetchone()[0]).lower()
        if mode != desired:
            raise sqlite3.OperationalError('Unable to select safe local-cache journal mode')
        self.actual_journal_mode = mode
        db.execute('CREATE TABLE IF NOT EXISTS prices (k TEXT PRIMARY KEY, saved REAL, n INTEGER, payload BLOB)')
        db.execute('CREATE INDEX IF NOT EXISTS prices_saved_idx ON prices(saved)')

    def key(self,ticker,period,as_of):
        obj=(PARSER_VERSION,str(ticker),str(period),_price_as_of_key(as_of))
        return hashlib.sha256(repr(obj).encode()).hexdigest()

    def get(self,ticker,period,as_of):
        if not self.enabled:
            return None
        try:
            r = self._run(lambda db: db.execute('SELECT payload FROM prices WHERE k=?',
                                               (self.key(ticker,period,as_of),)).fetchone())
        except (sqlite3.DatabaseError,OSError) as exc:
            kind = self._storage_failure_kind(exc)
            if kind is None:
                raise
            self._record_failure(kind,'read',exc)
            return None
        if r is None:
            return None
        try:
            return deserialize_pair(r[0])
        except (ValueError,TypeError,KeyError,OSError,EOFError,zlib.error,UnicodeError,OverflowError) as exc:
            # A bad record is not evidence that every row is corrupt. A successful
            # provider response can replace THIS entry via the ordinary put path.
            self._record_failure('payload_invalid','decode',exc)
            return None

    def put(self,ticker,period,as_of,pair):
        if not self.enabled or pair.get('execution_verified') is not True or pair.get('signal_adjustment_verified') is not True:
            return False
        blob = serialize_pair(pair)  # Compression is outside the write transaction.
        if len(blob) > self.max_bytes:
            self._record_failure('payload_budget','write')
            return False
        key = self.key(ticker,period,as_of)
        def write(db):
            db.execute('INSERT OR REPLACE INTO prices VALUES (?,?,?,?)', (key,time.time(),len(blob),blob))
            total = db.execute('SELECT COALESCE(SUM(n),0) FROM prices').fetchone()[0]
            while total > self.max_bytes:
                old = db.execute('SELECT k,n FROM prices ORDER BY saved,k LIMIT 1').fetchone()
                if old is None:
                    break
                db.execute('DELETE FROM prices WHERE k=?',(old[0],))
                total -= old[1]
        try:
            self._run(write, write=True)
            return True
        except (sqlite3.DatabaseError,OSError) as exc:
            kind = self._storage_failure_kind(exc)
            if kind is None:
                raise
            self._record_failure(kind,'write',exc)
            return False

    def clear(self):
        if not self.enabled:
            return False
        try:
            self._run(lambda db: db.execute('DELETE FROM prices'), write=True)
            return True
        except (sqlite3.DatabaseError,OSError) as exc:
            kind=self._storage_failure_kind(exc)
            if kind is None:
                raise
            self._record_failure(kind,'clear',exc)
            return False

    def fetch(self,tickers,period,as_of):
        ticks = list(dict.fromkeys(tickers))
        result = {}; missing = []; hits = 0
        read_allowed = self.enabled
        cache_bypassed = not self.enabled
        for ticker in ticks:
            before = self.failure_sequence
            pair = self.get(ticker,period,as_of) if read_allowed else None
            if (self.failure_sequence != before and
                    self.last_failure_kind not in {'payload_invalid','payload_budget'}):
                read_allowed = False
                cache_bypassed = True
            if pair is None:
                missing.append(ticker)
            else:
                result[ticker] = pair; hits += 1
        diag = {'requested':len(ticks),'cache_hits':hits, 'errors':[], 'failed_tickers':[],
                'cache_journal':self.journal_mode, 'cache_notice':self.cache_notice}
        if missing:
            try:
                live, dd = _download_live(tuple(missing),period,as_of)
                diag.update(dd); diag['cache_hits'] = hits
                write_allowed = self.enabled and not cache_bypassed
                for ticker,pair in live.items():
                    if ticker not in missing:
                        continue
                    result[ticker] = pair
                    if not write_allowed:
                        continue
                    before = self.failure_sequence
                    try:
                        self.put(ticker,period,as_of,pair)
                    except (ValueError,OSError) as exc:
                        diag['errors'].append('Cache payload skipped: '+safe_error_text(exc,180))
                    if (self.failure_sequence != before and
                            self.last_failure_kind not in {'payload_invalid','payload_budget'}):
                        write_allowed = False
                        cache_bypassed = True
            except Exception as exc:
                diag['errors'].append(f'{type(exc).__name__}: {safe_error_text(exc,180,(_finmind_token(),))}')
        if self.last_cache_error:
            diag['errors'].append(self.last_cache_error)
        diag['errors']=[safe_error_text(e,320,(_finmind_token(),)) for e in diag.get('errors',[])][:12]
        diag['requested']=len(ticks)
        diag['cache_bypassed_after_failure']=cache_bypassed
        diag['cache_health']=self.health()
        diag['failed_tickers'] = [t for t in ticks if t not in result]
        diag['downloaded'] = len(result)
        return result,diag
