"""Alpha Radar V8.0-RC: return-first RESEARCH core (not an order router).

This module deliberately does NOT convert ranks or ATR multiples into forecasts.
It consumes a time-stamped, cost-accounted outcome ledger supplied by an external
PIT backtest/recording system. With no ledger, there is no return forecast.

All returns are DECIMALS: 0.08 means +8%. Each observation represents one decision
at signal_at and one fixed policy/cost model, including non-fills. Strategy cash
flows are valued at evaluation_at, even if the position exited earlier. Cash
interest is whatever the supplied ledger's policy specifies; never fabricated.
The asset_net_return is a separately simulated, next-tradable-entry buy-and-hold
baseline valued at the SAME evaluation date, not MFE. Net labels already include
fees/slippage/taxes; this estimator does not subtract them twice.

The conditional estimator and its uncertainty diagnostic are NOT OOS calibrated.
No method in this file establishes historical universe completeness, actual fills,
or live investment suitability. See MODEL_SPEC.md and README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping, Sequence
from typing import Any, Optional
import hashlib
import json
import math

import numpy as np
import pandas as pd

VERSION = "8.0-RC-return-first"
OUTCOME_COLS = ("strategy_net_return", "asset_net_return", "benchmark_net_return")
FORBIDDEN_FEATURES = set(OUTCOME_COLS) | {
    "filled", "mfe", "mae", "future_return", "label_available_at", "evaluation_at",
    "exit_price", "exit_at", "entry_price", "outcome", "alpha", "profit",
}
META_COLS = ("event_id", "ticker", "policy_id", "cost_model_id", "horizon_days",
             "signal_at", "features_available_at", "evaluation_at", "label_available_at",
             "setup", "regime", "filled")


class ModelDataError(ValueError):
    pass


def finite_scalar(value: Any) -> float:
    if value is None or isinstance(value, (bool, np.bool_, list, tuple, dict, set,
                                           pd.Series, pd.DataFrame, np.ndarray)):
        return float("nan")
    try:
        v = float(value)
        return v if math.isfinite(v) else float("nan")
    except (ValueError, TypeError, OverflowError):
        return float("nan")


def timestamp(value: Any) -> pd.Timestamp:
    """Missing/ambiguous time is an error, NEVER an instruction to use now."""
    if value is None or isinstance(value, (bool, int, float, np.number,
                                           list, tuple, dict, pd.Series, np.ndarray)):
        raise ModelDataError("An explicit, non-numeric timestamp is required")
    try:
        t = pd.Timestamp(value)
        if pd.isna(t):
            raise ValueError("NaT")
        return t.tz_localize("Asia/Taipei") if t.tzinfo is None else t.tz_convert("Asia/Taipei")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ModelDataError(f"Invalid timestamp: {value!r}") from exc


def _numeric(s: pd.Series) -> np.ndarray:
    return s.map(finite_scalar).to_numpy(dtype="float64")


def normalized_weights(values: Sequence[float], weights: Optional[Sequence[float]] = None):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ModelDataError("Outcomes must be non-empty finite scalar observations")
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w < 0) or w.max() <= 0:
        raise ModelDataError("Invalid distribution weights")
    w = w / w.max()
    w /= math.fsum(w.tolist())
    return y, w


def empirical_quantile(y: Sequence[float], q: float, weights=None) -> float:
    """Left inverse of the weighted empirical CDF; no interpolated tail prices."""
    values, w = normalized_weights(y, weights)
    if not 0 <= q <= 1:
        raise ModelDataError("Quantile must be in [0,1]")
    order = np.argsort(values, kind="stable")
    values, w = values[order], w[order]
    pos = int(np.searchsorted(np.cumsum(w), q, side="left"))
    return float(values[min(pos, len(values) - 1)])


def lower_tail_mean(y: Sequence[float], tail_mass: float = 0.10, weights=None) -> float:
    """Exact lower-tail integral, fractionally weighting the boundary atom."""
    values, w = normalized_weights(y, weights)
    if not 0 < tail_mass <= 1:
        raise ModelDataError("tail_mass must be in (0,1]")
    order = np.argsort(values, kind="stable")
    values, w = values[order], w[order]
    before = np.cumsum(w) - w
    take = np.minimum(w, np.maximum(0.0, tail_mass - before))
    return float(np.dot(take, values) / tail_mass)


def payoff_summary(y: Sequence[float], weights=None, large_gain: float = 0.10) -> dict:
    """All statistics come from the SAME net-return distribution, including zeros."""
    values, w = normalized_weights(y, weights)
    gain = values > 0
    loss = values < 0
    pg, pl = float(w[gain].sum()), float(w[loss].sum())
    gw = float(np.dot(w[gain], values[gain]))
    gl = float(-np.dot(w[loss], values[loss]))
    avg_win = gw / pg if pg > 0 else None
    avg_loss = gl / pl if pl > 0 else None
    # Ratios with no losses are undefined evidence, never infinite skill.
    payoff_ratio = avg_win / avg_loss if avg_win is not None and avg_loss and avg_loss > 0 else None
    return {
        "mean": float(np.dot(w, values)),
        "median": empirical_quantile(values, .5, w),
        "p10": empirical_quantile(values, .10, w),
        "p75": empirical_quantile(values, .75, w),
        "p90": empirical_quantile(values, .90, w),
        "expected_shortfall10_loss": max(0.0, -lower_tail_mean(values, .10, w)),
        "p_profit": pg, "p_loss": pl, "p_flat": max(0.0, 1.0-pg-pl),
        "p_large_gain": float(w[values >= large_gain].sum()),
        "average_win": avg_win, "average_loss": avg_loss,
        "expected_positive_payoff": gw, "expected_negative_payoff": gl,
        "payoff_ratio": payoff_ratio,
    }


@dataclass(frozen=True)
class TradeCosts:
    """Editable research assumptions, not a quote from any specific broker.

    Supply actual fill prices: slippage must already be in fills, not deducted twice.
    All entry/add/reduce/exit legs belong in the full ledger. This helper covers a
    one-entry, one-exit position with known post-action shares and cash distributions.
    """
    commission_rate: float = .001425
    sell_tax_rate: float = .003
    minimum_commission: float = 20.0

    def __post_init__(self):
        for name in ("commission_rate", "sell_tax_rate", "minimum_commission"):
            v = finite_scalar(getattr(self, name))
            if not np.isfinite(v) or v < 0:
                raise ModelDataError("Costs must be finite and nonnegative")
        if self.commission_rate >= 1 or self.sell_tax_rate >= 1:
            raise ModelDataError("Fees are decimal rates, not percentages")

    def net_return(self, entry_fill: float, exit_fill: float, entry_shares: float,
                   exit_shares: Optional[float] = None, cash_distributions: float = 0.0) -> float:
        es = entry_shares if exit_shares is None else exit_shares
        vals = [finite_scalar(v) for v in (entry_fill, exit_fill, entry_shares, es, cash_distributions)]
        a, b, q, qe, cash = vals
        if not np.isfinite(vals).all() or a <= 0 or q <= 0 or b < 0 or qe < 0 or cash < 0:
            raise ModelDataError("Invalid known fills/shares/distributions")
        buy, sell = a*q, b*qe
        buy_cost = max(self.minimum_commission, buy*self.commission_rate)
        sell_cost = max(self.minimum_commission, sell*self.commission_rate) if sell > 0 else 0.0
        proceeds = sell-sell_cost-sell*self.sell_tax_rate+cash
        return proceeds/(buy+buy_cost)-1.0


@dataclass(frozen=True)
class ForecastConfig:
    """Research priors; these are NOT parameters optimized on the user's outcomes."""
    horizon_days: int
    policy_id: str
    cost_model_id: str
    features: tuple[str, ...]
    neighbors: int = 160
    min_reference: int = 40
    min_local_effective_n: float = 25.0
    min_time_blocks: int = 8
    min_tickers: int = 5
    min_local_weight: float = .40
    prior_strength: float = 30.0
    kernel_width: float = 1.0
    max_scaled_distance: float = 2.5
    uncertainty_multiplier: float = 1.0
    large_gain_return: float = .10
    model_family: str = "price_only"

    def __post_init__(self):
        if not isinstance(self.horizon_days, int) or isinstance(self.horizon_days, bool) or self.horizon_days < 1:
            raise ModelDataError("Invalid horizon")
        if not self.policy_id or not self.cost_model_id:
            raise ModelDataError("Policy and cost version IDs are mandatory")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ModelDataError("A fixed, nonempty feature set is mandatory")
        if any(f in FORBIDDEN_FEATURES or f.startswith(("future_", "label_", "outcome_")) for f in self.features):
            raise ModelDataError("Outcome leakage column is forbidden as a feature")
        if any(not isinstance(getattr(self, k), int) or getattr(self, k) < 1
               for k in ("neighbors", "min_reference", "min_time_blocks", "min_tickers")):
            raise ModelDataError("Invalid sample-count configuration")
        for k in ("min_local_effective_n", "prior_strength", "kernel_width", "max_scaled_distance"):
            if not np.isfinite(finite_scalar(getattr(self, k))) or getattr(self, k) <= 0:
                raise ModelDataError(f"Invalid {k}")
        if (not 0 <= self.min_local_weight <= 1 or not np.isfinite(finite_scalar(self.uncertainty_multiplier))
                or self.uncertainty_multiplier < 0 or not np.isfinite(finite_scalar(self.large_gain_return))
                or self.large_gain_return <= 0):
            raise ModelDataError("Invalid research thresholds")
        if self.model_family not in {"price_only", "business_confirmed", "flow_confirmed", "full"}:
            raise ModelDataError("Unknown information/model family")


@dataclass(frozen=True)
class Forecast:
    ticker: str
    snapshot_id: str
    as_of: str
    horizon_days: int
    policy_id: str
    cost_model_id: str
    model_family: str
    estimate_available: bool
    sample_supported: bool
    reasons: tuple[str, ...]
    reference_n: int = 0
    local_n: int = 0
    local_effective_n: float = 0.0
    local_time_blocks: int = 0
    local_tickers: int = 0
    local_weight: float = 0.0
    fill_probability: Optional[float] = None
    conditional_filled_mean: Optional[float] = None
    strategy: Optional[dict] = None
    asset_potential: Optional[dict] = None
    alpha_mean: Optional[float] = None
    mean_uncertainty_diagnostic: Optional[float] = None
    uncertainty_adjusted_ev: Optional[float] = None
    local_mean_without_best_block: Optional[float] = None
    validation_status: str = "RESEARCH_NOT_OOS_VALIDATED"

    def to_dict(self) -> dict:
        return asdict(self)


def _block_ids(signals: pd.Series, days: int) -> np.ndarray:
    # Calendar clusters are conservative diagnostics, not a substitute for full
    # overlapping-label purging and block-bootstrap coverage calibration.
    utc_ns = signals.dt.tz_convert("UTC").astype("int64").to_numpy()
    width = max(30, math.ceil(days*7/5))*86_400_000_000_000
    return (utc_ns//width).astype(str)


def _cluster_se(y: np.ndarray, w: np.ndarray, clusters: Sequence[Any]) -> float:
    groups = pd.DataFrame({"g": np.asarray(clusters), "z": w*(y-np.dot(w,y))}).groupby("g")["z"].sum()
    n = len(groups)
    if n < 2:
        return float("inf")
    return float(math.sqrt(n/(n-1)*np.square(groups.to_numpy()).sum()))


class ReturnFirstSnapshot:
    """Point-in-time fitted research estimator; predict() never adds the query.

    Ranking is across queries with the SAME snapshot, horizon, policy, feature
    family and cost model. No candidate-rank-to-1800-stock mapping exists here.
    """
    def __init__(self, ledger: pd.DataFrame, as_of: Any, config: ForecastConfig):
        self.config = config
        self.as_of = timestamp(as_of)
        self._ref, self.audit = self._validate_and_freeze(ledger)
        self._feature_values = self._ref[list(config.features)].to_numpy(dtype=float)
        if len(self._ref):
            self._median = np.median(self._feature_values, axis=0)
            q75, q25 = np.quantile(self._feature_values, [.75,.25], axis=0)
            scale = (q75-q25)/1.349
            # A constant training feature must not turn arbitrary extrapolation
            # into zero distance; use a finite absolute/relative scale floor.
            floor = np.maximum(np.abs(self._median)*.01, .01)
            self._scale = np.maximum(scale, floor)
        else:
            self._median = np.zeros(len(config.features)); self._scale = np.ones(len(config.features))
        self._feature_values.setflags(write=False)
        self._median.setflags(write=False); self._scale.setflags(write=False)
        payload = json.dumps(asdict(config), sort_keys=True) + self.as_of.isoformat()
        payload += self._ref.to_json(date_format="iso", orient="split", double_precision=15)
        self.snapshot_id = hashlib.sha256(payload.encode()).hexdigest()[:20]

    def _validate_and_freeze(self, ledger: pd.DataFrame):
        needed = set(META_COLS) | set(OUTCOME_COLS) | set(self.config.features)
        missing = needed-set(ledger.columns)
        if missing:
            raise ModelDataError("Ledger missing columns: " + ",".join(sorted(missing)))
        if not ledger.columns.is_unique:
            raise ModelDataError("Duplicate ledger columns")
        d = ledger.copy(deep=True).reset_index(drop=True)
        for k in ["event_id", "ticker", "policy_id", "cost_model_id", "setup", "regime"]:
            if d[k].isna().any() or d[k].astype(str).str.strip().eq("").any():
                raise ModelDataError(f"Missing metadata: {k}")
            d[k] = d[k].astype(str).str.strip()
        if d["event_id"].duplicated().any():
            raise ModelDataError("Duplicate event_id; repeated data cannot increase evidence")
        for k in ["signal_at", "features_available_at", "evaluation_at", "label_available_at"]:
            d[k] = pd.to_datetime(d[k].map(timestamp), utc=True).dt.tz_convert("Asia/Taipei")
        if (d["features_available_at"] > d["signal_at"]).any():
            raise ModelDataError("Future feature supplied to a past signal")
        if (d["evaluation_at"] <= d["signal_at"]).any():
            raise ModelDataError("Outcome evaluation must be after signal")
        if (d["label_available_at"] < d["evaluation_at"]).any():
            raise ModelDataError("Full-horizon labels cannot be available before maturity")
        if not d["filled"].map(lambda v: isinstance(v,(bool,np.bool_))).all():
            raise ModelDataError("filled must be a real boolean, not a truthy string")
        h = _numeric(d["horizon_days"])
        if not np.isfinite(h).all() or (h<=0).any() or np.any(h != np.floor(h)):
            raise ModelDataError("Invalid horizon_days")
        d["horizon_days"] = h.astype(int)
        # Do not inspect future/unrelated outcome values to determine inclusion in
        # a historical snapshot. Only metadata known for maturity defines slicing.
        relevant = (d["horizon_days"]==self.config.horizon_days) & (d["policy_id"]==self.config.policy_id) & (d["cost_model_id"]==self.config.cost_model_id)
        mature = (d["label_available_at"] <= self.as_of) & (d["evaluation_at"] <= self.as_of) & (d["signal_at"] < self.as_of)
        audit = {"provided": len(d), "policy_horizon_rows": int(relevant.sum()),
                 "not_available_yet": int((relevant & ~mature).sum())}
        d = d.loc[relevant & mature].copy()
        for col in (*self.config.features, *OUTCOME_COLS):
            d[col] = _numeric(d[col])
        if len(d) and not np.isfinite(d[list(OUTCOME_COLS)].to_numpy()).all():
            raise ModelDataError("A mature label is missing/invalid; do not silently omit unresolved losses")
        if len(d) and not np.isfinite(d[list(self.config.features)].to_numpy()).all():
            raise ModelDataError("Missing feature in a fixed model family; use a declared reduced-information model")
        if len(d) and np.any((~d["filled"]) & ~np.isclose(d["strategy_net_return"],0)):
            raise ModelDataError("This policy convention requires non-fill strategy cash return = 0")
        # Avoid duplicate overlapping events for a ticker/policy. Keep earliest,
        # using only signal/evaluation timestamps, never their returns.
        d = d.sort_values(["signal_at", "ticker", "event_id"], kind="stable")
        retained=[]; last_end={}
        for idx,row in d.iterrows():
            end=last_end.get(row["ticker"])
            if end is None or row["signal_at"] > end:
                retained.append(idx); last_end[row["ticker"]]=row["evaluation_at"]
        audit["overlap_removed"]=len(d)-len(retained)
        d=d.loc[retained].reset_index(drop=True)
        audit["retained"]=len(d)
        return d, audit

    @property
    def reference_n(self):
        return len(self._ref)

    def export_manifest(self) -> dict:
        return {"version":VERSION, "snapshot_id":self.snapshot_id, "as_of":self.as_of.isoformat(),
                "config":asdict(self.config), "audit":dict(self.audit),
                "status":"RESEARCH_NOT_OOS_VALIDATED", "feature_scaling":"training-IQR only",
                "universe_completeness":"not certified by this module"}

    def _empty(self,ticker: str,why: str) -> Forecast:
        return Forecast(ticker=ticker,snapshot_id=self.snapshot_id,as_of=self.as_of.isoformat(),
                        horizon_days=self.config.horizon_days,policy_id=self.config.policy_id,
                        cost_model_id=self.config.cost_model_id,model_family=self.config.model_family,
                        estimate_available=False,sample_supported=False,reasons=(why,),reference_n=len(self._ref))

    def estimate(self, ticker: str, features: Mapping[str, float], *, setup: str,
                 regime: str, features_available_at: Any) -> Forecast:
        cfg=self.config
        if timestamp(features_available_at)>self.as_of:
            raise ModelDataError("Query features are newer than the frozen snapshot")
        q=np.asarray([finite_scalar(features.get(k)) for k in cfg.features])
        if not np.isfinite(q).all():
            return self._empty(ticker,"MISSING_REQUIRED_FEATURES")
        eligible=(self._ref["setup"].eq(setup) & self._ref["regime"].eq(regime)).to_numpy()
        ix=np.flatnonzero(eligible)
        if len(ix)<cfg.min_reference:
            return self._empty(ticker,"INSUFFICIENT_MATURE_SAME_POLICY_STATE")
        d=self._ref.iloc[ix]
        distance=np.sqrt(np.mean(np.square((self._feature_values[ix]-q)/self._scale),axis=1))
        near=np.flatnonzero(distance<=cfg.max_scaled_distance)
        near=near[np.argsort(distance[near],kind="stable")[:cfg.neighbors]]
        if len(near)<2:
            return self._empty(ticker,"NO_SUPPORTED_ANALOGUE_DO_NOT_EXTRAPOLATE")
        blocks=_block_ids(d["signal_at"],cfg.horizon_days)
        stocks=d["ticker"].to_numpy()
        # Equalize broad prior across stocks, reducing high-frequency issuer dominance.
        counts=pd.Series(stocks).value_counts()
        prior=np.asarray([1/counts[s] for s in stocks],dtype=float); prior/=prior.sum()
        local=np.zeros(len(d))
        local[near]=np.exp(-.5*np.square(distance[near]/cfg.kernel_width))
        # Downweight simultaneous crowding in the local evidence pool. This is
        # sample weighting, NOT a penalty for a financial/electronics industry.
        local_stock=pd.Series(stocks[near]).value_counts()
        local_block=pd.Series(blocks[near]).value_counts()
        local[near] /= np.asarray([math.sqrt(local_stock[s]*local_block[b]) for s,b in zip(stocks[near],blocks[near])])
        if not np.isfinite(local).all() or local.sum()<=0:
            return self._empty(ticker,"DEGENERATE_LOCAL_WEIGHTS")
        local/=local.sum()
        ess=float(1/np.square(local).sum())
        lam=ess/(ess+cfg.prior_strength)
        w=lam*local+(1-lam)*prior
        y=d["strategy_net_return"].to_numpy(float)
        asset=d["asset_net_return"].to_numpy(float)
        benchmark=d["benchmark_net_return"].to_numpy(float)
        summary=payoff_summary(y,w,cfg.large_gain_return)
        asset_summary=payoff_summary(asset,w,cfg.large_gain_return)
        filled=d["filled"].to_numpy(bool)
        pfill=float(w[filled].sum())
        conditional=float(np.dot(w[filled],y[filled])/pfill) if pfill>0 else None
        # An uncertainty diagnostic, NOT a calibrated confidence interval.
        iid_se=math.sqrt(max(0.,float(np.dot(w,np.square(y-summary["mean"]))))*np.square(w).sum())
        se=max(iid_se,_cluster_se(y,w,stocks),_cluster_se(y,w,blocks))
        # Do not let broad-prior sample size erase local extrapolation uncertainty.
        local_y=y[near]; local_w=local[near]/local[near].sum()
        if len(np.unique(blocks[near]))>=2:
            se=max(se,lam*_cluster_se(local_y,local_w,blocks[near]))
        else:
            se=float("inf")
        nblocks=len(np.unique(blocks[near])); nstocks=len(np.unique(stocks[near]))
        local_ev=float(np.dot(local,y))
        best_block=None; best_contribution=-math.inf
        for block in np.unique(blocks[near]):
            mask=(blocks==block)
            contribution=float(np.dot(local[mask],y[mask]))
            if contribution>best_contribution:
                best_block,best_contribution=block,contribution
        rest=(blocks!=best_block)&(local>0)
        without_best=float(np.dot(local[rest],y[rest])/local[rest].sum()) if local[rest].sum()>0 else None
        reasons=[]
        if ess<cfg.min_local_effective_n: reasons.append("LOW_EFFECTIVE_SAMPLE")
        if nblocks<cfg.min_time_blocks: reasons.append("FEW_INDEPENDENT_TIME_BLOCKS")
        if nstocks<cfg.min_tickers: reasons.append("ISSUER_CONCENTRATION")
        if lam<cfg.min_local_weight: reasons.append("PRIOR_DOMINATED")
        if without_best is not None and local_ev>0 and without_best<=0: reasons.append("EDGE_DEPENDS_ON_BEST_TIME_BLOCK")
        if not np.isfinite(se): reasons.append("UNCERTAINTY_NOT_IDENTIFIABLE")
        return Forecast(
            ticker=ticker,snapshot_id=self.snapshot_id,as_of=self.as_of.isoformat(),
            horizon_days=cfg.horizon_days,policy_id=cfg.policy_id,cost_model_id=cfg.cost_model_id,
            model_family=cfg.model_family,estimate_available=True,sample_supported=not reasons,
            reasons=tuple(reasons),reference_n=len(ix),local_n=len(near),local_effective_n=ess,
            local_time_blocks=nblocks,local_tickers=nstocks,local_weight=lam,
            fill_probability=pfill,conditional_filled_mean=conditional,strategy=summary,
            asset_potential=asset_summary,alpha_mean=float(np.dot(w,y-benchmark)),
            mean_uncertainty_diagnostic=se if np.isfinite(se) else None,
            uncertainty_adjusted_ev=(summary["mean"]-cfg.uncertainty_multiplier*se) if np.isfinite(se) else None,
            local_mean_without_best_block=without_best,
        )


@dataclass(frozen=True)
class EntryPlan:
    policy_id: str
    horizon_days: int
    setup: str
    entry_mode: str                 # zone_confirmed OR breakout_confirmed; no post-hoc switching
    signal_at: str
    expires_at: str
    zone_low: float
    zone_high: float
    trigger: float
    chase_limit: float
    invalidation: float

    def __post_init__(self):
        if not self.policy_id or not isinstance(self.horizon_days,int) or isinstance(self.horizon_days,bool) or self.horizon_days<1:
            raise ModelDataError("Plan policy ID and positive horizon are mandatory")
        for name in ("zone_low","zone_high","trigger","chase_limit","invalidation"):
            if not math.isfinite(finite_scalar(getattr(self,name))) or getattr(self,name)<=0:
                raise ModelDataError("Nonpositive/missing plan price")
        if not self.invalidation<self.zone_low<=self.zone_high<=self.chase_limit:
            raise ModelDataError("Invalid entry zone / structural stop ordering")
        if not self.invalidation<self.trigger<=self.chase_limit:
            raise ModelDataError("Trigger outside allowed structural range")
        if self.entry_mode not in {"zone_confirmed","breakout_confirmed"}:
            raise ModelDataError("Declare one entry policy, not OR two alternatives")
        if timestamp(self.expires_at)<=timestamp(self.signal_at):
            raise ModelDataError("Invalid plan validity window")


def entry_state(plan: EntryPlan, current_price: float, quote_at: Any, *,
                price_verified: bool, confirmation_met: bool) -> str:
    """No model score can override a missing price trigger or stale quote."""
    p=finite_scalar(current_price); t=timestamp(quote_at)
    if price_verified is not True or not math.isfinite(p) or p<=0: return "DATA_UNVERIFIED"
    if t<timestamp(plan.signal_at): return "QUOTE_BEFORE_PLAN"
    if t>timestamp(plan.expires_at): return "PLAN_EXPIRED_REBUILD_NOT_REANCHOR"
    if p<=plan.invalidation: return "INVALIDATED"
    if p>plan.chase_limit: return "DO_NOT_CHASE"
    if confirmation_met is not True: return "WAIT_CONFIRMATION"
    if plan.entry_mode=="zone_confirmed":
        return "CONDITIONS_MET_NOT_FILLED" if plan.zone_low<=p<=plan.zone_high else "WAIT_ENTRY_ZONE"
    return "CONDITIONS_MET_NOT_FILLED" if p>=plan.trigger else "WAIT_BREAKOUT"


@dataclass(frozen=True)
class ResearchMandate:
    """User/research constraints, not data-fitted alpha coefficients.

    Defaults intentionally do not produce official/live recommendations.
    Oversized risk cannot be redeemed by high rank or optimistic potential.
    """
    min_net_ev: float = 0.0
    min_uncertainty_adjusted_ev: float = 0.0
    min_alpha: float = 0.0
    max_expected_shortfall10: float = .20
    max_stop_distance: float = .12
    require_business_for_long: bool = True

    def __post_init__(self):
        for k in ("min_net_ev", "min_uncertainty_adjusted_ev", "min_alpha",
                  "max_expected_shortfall10", "max_stop_distance"):
            if not np.isfinite(finite_scalar(getattr(self,k))) or getattr(self,k)<0:
                raise ModelDataError("Research limits must be finite and nonnegative")
        if not 0 < self.max_stop_distance < 1 or self.max_expected_shortfall10<=0:
            raise ModelDataError("Invalid risk budget")


def rank_research_candidates(forecasts: Sequence[Forecast], plans: Mapping[str,EntryPlan],
                             prices: Mapping[str,float], mandate: ResearchMandate) -> pd.DataFrame:
    """Rank SAME-horizon alternatives by return in %, never a fake 100-point rank.

    This creates a research candidate table only. Confirmation/plan state remains
    a separate check. Sector quotas and low-volatility rewards are absent.
    """
    if not forecasts: return pd.DataFrame()
    keys={(f.snapshot_id,f.horizon_days,f.policy_id,f.cost_model_id,f.model_family) for f in forecasts}
    if len(keys)!=1: raise ModelDataError("Do not compare different horizons/policies/snapshots in one score")
    if len({f.ticker for f in forecasts}) != len(forecasts): raise ModelDataError("Duplicate candidate ticker")
    rows=[]
    for f in forecasts:
        reasons=list(f.reasons)
        if not f.estimate_available: reasons.append("NO_RETURN_ESTIMATE")
        if not f.sample_supported: reasons.append("EVIDENCE_INSUFFICIENT")
        p=plans.get(f.ticker); current=finite_scalar(prices.get(f.ticker))
        risk=None
        if p is None or p.policy_id!=f.policy_id or p.horizon_days!=f.horizon_days:
            reasons.append("MISSING_OR_MISMATCHED_PLAN")
        else:
            # Budget using maximum acceptable entry, never shrink a stop to boost R/R.
            risk=(p.chase_limit-p.invalidation)/p.chase_limit
            if risk>mandate.max_stop_distance: reasons.append("STRUCTURAL_RISK_OVER_BUDGET")
            if not np.isfinite(current) or current<=p.invalidation: reasons.append("CURRENT_STRUCTURE_INVALID")
        ev=f.strategy["mean"] if f.strategy else None
        es=f.strategy["expected_shortfall10_loss"] if f.strategy else None
        if ev is None or ev<=mandate.min_net_ev: reasons.append("NET_EV_BELOW_REQUIREMENT")
        if f.uncertainty_adjusted_ev is None or f.uncertainty_adjusted_ev<=mandate.min_uncertainty_adjusted_ev: reasons.append("UNCERTAINTY_NOT_COMPENSATED")
        if f.alpha_mean is None or f.alpha_mean<=mandate.min_alpha: reasons.append("ALPHA_BELOW_REQUIREMENT")
        if es is None or es>mandate.max_expected_shortfall10: reasons.append("TAIL_RISK_OVER_BUDGET")
        if mandate.require_business_for_long and f.horizon_days>=60 and f.model_family not in {"business_confirmed","full"}:
            reasons.append("LONG_HORIZON_BUSINESS_EVIDENCE_MISSING")
        rows.append({"ticker":f.ticker,"research_qualified":not reasons,
                     "net_ev":ev,"uncertainty_adjusted_ev":f.uncertainty_adjusted_ev,
                     "alpha":f.alpha_mean,"es10_loss":es,"stop_budget_distance":risk,
                     "p75":f.strategy["p75"] if f.strategy else None,
                     "p_large_gain":f.strategy["p_large_gain"] if f.strategy else None,
                     "local_effective_n":f.local_effective_n,
                     "reasons":";".join(dict.fromkeys(reasons)),
                     "status":"RESEARCH_ONLY_NOT_A_LIVE_BUY_SIGNAL"})
    out=pd.DataFrame(rows).sort_values(["research_qualified","uncertainty_adjusted_ev","net_ev","p75","ticker"],ascending=[False,False,False,False,True],na_position="last")
    # No backfilled slots: an empty qualified list is a valid result.
    out["research_rank"]=pd.Series(pd.NA,index=out.index,dtype="Int64")
    ok=out["research_qualified"]
    out.loc[ok,"research_rank"]=range(1,int(ok.sum())+1)
    return out.reset_index(drop=True)


def holding_review(current_price: float, *, original_invalidation: Optional[float],
                   trailing_protection: Optional[float], thesis_broken: Optional[bool],
                   prices_verified: bool) -> str:
    """Cost basis is intentionally NOT an argument. New-buy rank is not an exit rule."""
    p=finite_scalar(current_price); stop=finite_scalar(original_invalidation)
    trail=finite_scalar(trailing_protection)
    if prices_verified is not True or not np.isfinite(p) or p<=0: return "DATA_UNVERIFIED"
    if np.isfinite(stop) and p<=stop: return "ORIGINAL_STRUCTURE_INVALIDATED"
    if thesis_broken is True: return "ORIGINAL_THESIS_INVALIDATED"
    if np.isfinite(trail) and p<=trail: return "PROTECTION_TRIGGER_REVIEW_EXECUTION"
    if thesis_broken is None or not np.isfinite(stop): return "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW"
    return "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE"
