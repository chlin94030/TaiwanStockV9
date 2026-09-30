"""Thin contract for migrating V7.6.4 RAW features to V8 research queries.

This does not infer absent financial/flow data from price, does not call the old
Edge/Confidence formulas, and does not build a historical ledger from today's
winners. The same conversion MUST run when recording historical signal features.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
from return_first_model import finite_scalar

PRICE_FEATURES = (
    'return20', 'return60', 'return120', 'atr_fraction', 'rvol',
    'extension_atr', 'log_avg_trading_value',
)
BUSINESS_FEATURES = ('revenue_growth', 'eps_growth', 'operating_margin', 'roic')
FLOW_FEATURES = ('foreign_5d_intensity', 'trust_5d_intensity')


def query_features_from_v764(row: Mapping) -> dict:
    out={}
    # Legacy percentage-point columns -> decimal fractions.
    for old,new in [('ret20','return20'),('ret60','return60'),('ret120','return120'),
                    ('atr_pct','atr_fraction'),('revenue_yoy','revenue_growth'),
                    ('eps_yoy','eps_growth'),('operating_margin','operating_margin'),('roic','roic')]:
        out[new]=finite_scalar(row.get(old))/100.0
    for key in ('rvol','extension_atr',*FLOW_FEATURES):
        out[key]=finite_scalar(row.get(key))
    volume=finite_scalar(row.get('avg_turnover20'))
    out['log_avg_trading_value']=math.log(volume) if math.isfinite(volume) and volume>0 else float('nan')
    return out


def legacy_display_fields_to_remove() -> tuple[str,...]:
    return tuple(f'{h}_{suffix}' for h in ('short','mid','long') for suffix in
                 ('edge','hold_edge','conf','evidence_tier'))+('prime_score','prime_raw',)
