"""Coordinate protection for existing holdings and labeled technical references.

The adjustment ratio is NOT a corporate-action identity or an actual cash ledger.
It maps a total-return signal price to today's raw units. We never subtract cash
from both signal and raw prices, and never silently rewrite an original stop.
"""
from __future__ import annotations
from collections.abc import Mapping
import math
import numpy as np
import pandas as pd
from market_data import _align_price_pair, _daily_cutoff, _taipei_timestamp
from return_first_model import ModelDataError, finite_scalar


def coordinate_review(pair: dict, basis_date, as_of, levels: Mapping[str,float] | None=None):
    if not pair or pair.get('execution_verified') is not True or pair.get('signal_adjustment_verified') is not True:
        return {'status':'UNVERIFIED_ACTION_COORDINATES','can_use_original':False}
    s,e=_align_price_pair(pair.get('signal'),pair.get('exec'),as_of)
    if s.empty or e.empty: return {'status':'MISSING_ACTION_COORDINATES','can_use_original':False}
    if basis_date is None or not str(basis_date).strip():
        return {'status':'ORIGINAL_BASIS_DATE_UNKNOWN','can_use_original':False}
    day=_taipei_timestamp(basis_date).normalize()
    end=s.index[-1]
    if day>end or day<s.index[0]:
        return {'status':'BASIS_OUTSIDE_KNOWN_HISTORY','can_use_original':False}
    # A nontrading-day reference uses the last actual session, explicitly reported.
    ix=s.index[s.index<=day]
    basis=ix[-1]
    factor=e['Close']/s['Close']
    block=factor.loc[basis:end]
    if not np.isfinite(block).all() or (block<=0).any():
        return {'status':'UNVERIFIED_ACTION_COORDINATES','can_use_original':False}
    ratios=block/block.shift(1)
    changed=ratios.loc[(ratios-1).abs()>2e-5].dropna()
    scale=float(block.iloc[-1]/block.iloc[0])
    mapped={}
    for name,value in (levels or {}).items():
        v=finite_scalar(value)
        if np.isfinite(v) and v>0: mapped[name]=v*scale
    return {'status':'ACTION_COORDINATE_CHANGE' if len(changed) else 'SAME_PRICE_COORDINATE',
            'can_use_original':not len(changed),'basis_session':str(basis.date()),
            'target_session':str(end.date()),'reference_scale':scale,
            'changed_dates':[str(t.date()) for t in changed.index],
            'mapped_references':mapped,
            'interpretation':'TOTAL_RETURN_RATIO_REFERENCE_NOT_AUTOMATIC_STOP_REPLACEMENT'}


def technical_reference(pair: dict, as_of, basis_date=None, multiple: float=2.5):
    """ATR reference only. Optional historical basis uses the PREVIOUS close.

    Missing basis means current completed-day close. No entry date/cost is guessed.
    Old prices and ATR are mapped using the same current adjustment vintage.
    """
    m=finite_scalar(multiple)
    if not np.isfinite(m) or not .5<=m<=10: raise ModelDataError('Invalid reference ATR multiplier')
    if not pair or pair.get('execution_verified') is not True or pair.get('signal_adjustment_verified') is not True:
        return {'status':'UNVERIFIED_PRICE','reference':None}
    s,e=_align_price_pair(pair.get('signal'),pair.get('exec'),as_of)
    if len(s)<20: return {'status':'INSUFFICIENT_REFERENCE_HISTORY','reference':None}
    current_factor=float(e['Close'].iloc[-1]/s['Close'].iloc[-1])
    hist=s
    purpose='CURRENT_COMPLETED_CLOSE_REFERENCE'
    if basis_date is not None and str(basis_date).strip():
        day=_taipei_timestamp(basis_date).normalize()
        if day>s.index[-1]+pd.Timedelta(days=1):
            return {'status':'FUTURE_REFERENCE_DATE','reference':None}
        hist=s.loc[s.index<day]
        purpose='PRE_BASIS_SESSION_ATR_REFERENCE'
    if len(hist)<20: return {'status':'INSUFFICIENT_REFERENCE_HISTORY','reference':None}
    pc=hist['Close'].shift()
    tr=pd.concat([hist['High']-hist['Low'],(hist['High']-pc).abs(),(hist['Low']-pc).abs()],axis=1).max(axis=1)
    af=(tr/pc).ewm(alpha=1/14,adjust=False,min_periods=14).mean().iloc[-1]
    raw_basis=float(hist['Close'].iloc[-1]*current_factor)
    ref=raw_basis*(1-m*float(af))
    if not np.isfinite(ref) or ref<=0: return {'status':'INVALID_REFERENCE','reference':None}
    return {'status':'REFERENCE_ONLY','purpose':purpose,'basis_session':str(hist.index[-1].date()),
            'target_session':str(s.index[-1].date()),'atr_fraction':float(af),'multiple':m,
            'anchor_in_current_units':raw_basis,'reference':ref,
            'is_original_plan':False,'automatic_holding_action_allowed':False}
