"""V8.2 versioned daily policies + exploratory historical outcome builder.

This is NOT a historical constituent database or a tick-level execution model.
Only complete, mature label windows enter the V8 estimator. Adjusted OHLC ratios
approximate reinvested total returns; stop/entry levels are in signal coordinates
and map to raw currency for display. Hypothetical fills cannot certify execution.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Optional
import hashlib, json, math
import numpy as np
import pandas as pd
from market_data import _align_price_pair, _daily_cutoff, _taipei_timestamp, _known_timestamp
from return_first_model import EntryPlan, ModelDataError, timestamp, finite_scalar

ENGINE_VERSION='8.2.0-policy-evidence-review'
POLICY_VERSION='daily_open1_structural_trail_v2'
ORDER_MODES=('next_open','open_or_pullback_limit')
PRICE_FEATURES=('return20','return60','return120','atr_fraction','rvol',
                'extension_atr','log_avg_trading_value','market_return20')
BUSINESS_FEATURES=('revenue_growth','eps_growth','roe','pb')
FLOW_FEATURES=('foreign_5d_intensity','trust_5d_intensity')
HORIZONS={'short':10,'mid':40,'long':120}

@dataclass(frozen=True)
class PolicyConfig:
    commission: float=.001425
    sell_tax: float=.003
    benchmark_sell_tax: float=.001
    min_fee: float=20.0
    notional: float=100_000.0
    slippage: float=.0005
    min_turnover: float=15_000_000.0
    min_price: float=10.0
    min_history: int=150
    event_stride: int=5
    order_mode: str='next_open'

    def __post_init__(self):
        if self.order_mode not in ORDER_MODES: raise ModelDataError('Unknown execution policy')
        for k in ('commission','sell_tax','benchmark_sell_tax','min_fee','slippage','min_turnover','min_price'):
            v=finite_scalar(getattr(self,k))
            if not np.isfinite(v) or v<0: raise ModelDataError(f'Invalid {k}')
        for k in ('commission','sell_tax','benchmark_sell_tax','slippage'):
            if getattr(self,k)>=.2: raise ModelDataError('Cost inputs must be decimal fractions')
        if not np.isfinite(finite_scalar(self.notional)) or self.notional<=0: raise ModelDataError('Invalid notional')
        if self.min_history<125 or self.event_stride<1: raise ModelDataError('Insufficient warmup / invalid stride')

    @property
    def cost_id(self):
        payload={k:getattr(self,k) for k in ('commission','sell_tax','benchmark_sell_tax','min_fee','notional','slippage')}
        return 'adjusted_ratio_cost_v1_'+hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:12]

    def net_ratio(self, ratio: float, *, benchmark: bool=False) -> float:
        """One-entry/exit, proportional adjusted-value approximation, with fees.
        Slippage is already part of the price ratio. Not a cash dividend ledger.
        """
        if not np.isfinite(ratio) or ratio<0: raise ModelDataError('Invalid price ratio')
        buy=self.notional; sell=buy*ratio
        buyfee=max(self.min_fee,buy*self.commission)
        sellfee=max(self.min_fee,sell*self.commission) if sell>0 else 0.0
        tax=self.benchmark_sell_tax if benchmark else self.sell_tax
        return (sell-sellfee-sell*tax)/(buy+buyfee)-1.0


def policy_id(h: int,order_mode: str='next_open') -> str:
    if h not in (5,10,20,40,60,120,180): raise ModelDataError('Unsupported horizon')
    if order_mode not in ORDER_MODES: raise ModelDataError('Unknown order mode')
    version=POLICY_VERSION if order_mode=='next_open' else 'daily_open_then_gap_pullback_limit_v1'
    return f'{version}_h{h}'


def equity_tick(p: float) -> Decimal:
    """Ordinary Taiwan equity schedule checked 2026-09-30; not futures/ETF.
    Review before future rule changes. No assumption of order execution.
    """
    if not np.isfinite(p) or p<=0: raise ModelDataError('Invalid raw price')
    return Decimal('0.01' if p<10 else '0.05' if p<50 else '0.1' if p<100 else '0.5' if p<500 else '1' if p<1000 else '5')


def price_tick(p: float,up: bool) -> float:
    d=Decimal(str(float(p)))
    for _ in range(3):
        tick=equity_tick(float(d))
        value=(d/tick).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR)*tick
        if equity_tick(max(.01,float(value)))==tick: return max(.01,float(value))
        d=value
    return max(.01,float(d))


def compute_features(pair: dict, benchmark: pd.DataFrame, cfg: PolicyConfig, as_of=None) -> pd.DataFrame:
    if pair.get('execution_verified') is not True or pair.get('signal_adjustment_verified') is not True:
        raise ModelDataError('Unverified raw/adjusted prices; cannot estimate returns')
    s,e=_align_price_pair(pair.get('signal'),pair.get('exec'),as_of)
    if s.empty or len(s)<cfg.min_history: return pd.DataFrame()
    x=s.copy(); c=x['Close']; pc=c.shift(1)
    for n in (5,10,20,60,120,240): x[f'ma{n}']=c.rolling(n).mean()
    tr=pd.concat([x['High']-x['Low'],(x['High']-pc).abs(),(x['Low']-pc).abs()],axis=1).max(axis=1)
    x['atr_fraction']=(tr/pc).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    x['atr_signal']=c*x['atr_fraction']
    for n in (20,60,120): x[f'return{n}']=c.pct_change(n,fill_method=None)
    x['rvol']=x['Volume']/x['Volume'].shift(1).rolling(20).median().replace(0,np.nan)
    x['vdu']=x['Volume'].rolling(5).mean()/x['Volume'].shift(1).rolling(20).mean().replace(0,np.nan)
    spread=x['High']-x['Low']
    x['clv']=((c-x['Low'])/spread.replace(0,np.nan)).fillna(.5)
    x['extension_atr']=(c-x['ma20'])/x['atr_signal'].replace(0,np.nan)
    x['raw_close']=e['Close']; x['exec_factor']=e['Close']/c
    x['exec_atr']=e['Close']*x['atr_fraction']
    x['avg_value']=(e['Close']*e['Volume']).shift(1).rolling(20).mean()
    x['log_avg_trading_value']=np.log(x['avg_value'].where(x['avg_value']>0))
    for n in (10,20,55,60,120):
        x[f'hi{n}']=x['High'].shift(1).rolling(n).max()
        x[f'lo{n}']=x['Low'].shift(1).rolling(n).min()
    bc=benchmark['Close'].sort_index()
    x['market_return20']=bc.pct_change(20,fill_method=None).reindex(x.index)
    ma20=bc.rolling(20).mean(); ma60=bc.rolling(60).mean()
    reg=pd.Series(np.where((bc>ma20)&(ma20>ma60),'bull',np.where((bc<ma20)&(ma20<ma60),'bear','neutral')),index=bc.index)
    x['regime']=reg.reindex(x.index)
    up=(c>x['ma60'])&(x['ma20']>x['ma60'])&(x['ma20']>x['ma20'].shift(5))
    breakout=(c>x['hi55']*1.001)&(c>x['ma60'])
    pullback=up&(x['Low'].rolling(3).min()<=x['ma20']+.5*x['atr_signal'])&(c>=x['ma20'])&(x['extension_atr']<=1.5)&(x['rvol']<=1.3)
    reclaim=(pc<x['ma20'].shift(1))&(c>x['ma20'])
    dry=(c<=x['lo60'])&(x['rvol']<=.5)
    x['setup']=np.select([breakout,pullback,reclaim,up,dry],['BREAKOUT','PULLBACK','RECLAIM','TREND','DRYUP'],default='BASE')
    x['confirmed']=np.select([breakout,pullback,reclaim,up],[(x['rvol']>=1.15)&(x['clv']>=.6),(x['vdu']<=1.05)&(x['clv']>=.55),x['clv']>=.6,x['clv']>=.55],default=False).astype(bool)
    x['tradable_signal']=(x['raw_close']>=cfg.min_price)&(x['avg_value']>=cfg.min_turnover)&(x['Volume']>0)
    # No back-fill of failed dates or missing benchmark values.
    return x.replace([np.inf,-np.inf],np.nan)


def signal_plan(row: dict,h: int) -> dict:
    """All values in adjusted-signal coordinates; deterministic before outcomes."""
    close=finite_scalar(row.get('Close')); atr=finite_scalar(row.get('atr_signal'))
    m20=finite_scalar(row.get('ma20')); m60=finite_scalar(row.get('ma60'))
    setup=str(row.get('setup','BASE'))
    if not np.isfinite([close,atr,m20,m60]).all() or atr<=0: raise ModelDataError('Insufficient structure')
    width=.45 if h<=10 else .65 if h<=40 else 1.0
    if setup=='BREAKOUT':
        pivot=finite_scalar(row.get('hi55'))
        low=pivot; high=pivot+width*atr; trigger=pivot*1.001
        stop=finite_scalar(row.get('lo10' if h<=10 else 'lo20' if h<=40 else 'lo60'))-.25*atr
        mode='breakout_confirmed'
    else:
        anchor=m20 if h<=40 else .5*(m20+m60)
        low=anchor-.30*atr; high=anchor+width*atr
        trigger=finite_scalar(row.get('High'))*1.001
        stop=finite_scalar(row.get('lo20' if h<=40 else 'lo60'))-.25*atr
        mode='zone_confirmed'
    chase=max(high,trigger)+(0.75 if h<=10 else 1.25 if h<=40 else 2.0)*atr
    if not np.isfinite([low,high,trigger,stop,chase]).all() or not 0<stop<min(low,trigger):
        raise ModelDataError('No valid structural stop: do not force one close to price')
    return {'low':low,'high':high,'trigger':trigger,'stop':stop,'chase':chase,'mode':mode}


def build_entry_plan(row: dict,h: int,signal_at,order_mode: str='next_open') -> EntryPlan:
    z=signal_plan(row,h); f=finite_scalar(row.get('exec_factor'))
    if not np.isfinite(f) or f<=0: raise ModelDataError('Missing execution factor')
    t=timestamp(signal_at)
    # A calendar envelope is not a trading-calendar guarantee; simulation only
    # considers the NEXT observed market session. UI explicitly says recheck open.
    return EntryPlan(policy_id=policy_id(h,order_mode),horizon_days=h,setup=str(row.get('setup')),
                     entry_mode=z['mode'],signal_at=t.isoformat(),expires_at=(t+pd.Timedelta(days=4)).isoformat(),
                     zone_low=price_tick(z['low']*f,True),zone_high=price_tick(z['high']*f,False),
                     trigger=price_tick(z['trigger']*f,True),chase_limit=price_tick(z['chase']*f,False),
                     invalidation=price_tick(z['stop']*f,False))


def _one_price_or_no_volume(row) -> bool:
    # Without a limit-order book, a one-price session is not assumed fillable.
    values=[finite_scalar(row.get(k)) for k in ('Open','High','Low','Close','Volume')]
    return not np.isfinite(values).all() or values[4]<=0 or values[1]-values[2]<=max(1e-10,values[3]*1e-9)


def attempt_entry(signal_row, first, plan: dict, cfg: PolicyConfig) -> dict:
    """Fixed next-session entry policy; daily bar is NOT evidence of an actual fill.

    The optional challenger places a limit only after an over-chase opening gap.
    It requires a strict trade-through, rejects one-price/zero-volume bars and
    pays the limit (no favorable fill or slippage beyond the limit). If stop is
    also touched on this day, simulate_event assumes the adverse post-fill stop.
    No comparison of two outcomes is used to select which order was 'sent'.
    """
    no={'filled':False,'entry':None,'reason':'NO_CONFIRMATION','entry_kind':'NONE',
        'intraday_path_unknown':False}
    confirmed=signal_row.get('confirmed',False)
    if not isinstance(confirmed,(bool,np.bool_)) or not bool(confirmed) or str(signal_row.get('setup')) in ('BASE','DRYUP'):
        return no
    if _one_price_or_no_volume(first): return {**no,'reason':'NEXT_OPEN_NOT_FILLABLE'}
    op=float(first['Open']); candidate=op*(1+cfg.slippage)
    inside=(plan['low']<=candidate<=plan['high']) if plan['mode']=='zone_confirmed' else candidate>=plan['trigger']
    if inside and plan['stop']<candidate<=plan['chase']:
        return {**no,'filled':True,'entry':candidate,'reason':'OPEN_ASSUMED','entry_kind':'OPEN'}
    if cfg.order_mode!='open_or_pullback_limit' or op<=plan['chase']:
        return {**no,'reason':'NEXT_OPEN_OUTSIDE_PLAN'}
    limit=min(plan['high'],plan['chase'])
    if not plan['stop']<limit<op: return {**no,'reason':'INVALID_GAP_LIMIT'}
    # One raw tick translated into signal units. Unknown factor fails closed.
    factor=finite_scalar(first.get('exec_factor'))
    if not np.isfinite(factor) or factor<=0: return {**no,'reason':'LIMIT_PRICE_COORDINATE_UNKNOWN'}
    tick=float(equity_tick(limit*factor))/factor
    if float(first['Low'])>limit-tick+1e-10:
        return {**no,'reason':'LIMIT_NOT_TRADED_THROUGH'}
    return {**no,'filled':True,'entry':limit,'reason':'GAP_PULLBACK_LIMIT_ASSUMED',
            'entry_kind':'GAP_PULLBACK_LIMIT','intraday_path_unknown':True}


def simulate_event(x: pd.DataFrame,bench: pd.DataFrame,signal_date,h: int,cfg: PolicyConfig) -> dict:
    """One preselected entry policy, optional gap limit; stop may gap, no target cap.

    Missing mature outcome data raises; it is never silently assigned a profit
    or dropped from the training set. Session calendar comes from benchmark.
    """
    signal_date=pd.Timestamp(signal_date)
    calendar=bench.index
    if signal_date not in calendar or signal_date not in x.index: raise ModelDataError('Signal is not on market calendar')
    pos=calendar.get_loc(signal_date)
    if not isinstance(pos,(int,np.integer)) or pos+h>=len(calendar): raise ModelDataError('IMMATURE_WINDOW')
    dates=calendar[pos+1:pos+h+1]
    if not dates.isin(x.index).all(): raise ModelDataError('UNRESOLVED_MATURE_WINDOW_MISSING_BARS')
    future=x.reindex(dates)
    if future[['Open','High','Low','Close','Volume']].isna().any().any(): raise ModelDataError('UNRESOLVED_MATURE_PRICE')
    row=x.loc[signal_date]
    # Use the same tick-rounded plan as the UI, mapped back to signal units.
    executable=build_entry_plan(row.to_dict(),h,signal_date+pd.Timedelta(hours=16),cfg.order_mode)
    f=float(row['exec_factor'])
    plan={'low':executable.zone_low/f,'high':executable.zone_high/f,
          'trigger':executable.trigger/f,'stop':executable.invalidation/f,
          'chase':executable.chase_limit/f,'mode':executable.entry_mode}
    first=future.iloc[0]; stop=plan['stop']
    attempt=attempt_entry(row,first,plan,cfg)
    # The policy and the displayed four-day envelope must agree, including holidays.
    if timestamp(dates[0]+pd.Timedelta(hours=9))>timestamp(executable.expires_at):
        attempt={'filled':False,'entry':None,'reason':'PLAN_EXPIRED_BEFORE_NEXT_SESSION',
                 'entry_kind':'NONE','intraday_path_unknown':False}
    entry=attempt['entry']; filled=attempt['filled']; nofill=attempt['reason']
    # Baseline buys first observed fillable open within 5 sessions; same end date.
    baseline=None
    for j,(_,day) in enumerate(future.iloc[:min(5,h)].iterrows()):
        if not _one_price_or_no_volume(day): baseline=float(day['Open'])*(1+cfg.slippage); break
    end=future.iloc[-1]
    if _one_price_or_no_volume(end): raise ModelDataError('UNRESOLVED_TERMINAL_LIQUIDATION')
    asset=cfg.net_ratio(float(end['Close'])*(1-cfg.slippage)/baseline) if baseline else 0.0
    bfuture=bench.reindex(dates)
    if bfuture[['Open','Close']].isna().any().any(): raise ModelDataError('BENCHMARK_MISSING')
    benchmark=cfg.net_ratio(float(bfuture['Close'].iloc[-1])*(1-cfg.slippage)/(float(bfuture['Open'].iloc[0])*(1+cfg.slippage)),benchmark=True)
    result=0.0; exit_reason=nofill; exit_date=None
    if filled:
        high_close=entry; initial_risk=entry-stop
        multiplier=2.8 if h<=10 else 3.5 if h<=40 else 4.5
        sold=None
        for daydate,day in future.iterrows():
            if _one_price_or_no_volume(day):
                continue # cannot pretend a stop executed on a locked session
            op,lo=float(day['Open']),float(day['Low'])
            if op<=stop:
                sold=op*(1-cfg.slippage); exit_reason='GAP_OR_STOP_OPEN'; exit_date=daydate; break
            if lo<=stop:
                sold=stop*(1-cfg.slippage); exit_reason='STOP_TOUCH_ASSUMED'; exit_date=daydate; break
            high_close=max(high_close,float(day['Close']))
            # Use today's completed data ONLY for tomorrow's stop.
            if high_close-entry>=initial_risk:
                stop=max(stop,high_close-multiplier*float(day['atr_signal']))
        if sold is None:
            sold=float(end['Close'])*(1-cfg.slippage); exit_reason='HORIZON_CLOSE_ASSUMED'; exit_date=dates[-1]
        result=cfg.net_ratio(sold/entry)
    return {'filled':bool(filled),'strategy_net_return':result,'asset_net_return':asset,
            'benchmark_net_return':benchmark,'evaluation_at':(dates[-1]+pd.Timedelta(hours=16)).isoformat(),
            'label_available_at':(dates[-1]+pd.Timedelta(hours=16)).isoformat(),
            'exit_reason':exit_reason,'exit_at':None if exit_date is None else str(exit_date),
            'outcome_source':'EXPLORATORY_DAILY_OHLC_SIMULATION',
            'execution_policy':cfg.order_mode, 'entry_kind':attempt['entry_kind'],
            'entry_assumed_signal_price':entry,
            'intraday_path_unknown':attempt['intraday_path_unknown'],
            'execution_assumption':'fixed_policy;slippage;one_price_no_fill;adjusted_total_return_proxy;limit_trade_through;adverse_same_bar_stop'}


def attach_external(features: pd.DataFrame,ticker: str,external: pd.DataFrame,as_of=None) -> pd.DataFrame:
    """Optional vintaged data; no price proxies, no fill from future, 400d max age."""
    x=features.copy()
    for c in (*BUSINESS_FEATURES,*FLOW_FEATURES): x[c]=np.nan
    if external is None or external.empty: return x
    if not {'ticker','available_from'}.issubset(external.columns): raise ModelDataError('External data needs ticker,available_from')
    f=external.loc[external['ticker'].eq(ticker)].copy()
    if f.empty: return x
    f['available_from']=f['available_from'].map(timestamp)
    f=f.sort_values('available_from')
    f=f.drop_duplicates('available_from',keep='last')
    # as-of join each data field independently: missing monthly fields do not
    # erase the last available quarter, and dates are retained for staleness.
    for c in (*BUSINESS_FEATURES,*FLOW_FEATURES):
        if c not in f: continue
        valid=f[c].map(finite_scalar).notna()
        field=f.loc[valid,['available_from',c]].copy()
        if field.empty: continue
        field[c]=field[c].map(finite_scalar)
        times=[timestamp(t)+pd.Timedelta(hours=16) for t in x.index]
        if as_of is not None:
            cutoff=timestamp(as_of)
            times=[min(t,cutoff) for t in times]
        left=pd.DataFrame({'at':times})
        merged=pd.merge_asof(left,field,left_on='at',right_on='available_from',direction='backward')
        age=(merged['at']-merged['available_from']).dt.total_seconds()/86400
        max_age=10 if c in FLOW_FEATURES else 400
        x[c]=merged[c].where(age<=max_age).to_numpy()
    return x


def build_ledger(ticker: str,x: pd.DataFrame,benchmark: pd.DataFrame,cfg: PolicyConfig,
                 horizons=(10,40,120)) -> tuple[pd.DataFrame,dict]:
    """Current-survivor reference panel. Missing mature labels poison that horizon.

    Explicit audit entries accompany invalid rows; ReturnFirstSnapshot will reject
    invalid mature labels rather than selecting away unresolved losers.
    """
    rows=[]; audit={'attempts':0,'unresolved':0,'invalid_structure':0,'errors':[]}
    columns=list(PRICE_FEATURES)+list(BUSINESS_FEATURES)+list(FLOW_FEATURES)
    if x.empty: return pd.DataFrame(),audit
    dates=benchmark.index.intersection(x.index)
    next_allowed={}
    for d in dates[cfg.min_history::cfg.event_stride]:
        r=x.loc[d]
        if not bool(r['tradable_signal']): continue
        if not np.isfinite([finite_scalar(r.get(c)) for c in PRICE_FEATURES]).all(): continue
        for h in horizons:
            if h in next_allowed and d<=next_allowed[h]: continue
            pos=benchmark.index.get_indexer([d])[0]
            if pos+h>=len(benchmark): continue
            audit['attempts']+=1
            meta={c:finite_scalar(r.get(c)) for c in columns}
            at=(d+pd.Timedelta(hours=16)).isoformat()
            event=f'{ticker}|{at}|{policy_id(h,cfg.order_mode)}|{cfg.cost_id}'
            meta.update({'event_id':hashlib.sha256(event.encode()).hexdigest()[:24],
                         'ticker':ticker,'signal_at':at,'features_available_at':at,
                         'policy_id':policy_id(h,cfg.order_mode),'cost_model_id':cfg.cost_id,'horizon_days':h,
                         'setup':str(r['setup']),'regime':str(r['regime'])})
            try:
                result=simulate_event(x,benchmark,d,h,cfg)
            except ModelDataError as exc:
                code=str(exc)
                if code.startswith('No valid structural stop') or code.startswith('Insufficient structure'):
                    # Signal-time information determines this exclusion, not outcome.
                    audit['invalid_structure']+=1; continue
                audit['unresolved']+=1
                if len(audit['errors'])<8: audit['errors'].append(f'{ticker} {d.date()} H{h}: {code}')
                end=benchmark.index[pos+h]+pd.Timedelta(hours=16)
                result={'filled':False,'strategy_net_return':np.nan,'asset_net_return':np.nan,
                        'benchmark_net_return':np.nan,'evaluation_at':end.isoformat(),'label_available_at':end.isoformat(),
                        'exit_reason':code,'outcome_source':'UNRESOLVED_SIMULATION'}
            rows.append({**meta,**result})
            next_allowed[h]=benchmark.index[pos+h]
    return pd.DataFrame(rows),audit
