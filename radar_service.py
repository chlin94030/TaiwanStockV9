"""Connect the live daily data interface to the V8 return-distribution engine.

No old Edge/rank/Safety percentages are reused as return forecasts. Current
features and frozen outcome data are persisted separately from chart histories.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional
import hashlib, json, gzip, os, time, math, uuid
import numpy as np
import pandas as pd
from market_data import DailyPriceStore, get_universe, resolve_ticker, _taipei_timestamp, _daily_cutoff, _price_as_of_key, provider_runtime_status
from trading_calendar import calendar_reference, daily_freshness
from input_validation import active_input_paths
from corporate_actions import coordinate_review, technical_reference
from policy_engine import (PolicyConfig, ENGINE_VERSION, POLICY_VERSION, PRICE_FEATURES,
                           BUSINESS_FEATURES, FLOW_FEATURES, HORIZONS, compute_features,
                           attach_external, build_entry_plan, build_ledger, policy_id)
from return_first_model import (ForecastConfig, ReturnFirstSnapshot, Forecast, EntryPlan,
                                ResearchMandate, rank_research_candidates, entry_state,
                                holding_review, finite_scalar, timestamp, ModelDataError)

STATUS='RESEARCH_NOT_OOS_VALIDATED'
from operational_tools import OPERATIONS_VERSION, exclusive_scan, ScanBusyError
CHART_ROWS=180
MAX_SESSION_CHARTS=18

@dataclass(frozen=True)
class RunSettings:
    reference_size: int=96
    candidate_size: int=160
    history_period: str='5y'
    model_family: str='price_only'
    benchmark: str='0050.TW'
    order_mode: str='next_open'
    commission: float=.001425
    sell_tax: float=.003
    slippage: float=.0005
    min_fee: float=20.0
    notional: float=100_000.0
    min_turnover: float=15_000_000.0
    min_ev_short: float=.01
    min_ev_mid: float=.03
    min_ev_long: float=.08
    def __post_init__(self):
        if self.order_mode not in ('next_open','open_or_pullback_limit'): raise ModelDataError('Unknown order policy')
        if not 12<=self.reference_size<=300: raise ModelDataError('Reference size must be 12-300')
        if not 30<=self.candidate_size<=600: raise ModelDataError('Candidate size must be 30-600')
        for name in ('min_ev_short','min_ev_mid','min_ev_long'):
            v=finite_scalar(getattr(self,name))
            if not np.isfinite(v) or v<0 or v>=1: raise ModelDataError('Invalid return mandate')
        if self.model_family not in {'price_only','business_confirmed','flow_confirmed','full'}:
            raise ModelDataError('Unknown model family')
        if self.history_period not in {'3y','5y','8y','10y'}: raise ModelDataError('Unsupported research window')
    def policy(self):
        return PolicyConfig(commission=self.commission,sell_tax=self.sell_tax,slippage=self.slippage,
                            min_fee=self.min_fee,notional=self.notional,min_turnover=self.min_turnover,order_mode=self.order_mode)


def json_safe(x):
    if isinstance(x,dict): return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_safe(v) for v in x]
    if isinstance(x,(pd.Timestamp,)): return x.isoformat()
    if isinstance(x,(bool,np.bool_)): return bool(x)
    if isinstance(x,(int,np.integer)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if x is pd.NA or x is pd.NaT: return None
    return x


def dump_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(json_safe(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
    temp.replace(path)


def snapshot_bytes(snap: dict) -> bytes:
    return json.dumps(json_safe(snap),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')


def read_external(data_dir: Path, active_paths=None) -> pd.DataFrame:
    pieces=[]
    active=active_input_paths(data_dir) if active_paths is None else active_paths
    for name in ('fundamentals.csv','flows.csv','external.csv'):
        p=active.get(name,Path(data_dir)/name)
        if not p.exists(): continue
        d=pd.read_csv(p,dtype={'ticker':str})
        if not {'ticker','available_from'}.issubset(d.columns):
            raise ModelDataError(f'{name} requires ticker and actual available_from timestamp')
        d['ticker']=d['ticker'].astype(str).str.strip().str.upper()
        # Explicit contract uses decimal growth rates. Legacy percent-point files
        # must be intentionally converted; do not silently divide unknown units.
        d['available_from']=d['available_from'].map(timestamp).map(lambda t:t.isoformat())
        pieces.append(d)
    return pd.concat(pieces,ignore_index=True,sort=False) if pieces else pd.DataFrame()


def model_features(family: str):
    return PRICE_FEATURES+(BUSINESS_FEATURES if family in {'business_confirmed','full'} else ())+(
        FLOW_FEATURES if family in {'flow_confirmed','full'} else ())


def stratified_reference(u: pd.DataFrame,n: int) -> list[str]:
    """Outcome- and current-momentum-independent order inside a current roster.
    NOT survivor-free historical sampling. Every current industry gets a turn.
    """
    d=u[['ticker','industry']].drop_duplicates('ticker').copy()
    d['_hash']=d['ticker'].map(lambda t:hashlib.sha256(('reference_v1|'+str(t)).encode()).hexdigest())
    groups=[g.sort_values('_hash')['ticker'].tolist() for _,g in d.groupby('industry',sort=True)]
    result=[]; k=0
    while len(result)<min(n,len(d)):
        for g in groups:
            if k<len(g): result.append(g[k])
            if len(result)>=min(n,len(d)): break
        k+=1
    return result


def candidate_union(current: pd.DataFrame,n: int) -> list[str]:
    """Cheap high-recall candidate selection, never labeled an EV forecast.
    No low-volatility reward, no sector ban, and not used to pick training names.
    """
    if current.empty: return []
    d=current.copy()
    momentum=d['return20'].rank(pct=True)*.5+d['return60'].rank(pct=True)*.3+d['return120'].rank(pct=True)*.2
    leaders=momentum.sort_values(ascending=False)
    keep=[]
    for _,group in d.assign(_m=momentum).groupby('industry',sort=True):
        keep.extend(group.nlargest(2,'_m').index.tolist())
    event=d[d['setup'].isin(['BREAKOUT','RECLAIM','PULLBACK'])]
    keep.extend(momentum.reindex(event.index).nlargest(max(10,n//3)).index.tolist())
    # Real supplied business/flow observations may protect earlier price phases.
    for c in ('eps_growth','revenue_growth','foreign_5d_intensity','trust_5d_intensity'):
        if c in d and d[c].notna().any(): keep.extend(d[c].nlargest(max(5,n//10)).index.tolist())
    keep=list(dict.fromkeys(keep))
    protected=set(keep)
    keep=sorted(keep,key=lambda t:(-float(momentum.get(t,0)),str(t)))[:n]
    for ticker in leaders.index:
        if ticker not in keep: keep.append(ticker)
        if len(keep)>=n: break
    return keep[:n]


def freeze_external(data_dir, external):
    if external.empty: return 'none'
    key=_external_hash(external)
    path=Path(data_dir)/'inputs'/f'{key}.csv.gz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp.gz')
        external.to_csv(tmp,index=False,compression='gzip')
        tmp.replace(path)
    return key


def load_frozen_external(data_dir, key):
    if key=='none': return pd.DataFrame()
    if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):
        raise ModelDataError('Invalid frozen external reference')
    path=Path(data_dir)/'inputs'/f'{key}.csv.gz'
    if not path.exists(): raise ModelDataError('Frozen external inputs missing; rebuild market snapshot')
    return pd.read_csv(path,dtype={'ticker':str},float_precision='round_trip')


def _ledger_path(data_dir,key):
    if not isinstance(key,str) or not key or any(c not in '0123456789abcdefimport_' for c in key):
        raise ModelDataError('Invalid outcome-ledger cache key')
    return Path(data_dir)/'outcomes'/f'{key}.csv.gz'



def load_ledger(path: Path) -> pd.DataFrame:
    d=pd.read_csv(path,dtype={'ticker':str,'event_id':str},float_precision='round_trip')
    if 'filled' not in d: raise ModelDataError('Ledger needs filled column')
    def parse(v):
        if isinstance(v,(bool,np.bool_)): return bool(v)
        if str(v).lower()=='true': return True
        if str(v).lower()=='false': return False
        raise ModelDataError('filled must be true/false, never a truthy string')
    d['filled']=d['filled'].map(parse)
    return d


def _save_ledger(d,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+f'.{uuid.uuid4().hex}.tmp.gz')
    d.to_csv(temp,index=False,compression='gzip'); temp.replace(path)


def _config(settings,h):
    return ForecastConfig(horizon_days=h,policy_id=policy_id(h,settings.order_mode),cost_model_id=settings.policy().cost_id,
                          features=model_features(settings.model_family),model_family=settings.model_family,
                          large_gain_return={10:.06,40:.15,120:.30}.get(h,.1))


def fit_models(ledger: pd.DataFrame,as_of,settings: RunSettings):
    models={}; audits={}
    for name,h in HORIZONS.items():
        try:
            # Filtering to declared feature availability is an information-model
            # definition, never a filter on returns. Missing mature outcomes stay.
            d=ledger.loc[pd.to_numeric(ledger['horizon_days'],errors='coerce').eq(h)].copy()
            cols=model_features(settings.model_family)
            for c in cols:
                if c not in d: d[c]=np.nan
            valid=d[list(cols)].apply(lambda s:s.map(finite_scalar)).notna().all(axis=1)
            d=d.loc[valid].copy()
            m=ReturnFirstSnapshot(d,as_of,_config(settings,h))
            models[name]=m; audits[name]=m.export_manifest()
        except (ModelDataError,KeyError,ValueError,TypeError) as exc:
            models[name]=None; audits[name]={'error':str(exc),'status':'ESTIMATOR_UNAVAILABLE'}
    return models,audits


def _available_evidence(row):
    return {'price':all(np.isfinite(finite_scalar(row.get(k))) for k in PRICE_FEATURES),
            'business_fields':sum(np.isfinite(finite_scalar(row.get(k))) for k in BUSINESS_FEATURES),
            'business_required':len(BUSINESS_FEATURES),
            'flow_fields':sum(np.isfinite(finite_scalar(row.get(k))) for k in FLOW_FEATURES),
            'flow_required':len(FLOW_FEATURES)}


def _forecast_one(ticker,row,model,as_of,h,settings):
    plan=None; plan_error=None
    try: plan=build_entry_plan(row,h,row['signal_at'],settings.order_mode)
    except (ModelDataError,ValueError,TypeError) as exc: plan_error=str(exc)
    f=None
    if model is not None:
        f=model.estimate(ticker,row,setup=str(row['setup']),regime=str(row['regime']),features_available_at=row['signal_at'])
    decision='NO_RETURN_ESTIMATE'; research=None
    if plan:
        decision=entry_state(plan,float(row['raw_close']),row['signal_at'],price_verified=True,
                             confirmation_met=bool(row.get('confirmed',False)))
    if f is not None:
        min_ev={10:settings.min_ev_short,40:settings.min_ev_mid,120:settings.min_ev_long}[h]
        mandate=ResearchMandate(min_net_ev=min_ev,max_stop_distance=.08 if h<=10 else .15 if h<=40 else .25,
                                max_expected_shortfall10=.12 if h<=10 else .22 if h<=40 else .35)
        tab=rank_research_candidates([f],{ticker:plan} if plan else {},{ticker:row['raw_close']},mandate)
        research=tab.iloc[0].to_dict()
        if not bool(row.get('tradable_signal',False)):
            research['research_qualified']=False
            research['reasons']+=';CURRENT_LIQUIDITY_OR_HISTORY_GATE'
            decision='LIQUIDITY_GATE'
    return {'forecast':None if f is None else f.to_dict(),'plan':None if plan is None else asdict(plan),
            'plan_error':plan_error,'entry_state':decision,'qualification':research}


def evaluate_current(current,models,settings,as_of):
    out={}
    for ticker,r in current.iterrows():
        row=r.to_dict()
        obj={'ticker':ticker,'name':row.get('name',ticker),'industry':row.get('industry','Unknown'),
             'price':row['raw_close'],'price_date':row['price_date'],'setup':row['setup'],
             'confirmed':bool(row.get('confirmed',False)),'evidence':_available_evidence(row),
             'features':row,'horizons':{}}
        for name,h in HORIZONS.items():
            obj['horizons'][name]=_forecast_one(ticker,row,models.get(name),as_of,h,settings)
        out[ticker]=json_safe(obj)
    return out


def select_view(snapshot: dict,h: str,qualified: bool=True,n: int=3):
    values=[]
    for obj in snapshot.get('stocks',{}).values():
        block=obj['horizons'].get(h,{})
        f=block.get('forecast') or {}; q=block.get('qualification') or {}
        if qualified:
            if not q.get('research_qualified',False): continue
            val=finite_scalar(q.get('uncertainty_adjusted_ev'))
        else:
            # Attack watch is not a recommendation. Require supported estimates
            # and positive potential; unsupported details stay in Doctor / audit.
            if not f.get('estimate_available') or not f.get('sample_supported'): continue
            val=finite_scalar((f.get('asset_potential') or {}).get('mean'))
            if not np.isfinite(val) or val<=0: continue
        values.append((val,obj['ticker'],obj))
    values.sort(key=lambda v:(-v[0],v[1]))
    return [obj for _,_,obj in values[:n]]


def displayed_tickers(snapshot: dict) -> list[str]:
    """Union of cards actually displayed in all views; forecasts are NOT pruned."""
    keep = []
    for h in HORIZONS:
        picked = select_view(snapshot,h,True,3)
        watch = select_view(snapshot,h,False,3)
        for obj in picked + watch:
            t = obj['ticker']
            if t not in keep:
                keep.append(t)
    return keep[:MAX_SESSION_CHARTS]


def compact_chart(chart, price_date):
    """Keep one numeric, dated raw OHLCV tail; never retain arbitrary nested data."""
    if not isinstance(chart,dict):
        return None
    try:
        from market_data import _normalize_ohlcv_frame
        dates = chart.get('dates',[]); values = chart.get('ohlcv',[])
        if len(dates) != len(values) or len(dates) > 15000:
            return None
        d = pd.DataFrame(values,columns=['Open','High','Low','Close','Volume'],index=pd.to_datetime(dates))
        d = _normalize_ohlcv_frame(d)
        cutoff = _taipei_timestamp(price_date).normalize()
        d = d.loc[d.index <= cutoff].tail(CHART_ROWS)
        if d.empty or d.index[-1] != cutoff:
            return None
        return {'dates':d.index.strftime('%Y-%m-%d').tolist(),'ohlcv':d.to_numpy(dtype=float).tolist()}
    except (ValueError,TypeError,KeyError,OverflowError):
        return None


def compact_dashboard(snapshot: dict) -> dict:
    """Idempotent session/file boundary. Never remove score/reference features.

    V8.1 already bounded run_scan charts. This also protects old/restored or
    alternate producer snapshots and strips non-OHLC chart extras.
    """
    if not isinstance(snapshot,dict):
        raise ModelDataError('Invalid dashboard snapshot')
    out = dict(snapshot)
    charts = snapshot.get('charts',{})
    if not isinstance(charts,dict):
        charts = {}
    kept = {}
    for t in displayed_tickers(snapshot):
        stock = snapshot['stocks'][t]
        chart = compact_chart(charts.get(t),stock.get('price_date',snapshot.get('price_date')))
        if chart:
            kept[t] = chart
    out['charts'] = kept
    out['operations_version'] = OPERATIONS_VERSION
    return out


def chart_on_demand(snapshot: dict,ticker: str,data_dir: Path,store_override=None,allow_fetch=False):
    """Exact frozen date only. View rendering reads disk; explicit Doctor may fetch."""
    obj = snapshot.get('stocks',{}).get(ticker,{})
    cutoff = obj.get('price_date',snapshot['price_date'])
    chart = compact_chart(snapshot.get('charts',{}).get(ticker),cutoff)
    if chart:
        return chart
    store = store_override or DailyPriceStore(Path(data_dir)/'daily_prices.sqlite')
    pair = store.get(ticker,'18mo',snapshot['as_of'])
    if pair is None and allow_fetch:
        pairs,_ = store.fetch([ticker],'18mo',snapshot['as_of'])
        pair = pairs.get(ticker)
    if not pair or pair.get('execution_verified') is not True:
        return None
    d = pair['exec'].loc[pair['exec'].index <= pd.Timestamp(cutoff)].tail(CHART_ROWS)
    return compact_chart({'dates':d.index.strftime('%Y-%m-%d').tolist(),
                          'ohlcv':d[['Open','High','Low','Close','Volume']].to_numpy().tolist()},cutoff)


def compact_doctor_result(result: dict) -> dict:
    out = dict(result)
    out['chart'] = compact_chart(result.get('chart'),result['stock']['price_date'])
    return out


def _external_hash(external):
    return hashlib.sha256(external.to_csv(index=False).encode()).hexdigest() if not external.empty else 'none'


def run_scan(data_dir: Path,settings: RunSettings,as_of=None,progress: Optional[Callable]=None,
             universe_override=None,store_override=None,external_override=None) -> dict:
    """One heavy scan per local data directory; charts/Doctor remain read-capable.

    OS lock coordinates cooperating threads/processes. No detached background
    work is launched, and a busy scan never replaces the previous snapshot.
    """
    with exclusive_scan(data_dir):
        return _run_scan_unlocked(data_dir,settings,as_of,progress,universe_override,
                                  store_override,external_override)


def _run_scan_unlocked(data_dir: Path,settings: RunSettings,as_of=None,progress: Optional[Callable]=None,
             universe_override=None,store_override=None,external_override=None) -> dict:
    started=time.time(); data_dir=Path(data_dir); data_dir.mkdir(parents=True,exist_ok=True)
    asof=_taipei_timestamp(as_of); cfg=settings.policy()
    notify=progress if progress is not None else lambda text,p:None
    store=store_override or DailyPriceStore(data_dir/'daily_prices.sqlite')
    # Resolve the publication pointer ONCE for this run; later uploads cannot
    # mix newly published outcomes with the factors already frozen here.
    active_paths=active_input_paths(data_dir)
    external=read_external(data_dir,active_paths) if external_override is None else external_override.copy()
    external_key=freeze_external(data_dir,external)
    if universe_override is None: universe,umeta=get_universe(data_dir)
    else:
        universe=universe_override.copy(); umeta={'source':'test_or_explicit_override','total':len(universe),'full_market_verified':False}
    notify('Loading benchmark / market calendar',.03)
    bp,bd=store.fetch([settings.benchmark],settings.history_period,asof)
    if settings.benchmark not in bp: raise ModelDataError('Benchmark unavailable; no zero-return fallback. '+ '; '.join(bd.get('errors',[])))
    benchmark_pair=bp[settings.benchmark]
    if benchmark_pair.get('signal_adjustment_verified') is not True:
        raise ModelDataError('Benchmark adjustments unverified; return comparison disabled')
    benchmark=benchmark_pair['signal'].copy()
    benchmark=benchmark.loc[benchmark.index<=_daily_cutoff(asof)]
    if benchmark.empty or len(benchmark)<260: raise ModelDataError('Insufficient benchmark history')
    enddate=benchmark.index[-1]
    calendar = calendar_reference(data_dir,asof,allow_fetch=universe_override is None)
    fresh = daily_freshness(enddate.strftime('%Y-%m-%d'),calendar)
    if fresh['status'] in {'STALE','FUTURE_OR_UNFINISHED','INVALID_DATE'}:
        raise ModelDataError('Benchmark date does not match latest scheduled completed session: '
                             + str(calendar.get('expected_date')) + '; got ' + str(enddate.date()))
    rows=[]; quick_stats={'requested':len(universe),'downloaded':0,'feature_valid':0,'unverified':0,'stale':0,'errors':[]}
    metadata=universe.set_index('ticker').to_dict('index')
    for start in range(0,len(universe),32):
        ticks=universe['ticker'].iloc[start:start+32].tolist()
        pairs,diag=store.fetch(ticks,'18mo',asof)
        quick_stats['downloaded']+=len(pairs)
        if len(quick_stats['errors'])<10: quick_stats['errors'].extend(diag.get('errors',[])[:2])
        for ticker,pair in pairs.items():
            try:
                x=compute_features(pair,benchmark,cfg,asof)
                if x.empty or x.index[-1]!=enddate:
                    quick_stats['stale']+=1; continue
                x=attach_external(x,ticker,external,asof)
                r=x.iloc[-1].to_dict()
                if not bool(r['tradable_signal']): continue
                if not np.isfinite([finite_scalar(r.get(c)) for c in PRICE_FEATURES]).all(): continue
                r.update(metadata[ticker]); r['ticker']=ticker
                r['price_date']=enddate.strftime('%Y-%m-%d'); r['signal_at']=(enddate+pd.Timedelta(hours=16)).isoformat()
                # as_of before canonical 16h: only use actual data-available decision
                # timestamp <= current as_of. Closed price inclusion policy is 14h.
                if timestamp(r['signal_at'])>timestamp(asof): r['signal_at']=timestamp(asof).isoformat()
                rows.append(r)
            except ModelDataError as exc:
                quick_stats['unverified']+=1
                if len(quick_stats['errors'])<10: quick_stats['errors'].append(f'{ticker}: {exc}')
        notify('Full-market current features',.05+.35*min(1,(start+32)/max(1,len(universe))))
    current=pd.DataFrame(rows)
    if current.empty: raise ModelDataError('No synchronized, verified, liquid current histories. '+ '; '.join(quick_stats['errors']))
    current=current.set_index('ticker'); quick_stats['feature_valid']=len(current)
    chosen=candidate_union(current,settings.candidate_size)
    # Reference selection uses ALL current company names, not today's quick winners
    # nor today's liquidity filter. Historical liquidity is tested at each signal.
    reference=stratified_reference(universe,settings.reference_size)
    payload=(ENGINE_VERSION,reference,_price_as_of_key(asof),settings.history_period,asdict(cfg),_external_hash(external))
    ledger_key=hashlib.sha256(repr(payload).encode()).hexdigest()[:24]
    ledger_path=_ledger_path(data_dir,ledger_key)
    audit_path=ledger_path.with_suffix('.audit.json')
    imported=active_paths.get('outcome_ledger.csv',data_dir/'outcome_ledger.csv')
    label_source='exploratory_daily_ohlc_current_survivors'
    if imported.exists():
        ledger=load_ledger(imported); label_source='user_supplied_ledger_not_independently_audited'
        ledger_key='import_'+hashlib.sha256(imported.read_bytes()).hexdigest()[:24]
        ledger_path=_ledger_path(data_dir,ledger_key); _save_ledger(ledger,ledger_path)
        ledger_audit={'source':label_source,'reference_requested':0,'reference_downloaded':0,'unresolved':0,'errors':[]}
    elif ledger_path.exists() and audit_path.exists():
        ledger=load_ledger(ledger_path); ledger_audit=json.loads(audit_path.read_text())
    else:
        parts=[]; ledger_audit={'source':label_source,'reference_requested':len(reference),'reference_downloaded':0,
                               'attempts':0,'unresolved':0,'invalid_structure':0,'errors':[]}
        for start in range(0,len(reference),16):
            pairs,dd=store.fetch(reference[start:start+16],settings.history_period,asof)
            ledger_audit['reference_downloaded']+=len(pairs)
            if len(ledger_audit['errors'])<10: ledger_audit['errors'].extend(dd.get('errors',[])[:2])
            for ticker,pair in pairs.items():
                try:
                    x=compute_features(pair,benchmark,cfg,asof)
                    if x.empty: continue
                    x=attach_external(x,ticker,external,asof)
                    part,audit=build_ledger(ticker,x,benchmark,cfg,tuple(HORIZONS.values()))
                    if not part.empty: parts.append(part)
                    for k in ('attempts','unresolved','invalid_structure'): ledger_audit[k]+=audit[k]
                    if len(ledger_audit['errors'])<10: ledger_audit['errors'].extend(audit['errors'][:2])
                except ModelDataError as exc:
                    if len(ledger_audit['errors'])<10: ledger_audit['errors'].append(f'{ticker}: {exc}')
            notify('Constructing fixed-policy historical outcomes',.43+.30*min(1,(start+16)/max(1,len(reference))))
        if not parts: raise ModelDataError('No mature historical outcomes; no fabricated forecasts')
        ledger=pd.concat(parts,ignore_index=True,sort=False)
        _save_ledger(ledger,ledger_path); dump_json(audit_path,ledger_audit)
    notify('Fitting frozen return-distribution models',.76)
    models,model_audits=fit_models(ledger,asof,settings)
    stocks=evaluate_current(current.loc[chosen],models,settings,asof)
    market={'regime':str(current.iloc[0]['regime']),'benchmark':settings.benchmark,
            'above20':float((current['Close']>current['ma20']).mean()),'above60':float((current['Close']>current['ma60']).mean())}
    snap_id=hashlib.sha256((ledger_key+str(asof)+current.to_json()+repr(asdict(settings))).encode()).hexdigest()[:20]
    snapshot={'version':ENGINE_VERSION,'snapshot_id':snap_id,'as_of':asof.isoformat(),'price_date':enddate.strftime('%Y-%m-%d'),
              'status':STATUS,'calendar_at_scan':calendar,'settings':asdict(settings),'source_type':label_source,'universe_meta':umeta,
              'external_key':external_key,'input_versions':{k:p.name for k,p in active_paths.items()},'universe':universe.to_dict('records'),'market':market,'coverage':quick_stats,'candidate_n':len(chosen),'ledger_key':ledger_key,
              'ledger_audit':ledger_audit,'model_audits':model_audits,'stocks':stocks,
              'current_features':{str(i):r.to_dict() for i,r in current.iterrows()},
              'reference_tickers':reference,'run_seconds':time.time()-started,
              'warnings':['Current surviving roster: not survivor-free historical validation',
                          'No official quote feed; daily close only',
                          'Historical OHLC fills and adjusted-value costs are exploratory approximations',
                          'Forecasts are not OOS calibrated; no automatic orders']}
    notify('Preparing display charts',.93)
    charts={}
    for ticker in displayed_tickers(snapshot):
        pair=store.get(ticker,'18mo',asof)
        if pair:
            raw=pair['exec'].loc[pair['exec'].index<=enddate].tail(CHART_ROWS)
            charts[ticker]={'dates':raw.index.strftime('%Y-%m-%d').tolist(),'ohlcv':raw[['Open','High','Low','Close','Volume']].to_numpy().tolist()}
    snapshot['charts']=charts
    snapshot['runtime_at_scan']=provider_runtime_status(store)
    snapshot=compact_dashboard(json_safe(snapshot))
    dump_json(data_dir/'dashboard_snapshot.json',snapshot)
    notify('Snapshot ready',1.0)
    return snapshot


def load_dashboard(path: Path) -> Optional[dict]:
    if not Path(path).exists(): return None
    if Path(path).stat().st_size>25_000_000: raise ModelDataError('Snapshot file too large')
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('version')!=ENGINE_VERSION: return None
    return compact_dashboard(obj)


def diagnose(code: str,snapshot: dict,data_dir: Path,store_override=None):
    settings=RunSettings(**snapshot['settings']); asof=snapshot['as_of']
    meta=pd.DataFrame(snapshot.get('universe',[]))
    if meta.empty:
        meta=pd.DataFrame([{'ticker':t,'name':r.get('name',t),'industry':r.get('industry','Unknown')}
                           for t,r in snapshot.get('current_features',{}).items()])
    ticker=resolve_ticker(code,meta)
    if ticker in snapshot['stocks']:
        return {'stock':snapshot['stocks'][ticker],'snapshot_id':snapshot['snapshot_id'],
                'chart':chart_on_demand(snapshot,ticker,data_dir,store_override=store_override,allow_fetch=True)}
    store=store_override or DailyPriceStore(Path(data_dir)/'daily_prices.sqlite')
    if ticker in snapshot.get('current_features',{}):
        row=dict(snapshot['current_features'][ticker])
    else:
        pairs,diag=store.fetch([ticker,settings.benchmark],'18mo',asof)
        if ticker not in pairs or settings.benchmark not in pairs: raise ModelDataError('No verified quote for requested ticker')
        x=compute_features(pairs[ticker],pairs[settings.benchmark]['signal'],settings.policy(),asof)
        if x.empty or x.index[-1].strftime('%Y-%m-%d')!=snapshot['price_date']:
            raise ModelDataError('Quote does not match frozen snapshot date')
        x=attach_external(x,ticker,load_frozen_external(data_dir,snapshot.get('external_key','none')),asof); row=x.iloc[-1].to_dict()
        known=meta.loc[meta['ticker'].eq(ticker)].iloc[0].to_dict() if meta['ticker'].eq(ticker).any() else {}
        row.update({'name':known.get('name',ticker),'industry':known.get('industry','Unknown'),'price_date':snapshot['price_date'],
                    'signal_at':min(timestamp(asof),timestamp(x.index[-1])+pd.Timedelta(hours=16)).isoformat()})
    ledger=load_ledger(_ledger_path(data_dir,snapshot['ledger_key']))
    models,_=fit_models(ledger,asof,settings)
    obj=evaluate_current(pd.DataFrame([row],index=[ticker]),models,settings,asof)[ticker]
    chart=chart_on_demand(snapshot,ticker,data_dir,store_override=store,allow_fetch=True)
    # Return only the requested stock. Never add it to training or mutate snapshot.
    return {'stock':obj,'snapshot_id':snapshot['snapshot_id'],'chart':chart}


def holding_references(code, snapshot, data_dir, basis_date=None, levels=None, store_override=None):
    """On explicit Doctor submit only; no history retained in returned payload."""
    u=pd.DataFrame(snapshot.get('universe',[]))
    ticker=resolve_ticker(code,u if not u.empty else None)
    store=store_override or DailyPriceStore(Path(data_dir)/'daily_prices.sqlite')
    period=snapshot.get('settings',{}).get('history_period','5y')
    pairs,diag=store.fetch([ticker],period,snapshot['as_of'])
    pair=pairs.get(ticker)
    if not pair:
        return {'coordinate':{'status':'UNVERIFIED_ACTION_COORDINATES','can_use_original':False},
                'technical':{'status':'NO_REFERENCE_HISTORY','reference':None}}
    coordinate=coordinate_review(pair,basis_date,snapshot['as_of'],levels)
    technical=technical_reference(pair,snapshot['as_of'],basis_date)
    return {'coordinate':coordinate,'technical':technical}
