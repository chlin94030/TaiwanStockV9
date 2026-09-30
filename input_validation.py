"""Explicit unit contracts and auditable, atomic input publishing.

No implicit percent guessing. Large growth can be genuine: it is a blocking
review warning until the operator explicitly acknowledges the DECIMAL units.
Unknown future return labels never become a reason to include/exclude a signal.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import csv, hashlib, io, json, math, os, uuid
import numpy as np
import pandas as pd
from return_first_model import ModelDataError, finite_scalar, timestamp

GROWTH_FIELDS = ('revenue_growth','eps_growth','roe')
FLOW_FIELDS = ('foreign_5d_intensity','trust_5d_intensity')
OUTCOMES = ('strategy_net_return','asset_net_return','benchmark_net_return')

@dataclass
class InputReview:
    frame: pd.DataFrame
    warnings: tuple[str, ...]
    unit_mode: str
    content_hash: str


def _read_csv(content: bytes) -> pd.DataFrame:
    if not isinstance(content,bytes) or not content or len(content)>40_000_000:
        raise ModelDataError('CSV must be nonempty and at most 40 MB')
    try:
        text=content.decode('utf-8-sig')
        header=next(csv.reader(io.StringIO(text)))
        if len(header)!=len(set(header)) or any(not h.strip() for h in header):
            raise ModelDataError('Empty/duplicate CSV header')
        d=pd.read_csv(io.StringIO(text),dtype={'ticker':'string','event_id':'string'})
    except (UnicodeError,ValueError,pd.errors.ParserError,StopIteration) as exc:
        raise ModelDataError('Invalid UTF-8 CSV') from exc
    if d.empty: raise ModelDataError('Empty data table')
    return d


def review_csv(content: bytes, kind: str, units: str='decimal') -> InputReview:
    if kind not in ('external','ledger'): raise ModelDataError('Unknown upload kind')
    if units not in ('decimal','percent_points'): raise ModelDataError('Explicit growth units required')
    d=_read_csv(content)
    required=({'ticker','available_from'} if kind=='external' else
              {'ticker','event_id','policy_id','cost_model_id','horizon_days','setup','regime',
               'signal_at','features_available_at','evaluation_at','label_available_at','filled',*OUTCOMES})
    if required-set(d): raise ModelDataError('Missing columns: '+','.join(sorted(required-set(d))))
    if d['ticker'].isna().any(): raise ModelDataError('Ticker cannot be missing')
    d['ticker']=d['ticker'].str.strip().str.upper()
    if not d['ticker'].str.fullmatch(r'\d{4}\.(?:TW|TWO)').all():
        raise ModelDataError('Use explicit ordinary-equity tickers, e.g. 2330.TW / 6187.TWO')
    warns=[]
    cols=[c for c in (*GROWTH_FIELDS,*FLOW_FIELDS,'pb',*OUTCOMES) if c in d]
    for c in cols:
        original=d[c]
        present=original.notna() & original.astype(str).str.strip().ne('')
        if original[present].astype(str).str.contains('%',regex=False).any():
            raise ModelDataError(c+': remove percent signs and declare the unit mode')
        numeric=original.map(finite_scalar)
        if (present & numeric.isna()).any(): raise ModelDataError(c+': invalid/nonfinite numeric value')
        if c in GROWTH_FIELDS and units=='percent_points': numeric=numeric/100.0
        d[c]=numeric.astype('float64')
        if c in GROWTH_FIELDS and (d[c].abs()>5).any():
            warns.append(c+': magnitude exceeds 5 (=500%); verify decimal units/base effects')
        if c in FLOW_FIELDS and (d[c].abs()>5).any():
            warns.append(c+': unusually large flow ratio; confirm denominator and units')
        if c in OUTCOMES:
            if d[c].isna().any() or (d[c]<-1-1e-12).any():
                raise ModelDataError(c+': finite long-only net returns >= -1 are required')
            if (d[c]>5).any(): warns.append(c+': result exceeds +500%; verify decimal returns')
        if c=='pb' and (d[c].dropna()<=0).any():
            raise ModelDataError('pb must be positive when supplied')
    times=['available_from'] if kind=='external' else ['signal_at','features_available_at','evaluation_at','label_available_at']
    for c in times: d[c]=d[c].map(timestamp).map(lambda t:t.isoformat())
    if kind=='external':
        known=set(GROWTH_FIELDS)|set(FLOW_FIELDS)|{'pb'}
        if not known.intersection(d.columns): raise ModelDataError('No supported external factor fields')
        dup=d.duplicated(['ticker','available_from'],keep=False)
        if dup.any(): raise ModelDataError('Repeated ticker/time: merge fields explicitly before upload')
    else:
        if d['event_id'].isna().any() or d['event_id'].duplicated().any():
            raise ModelDataError('event_id missing or duplicated')
        for c in ['policy_id','cost_model_id','setup','regime']:
            if d[c].isna().any() or d[c].astype(str).str.strip().eq('').any():
                raise ModelDataError('Missing '+c)
        def boolean(v):
            if isinstance(v,(bool,np.bool_)): return bool(v)
            if isinstance(v,str) and v.strip().lower() in ('true','false'): return v.strip().lower()=='true'
            raise ModelDataError('filled must be true/false, not 1/0 or an empty cell')
        d['filled']=d['filled'].map(boolean)
        h=d['horizon_days'].map(finite_scalar)
        if h.isna().any() or (h<=0).any() or (h%1!=0).any(): raise ModelDataError('Invalid horizon')
        d['horizon_days']=h.astype(int)
        t={k:pd.to_datetime(d[k],utc=True) for k in times}
        if (t['features_available_at']>t['signal_at']).any(): raise ModelDataError('Future input in past signal')
        if (t['evaluation_at']<=t['signal_at']).any() or (t['label_available_at']<t['evaluation_at']).any():
            raise ModelDataError('Invalid outcome maturity')
        if ((~d['filled']) & (d['strategy_net_return'].abs()>1e-12)).any():
            raise ModelDataError('This ledger policy requires unfilled strategy return = 0')
    return InputReview(d,tuple(warns),units,hashlib.sha256(content).hexdigest())


def publish_inputs(root, uploads: dict[str,bytes], units='decimal', acknowledge_extremes=False):
    """Validate ALL tables before any update; one atomic pointer publishes a batch.

    Files are content-addressed and immutable. Readers resolve the pointer once.
    This avoids partially mixing an old ledger with a new factor upload.
    """
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    allowed={'external.csv':'external','outcome_ledger.csv':'ledger'}
    if not uploads or set(uploads)-set(allowed): raise ModelDataError('Unsupported/empty upload set')
    reviews={name:review_csv(blob,allowed[name],units) for name,blob in uploads.items()}
    warnings=[w for r in reviews.values() for w in r.warnings]
    if warnings and not acknowledge_extremes:
        raise ModelDataError('Unit review required before saving: '+'; '.join(warnings))
    paths={}; receipts={}
    for name,r in reviews.items():
        normalized=r.frame.to_csv(index=False).encode('utf-8')
        key=hashlib.sha256(normalized).hexdigest()
        folder=root/'published_inputs'; folder.mkdir(exist_ok=True)
        path=folder/(key+'_'+name)
        if not path.exists():
            tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            tmp.write_bytes(normalized); os.replace(tmp,path)
        paths[name]=str(path.relative_to(root))
        receipts[name]={'input_sha256':r.content_hash,'normalized_sha256':key,
                        'growth_units':units,'saved_units':'decimal',
                        'extremes_acknowledged':bool(acknowledge_extremes),'warnings':r.warnings}
    pointer=root/'active_inputs.json'
    previous=json.loads(pointer.read_text()) if pointer.exists() else {'files':{},'receipts':{}}
    previous['files'].update(paths); previous['receipts'].update(receipts)
    previous['recorded_at']=pd.Timestamp.now(tz='Asia/Taipei').isoformat()
    tmp=pointer.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(previous,ensure_ascii=False,indent=2))
    os.replace(tmp,pointer)
    return previous


def active_input_paths(root):
    root=Path(root).resolve(); pointer=root/'active_inputs.json'
    if not pointer.exists(): return {}
    doc=json.loads(pointer.read_text()); out={}
    for name,value in doc.get('files',{}).items():
        if name not in ('external.csv','outcome_ledger.csv'): raise ModelDataError('Invalid input manifest')
        path=(root/value).resolve()
        if root not in path.parents or not path.is_file(): raise ModelDataError('Missing/unsafe published input')
        expected=doc.get('receipts',{}).get(name,{}).get('normalized_sha256')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ModelDataError('Published input changed after review')
        out[name]=path
    return out
