"""Prequential EVENT-PANEL evaluation, not a portfolio/backtest-completeness claim.

Each prediction trains solely on labels available at that prediction time. The
panel must have been collected point-in-time, including losses, non-fills and
then-investable/delisted names. This file CANNOT reconstruct absent panel rows.
Repeatedly looking at these results to tune parameters contaminates this test.
No CAGR/Sharpe is inferred by compounding overlapping event returns.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from return_first_model import ReturnFirstSnapshot, ForecastConfig, timestamp, ModelDataError


def evaluate_panel(ledger: pd.DataFrame, cfg: ForecastConfig, start, end, top_n=3):
    if top_n<1: raise ModelDataError("top_n must be positive")
    begin, finish=timestamp(start), timestamp(end)
    signals=ledger["signal_at"].map(timestamp)
    h=pd.to_numeric(ledger["horizon_days"],errors="coerce")
    mask=(signals>=begin)&(signals<=finish)&(h==cfg.horizon_days)&ledger["policy_id"].eq(cfg.policy_id)&ledger["cost_model_id"].eq(cfg.cost_model_id)
    targets=ledger.loc[mask].copy(); targets["_time"]=signals.loc[mask]
    rows=[]
    for when,group in targets.groupby("_time",sort=True):
        model=ReturnFirstSnapshot(ledger,when,cfg)
        for _,row in group.iterrows():
            f=model.estimate(str(row["ticker"]),{k:row[k] for k in cfg.features},setup=str(row["setup"]),
                             regime=str(row["regime"]),features_available_at=row["features_available_at"])
            mature=timestamp(row["label_available_at"])<=finish
            rows.append({"event_id":row["event_id"],"signal_at":when.isoformat(),"ticker":row["ticker"],
                         "snapshot_id":model.snapshot_id,"reference_n":model.reference_n,
                         "sample_supported":f.sample_supported,
                         "predicted_net_ev":f.strategy["mean"] if f.strategy else np.nan,
                         "uncertainty_adjusted_ev":f.uncertainty_adjusted_ev,
                         "actual_net_return":float(row["strategy_net_return"]) if mature else np.nan,
                         "actual_alpha":float(row["strategy_net_return"]-row["benchmark_net_return"]) if mature else np.nan,
                         "realized_available":mature,"reason":";".join(f.reasons)})
    result=pd.DataFrame(rows)
    if result.empty: return result,{"predictions":0,"status":"NO_EVENTS_IN_WINDOW"}
    result["selected"]=False
    for _,g in result.groupby("signal_at",sort=True):
        # Do not filter on future realization to select positions.
        c=g[g["sample_supported"] & (g["predicted_net_ev"]>0) & (g["uncertainty_adjusted_ev"]>0)]
        ix=c.sort_values(["uncertainty_adjusted_ev","predicted_net_ev","ticker"],ascending=[False,False,True]).head(top_n).index
        result.loc[ix,"selected"]=True
    settled=result[result["realized_available"] & result["selected"]]
    all_pred=result[result["realized_available"] & result["sample_supported"] & result["predicted_net_ev"].notna()]
    correlations=[]
    for _,g in all_pred.groupby("signal_at"):
        if len(g)>=5 and g["predicted_net_ev"].nunique()>1 and g["actual_net_return"].nunique()>1:
            correlations.append(float(g["predicted_net_ev"].rank().corr(g["actual_net_return"].rank())))
    summary={"predictions":len(result),"supported_predictions":int(result["sample_supported"].sum()),
             "selected_signals":int(result["selected"].sum()),"selected_matured":len(settled),
             "mean_selected_net_return":float(settled["actual_net_return"].mean()) if len(settled) else None,
             "mean_selected_alpha":float(settled["actual_alpha"].mean()) if len(settled) else None,
             "median_cross_section_rank_ic":float(np.median(correlations)) if correlations else None,
             "supported_prediction_mae":float((all_pred["predicted_net_ev"]-all_pred["actual_net_return"]).abs().mean()) if len(all_pred) else None,
             "status":"EVENT_PANEL_ONLY_NOT_FULL_MARKET_NOT_PORTFOLIO_RETURN",
             "config":asdict(cfg)}
    return result,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("ledger"); p.add_argument("config"); p.add_argument("--start",required=True);p.add_argument("--end",required=True)
    p.add_argument("--output",default="walk_forward_output")
    a=p.parse_args();d=pd.read_csv(a.ledger)
    if "filled" in d:
        mapped=d["filled"].astype(str).str.lower().map({"true":True,"false":False})
        if mapped.isna().any(): raise ModelDataError("CSV filled values must be true/false")
        d["filled"]=mapped
    config=json.loads(Path(a.config).read_text()); config["features"]=tuple(config["features"])
    r,s=evaluate_panel(d,ForecastConfig(**config),a.start,a.end)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    r.to_csv(out/"predictions.csv",index=False)
    (out/"summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__": main()
