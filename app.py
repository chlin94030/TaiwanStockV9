"""
Taiwan Alpha Radar V8.2.1 Return-First Research App (operations patch).
Five mobile views. No Edge 100, artificial confidence grades or automatic orders.
Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import html
import json
import math
import os
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp, _daily_cutoff, provider_runtime_status, _finmind_token
from operational_tools import ScanBusyError, safe_error_text
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed, save_closure_notice
from input_validation import publish_inputs
from presentation import plain_summary
from policy_engine import ENGINE_VERSION, HORIZONS
from return_first_model import holding_review, finite_scalar, ModelDataError, timestamp

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))
VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股"]
HORIZON_LABELS = {"short": "短線 · 10交易日", "mid": "中線 · 40交易日", "long": "長波段 · 120交易日"}
FAMILY_LABELS = {
    "價量研究｜無財報／法人代理": "price_only",
    "基本面確認｜需要歷史公告資料": "business_confirmed",
    "法人確認｜需要歷史籌碼資料": "flow_confirmed",
    "完整證據｜基本面＋法人": "full",
}
SETUP_LABELS = {
    "BREAKOUT": "平台突破", "PULLBACK": "趨勢回測", "RECLAIM": "重新站回",
    "TREND": "趨勢延續觀察", "BASE": "整理觀察", "DRYUP": "低量新低觀察",
}
STATE_LABELS = {
    "CONDITIONS_MET_NOT_FILLED": "收盤條件符合，次日開盤再確認",
    "WAIT_ENTRY_ZONE": "等待回到布局區",
    "WAIT_BREAKOUT": "等待突破確認",
    "WAIT_CONFIRMATION": "量價確認尚未成立",
    "DO_NOT_CHASE": "超出追價上限，等待新機會",
    "INVALIDATED": "原結構已失效",
    "DATA_UNVERIFIED": "資料待核對",
    "PLAN_EXPIRED_REBUILD_NOT_REANCHOR": "計畫已過期，需重新評估",
    "QUOTE_BEFORE_PLAN": "報價早於計畫",
    "NO_RETURN_ESTIMATE": "沒有可用的報酬估計",
    "LIQUIDITY_GATE": "目前流動性／歷史資料不符合交易條件",
}
REASON_LABELS = {
    "INSUFFICIENT_MATURE_SAME_POLICY_STATE": "相同策略與市場狀態的成熟樣本不足",
    "MISSING_REQUIRED_FEATURES": "缺少這套模型需要的真實資料",
    "NO_SUPPORTED_ANALOGUE_DO_NOT_EXTRAPOLATE": "找不到足夠相似案例，不外推報酬",
    "LOW_EFFECTIVE_SAMPLE": "局部有效樣本偏少",
    "FEW_INDEPENDENT_TIME_BLOCKS": "樣本集中在少數時間區段",
    "ISSUER_CONCENTRATION": "樣本過度集中於少數公司",
    "PRIOR_DOMINATED": "主要依賴較廣的歷史分布",
    "EDGE_DEPENDS_ON_BEST_TIME_BLOCK": "正收益依賴最有利的一段行情",
    "UNCERTAINTY_NOT_IDENTIFIABLE": "估計不確定性無法辨識",
    "NO_RETURN_ESTIMATE": "沒有可用報酬估計",
    "EVIDENCE_INSUFFICIENT": "報酬證據不足",
    "NET_EV_BELOW_REQUIREMENT": "淨期望報酬未達研究門檻",
    "UNCERTAINTY_NOT_COMPENSATED": "淨報酬尚不足以補償估計不確定性",
    "ALPHA_BELOW_REQUIREMENT": "相對基準的收益未達門檻",
    "TAIL_RISK_OVER_BUDGET": "最差情境損失超過研究預算",
    "STRUCTURAL_RISK_OVER_BUDGET": "可接受進場價至失效點的距離太大",
    "MISSING_OR_MISMATCHED_PLAN": "缺少有效的原始交易計畫",
    "CURRENT_STRUCTURE_INVALID": "目前價格已不符合原結構",
    "LONG_HORIZON_BUSINESS_EVIDENCE_MISSING": "缺少長波段必需的基本面證據，僅作價量觀察",
    "REFERENCE_DATA_INCOMPLETE": "歷史結果資料有未解決缺口",
    "CURRENT_LIQUIDITY_OR_HISTORY_GATE": "目前流動性或歷史資料未達必要門檻",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格資料未核對，暫不作留／賣判斷",
    "ORIGINAL_STRUCTURE_INVALIDATED": "原始結構失效：優先檢視退出與可成交條件",
    "ORIGINAL_THESIS_INVALIDATED": "原始買進理由失效：應重新檢視持有決策",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "已觸及保護價：核對報價及實際執行條件",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始失效點或持有理由，無法完整判斷",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "原規則尚未被破壞；不是續抱收益保證",
}

CSS = """
<style>
:root{--ink:#17243b;--muted:#66748b;--line:#e5eaf2;--blue:#1d4ed8;}
.stApp{background:#f5f7fb;color:var(--ink)}
.block-container{max-width:1020px;padding-top:1.1rem;padding-bottom:4rem;}
.hero{padding:24px;border-radius:24px;background:linear-gradient(125deg,#102139,#173c70 65%,#126975);color:white;margin-bottom:18px;}
.hero .eyebrow{letter-spacing:.12em;font-size:.75rem;opacity:.8}
.hero h1{font-size:1.9rem;line-height:1.25;color:white;margin:.35rem 0;}
.hero p{font-size:1rem;opacity:.85;margin:.45rem 0 0;line-height:1.6}
.topnote{padding:14px 17px;border:1px solid #d7e1f3;background:#eef4ff;border-radius:14px;margin:12px 0;color:#334a70;font-size:.94rem;line-height:1.7}
.statusline{font-size:.89rem;color:#556580;line-height:1.65;margin:8px 0 14px}
.card{background:white;border:1px solid var(--line);border-radius:21px;padding:22px;margin:14px 0 10px;box-shadow:0 6px 24px rgba(19,42,80,.04)}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;}
.card-title{font-size:1.3rem;font-weight:850;line-height:1.4}
.card-code{font-size:.83rem;color:var(--muted);margin-top:4px}
.card-price{font-size:1.22rem;font-weight:800;text-align:right;white-space:nowrap}
.badge{display:inline-block;font-size:.78rem;font-weight:700;padding:5px 9px;border-radius:8px;background:#f0f3f8;color:#52617a;margin:10px 5px 7px 0}
.badge-blue{color:#234aaa;background:#eaf0ff}
.return-box{background:#f6f9ff;border:1px solid #e4ebf9;border-radius:15px;padding:15px;margin:12px 0;}
.return-k{font-size:.9rem;color:#4c5c75}.return-v{font-size:2.15rem;font-weight:900;letter-spacing:-.04em;line-height:1.4;color:#1e46a3;}
.return-desc{font-size:.82rem;color:#708098;line-height:1.5}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:12px 0;}
.stat{border:1px solid #edf0f6;background:#fcfdff;padding:12px 10px;border-radius:12px;}
.stat .k{font-size:.77rem;color:#6a7890;line-height:1.4}.stat .v{font-size:1.1rem;font-weight:800;margin-top:5px;}
.decision{padding:13px 15px;border-radius:12px;background:#fff5df;border:1px solid #f4dfb1;color:#88621a;font-weight:750;font-size:.99rem;line-height:1.6}
.decision-ok{background:#eaf6f0;border-color:#d1e9dd;color:#225e44}
.evidence{font-size:.87rem;line-height:1.7;color:#69758c;margin-top:13px}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;}
.level{background:#f8fafc;border:1px solid #edf0f5;border-radius:11px;padding:10px}
.level .k{font-size:.8rem;color:#62718a}.level .v{font-weight:800;font-size:1rem;margin-top:5px;}
.subnote{font-size:.85rem;color:#758198;line-height:1.65;margin:10px 0}
.stButton>button,.stFormSubmitButton>button{border-radius:12px;min-height:48px;font-weight:750;}
div[data-testid="stRadio"] div[role="radiogroup"]{gap:.4rem;flex-wrap:wrap}
div[data-testid="stRadio"] label{background:white;border:1px solid #e3e8f2;border-radius:10px;padding:9px 13px}
@media(max-width:650px){
 .block-container{padding:.8rem .8rem 3rem;}.hero{padding:19px 17px;border-radius:19px}.hero h1{font-size:1.5rem}.hero p{font-size:.9rem}
 .card{padding:17px 14px;border-radius:17px}.card-title{font-size:1.15rem}.card-price{font-size:1.13rem}.return-v{font-size:1.92rem}
 .stats{grid-template-columns:repeat(2,1fr)}.levels{grid-template-columns:repeat(2,1fr)}
 .stat .v{font-size:1.12rem}.stRadio label{font-size:.87rem}
}
</style>
"""


def esc(value):
    return html.escape(str(value))


def percent(value, signed=True):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v*100:{'+' if signed else ''}.2f}%"


def money(value):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v:,.2f}".rstrip("0").rstrip(".")


def reason_texts(values):
    return [REASON_LABELS.get(code, code) for code in dict.fromkeys(v for v in values if v)]


def render_chart(chart, plan, key):
    if not chart:
        st.caption("這檔K線未常駐記憶體；個股診斷會嘗試讀取同日期快取。")
        return
    d = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart["dates"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.045, row_heights=[.74, .26])
    fig.add_trace(go.Candlestick(x=dates, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                                increasing_line_color="#d85053", decreasing_line_color="#22917e",
                                name="原始價格"), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=d.Volume, name="成交量",
                        marker_color=np.where(d.Close >= d.Open, "#d85053", "#22917e")), row=2, col=1)
    if plan:
        fig.add_hrect(y0=plan["zone_low"], y1=plan["zone_high"], line_width=0,
                      fillcolor="rgba(43,102,210,.10)", row=1, col=1)
        fig.add_hline(y=plan["trigger"], line_color="#ca902b", line_dash="dot", row=1, col=1)
        fig.add_hline(y=plan["invalidation"], line_color="#ba5360", line_dash="dash", row=1, col=1)
    fig.update_layout(height=430, margin=dict(l=7, r=8, t=10, b=8), showlegend=False,
                      xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type="category", nticks=5, fixedrange=True, showgrid=False)
    fig.update_yaxes(fixedrange=True, gridcolor="#e9edf4")
    st.caption("原始成交K線｜淡藍：布局區；金色點線：確認價；紅色虛線：結構失效。只供日線研究。")
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False})


def card(obj, h, snap, view, chart=None, calendar=None):
    block = obj["horizons"][h]
    f = block.get("forecast") or {}
    plan = block.get("plan")
    q = block.get("qualification") or {}
    summary = f.get("strategy") or {}
    asset = f.get("asset_potential") or {}
    enough = bool(f.get("estimate_available") and f.get("sample_supported"))
    qualified = bool(q.get("research_qualified"))
    condition = block.get("entry_state", "NO_RETURN_ESTIMATE")
    state_label = STATE_LABELS.get(condition, condition)
    # Being numerically entry-ready is not an action recommendation without return evidence.
    if not enough:
        headline = "報酬證據不足，僅供觀察"
    elif not qualified:
        headline = "未通過研究篩選｜" + state_label
    else:
        headline = state_label
    calendar = calendar or calendar_reference(DATA_DIR, now=_taipei_timestamp())
    fresh = daily_freshness(obj["price_date"], calendar)
    review_ok = entry_review_allowed(obj["price_date"], plan, calendar)
    if not fresh["current_daily"]:
        headline = ("交易日曆尚未核對，僅供歷史研究" if fresh["status"] == "UNKNOWN"
                    else "不是最近完成日線，請更新後再評估")
    elif calendar.get("next_open_passed"):
        headline = "已進入下一交易時段；最新日線不是盤中即時報價"
    elif plan and not review_ok:
        headline = "日線仍有效，但原計畫已過期或時間未核對，請重新評估"
    evidence = obj.get("evidence", {})
    ev = percent(summary.get("mean")) if enough else "—"
    summary_sentence = plain_summary(f)
    status = "合格研究候選" if qualified else "攻擊觀察／未通過篩選"
    setup = SETUP_LABELS.get(obj.get("setup"), obj.get("setup", ""))
    origin = "日線模擬估計，未完成樣本外驗證" if snap.get("source_type", "").startswith("exploratory") else "外部結果資料估計，尚未獨立驗證"
    st.markdown(f"""
<div class="card">
 <div class="card-head"><div><div class="card-title">{esc(obj.get('name', obj['ticker']))}</div>
 <div class="card-code">{esc(obj['ticker'])} · {esc(obj.get('industry',''))}</div></div>
 <div class="card-price">{money(obj['price'])}<div class="card-code">{esc(obj['price_date'])} 日線</div></div></div>
 <span class="badge badge-blue">{esc(HORIZON_LABELS[h])}</span><span class="badge">{esc(setup)}</span><span class="badge">{esc(status)}</span>
 <div class="topnote">{esc(summary_sentence)}</div>
 <div class="return-box"><div class="return-k">策略淨報酬 · 研究估計</div><div class="return-v">{ev}</div>
 <div class="return-desc">{esc(origin)}。已含設定成本；不是保證報酬或上漲機率。</div></div>
 <div class="stats">
  <div class="stat"><div class="k">同期超額報酬</div><div class="v">{percent(f.get('alpha_mean')) if enough else '—'}</div></div>
  <div class="stat"><div class="k">中間情境 · 中位數</div><div class="v">{percent(summary.get('median')) if enough else '—'}</div></div>
  <div class="stat"><div class="k">偏佳情境 · 第75分位</div><div class="v">{percent(summary.get('p75')) if enough else '—'}</div></div>
  <div class="stat"><div class="k">偏差情境 · 第10分位</div><div class="v">{percent(summary.get('p10')) if enough else '—'}</div></div>
 </div>
 <div class="decision {'decision-ok' if qualified and review_ok and condition=='CONDITIONS_MET_NOT_FILLED' else ''}">{esc(headline)}</div>
 <div class="evidence">基本面：{int(evidence.get('business_fields',0))}/{int(evidence.get('business_required',4))} 欄 ·
 法人：{int(evidence.get('flow_fields',0))}/{int(evidence.get('flow_required',2))} 欄<br>
 局部有效樣本 {float(f.get('local_effective_n',0)):.1f} · {int(f.get('local_time_blocks',0))} 個時段 ·
 {int(f.get('local_tickers',0))} 家公司。沒有「證據A」或假信心箭頭。</div>
</div>""", unsafe_allow_html=True)
    with st.expander("進出場計畫、收益分布與 K 線", expanded=False):
        if plan:
            mode = "等待布局區＋轉強確認" if plan["entry_mode"] == "zone_confirmed" else "突破確認後進場"
            execution_note = ("\u50c5\u4e0b\u4e00\u500b\u5e02\u5834\u4ea4\u6613\u65e5\u958b\u76e4\u5617\u8a66\u4e00\u6b21\u3002" if snap.get("settings", {}).get("order_mode", "next_open") == "next_open" else
                              "\u4e0b\u4e00\u4ea4\u6613\u65e5\u958b\u76e4\u5148\u6838\u5c0d\uff1b\u50c5\u7576\u8df3\u7a7a\u8d85\u904e\u4e0d\u8ffd\u50f9\u4e0a\u9650\uff0c\u624d\u4f9d\u9810\u5b9a\u56de\u6e2c\u9650\u50f9\u6a21\u64ec\u7576\u65e5\u4ea4\u6613\u3002")
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">布局區</div><div class="v">{money(plan['zone_low'])}–{money(plan['zone_high'])}</div></div>
<div class="level"><div class="k">確認價</div><div class="v">{money(plan['trigger'])}</div></div>
<div class="level"><div class="k">不追價上限</div><div class="v">{money(plan['chase_limit'])}</div></div>
<div class="level"><div class="k">原始結構失效</div><div class="v">{money(plan['invalidation'])}</div></div>
</div><div class="subnote">本次固定路徑：{esc(mode)}。{esc(execution_note)}日線資料無法確認實際委託成交。不是在兩種策略中事後挑贏家。</div>""", unsafe_allow_html=True)
            if snap.get("settings",{}).get("order_mode")=="open_or_pullback_limit":
                st.warning("本快照採實驗政策：開盤超出不追價上限時，當日回測到事前固定限價才模擬成交。日K內時序未知，同日停損採不利假設；不保證真實可成交。")
            current_risk = (float(obj["price"]) - plan["invalidation"]) / float(obj["price"])
            st.caption(f"現價至結構失效點：約{percent(current_risk, False)}。跳空與流動性可能使實際損失更大；不為美化風險報酬比而縮近失效點。")
        elif block.get("plan_error"):
            st.warning("暫無可靠進場計畫：" + block["plan_error"])
        reasons = list(f.get("reasons") or ())
        reasons += (q.get("reasons") or "").split(";")
        if reasons:
            st.write("尚待改善：" + "；".join(reason_texts(reasons)))
        if f.get("estimate_available"):
            detail = [
                ("策略淨平均報酬", percent(summary.get("mean"))),
                ("固定終點資產潛力／買入持有基線", percent(asset.get("mean"))),
                ("策略最差10%結果的平均損失", percent(summary.get("expected_shortfall10_loss"), False)),
                ("相似樣本盈利頻率（非校準機率）", percent(summary.get("p_profit"), False)),
                ("平均盈利／平均虧損", f"{percent(summary.get('average_win'))} / {percent(summary.get('average_loss'),False)}"),
                ("模擬成交頻率", percent(f.get("fill_probability"), False)),
                ("有成交條件下的淨平均報酬", percent(f.get("conditional_filled_mean"))),
                ("扣估計不確定性後研究排序值", percent(f.get("uncertainty_adjusted_ev"))),
                ("去掉最佳時段後局部平均報酬", percent(f.get("local_mean_without_best_block"))),
                ("局部樣本權重／其餘向廣泛同政策樣本收縮", percent(f.get("local_weight"), False)),
                ("報酬模型版本", str(f.get("snapshot_id", ""))),
            ]
            st.dataframe(pd.DataFrame(detail, columns=["項目", "研究估計"]),
                         use_container_width=True, hide_index=True,
                         key=f"detail_{view}_{h}_{obj['ticker']}_{snap['snapshot_id']}")
            if not enough:
                st.caption("以上僅為未達樣本門檻的診斷值，首頁不將其顯示成可靠預測。")
        chart = chart or snap.get("charts", {}).get(obj["ticker"])
        if chart is None:
            try:
                chart = service.chart_on_demand(snap,obj["ticker"],DATA_DIR,allow_fetch=False)
            except (OSError, ValueError) as exc:
                st.caption("圖表快取暫時無法讀取：" + type(exc).__name__)
        render_chart(chart, plan, f"chart_{view}_{h}_{obj['ticker']}_{snap['snapshot_id']}")


def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS[h])
    st.caption("先看可承擔風險下的淨期望收益；沒有合格者就空缺，不為金融或電子設定名額。")
    picked = service.select_view(snap, h, qualified=True, n=3)
    if picked:
        industries = [o["industry"] for o in picked]
        if len(set(industries)) < len(industries):
            st.warning("候選含同產業曝險；這不是多個獨立機會。原始收益排序未被產業配額改寫。")
        for obj in picked:
            card(obj, h, snap, h, calendar=calendar)
    else:
        st.info("本期沒有通過報酬、樣本及風險門檻的研究候選。")
    watch = service.select_view(snap, h, qualified=False, n=3)
    watch = [o for o in watch if o["ticker"] not in {p["ticker"] for p in picked}]
    if watch:
        st.markdown("#### 攻擊潛力觀察")
        st.caption("以下依固定終點資產潛力排列，尚未通過策略或證據條件；不是立即買進名單。")
        for obj in watch:
            card(obj, h, snap, "watch_" + h, calendar=calendar)
    elif not picked:
        st.caption("目前也沒有樣本充分、潛在均值為正的觀察標的。可在個股頁查看缺漏，不會補上不合格股票。")
    if h == "long" and snap["settings"]["model_family"] == "price_only":
        st.warning("目前採價量模型：缺少歷史基本面時，長波段只能作價量研究觀察，不列為完整長期投資推薦。")


def diagnostics(snap):
    with st.expander("本次資料、模型範圍與限制", expanded=False):
        c = snap["coverage"]
        st.write(f"名冊 {c['requested']} → 日線取得 {c['downloaded']} → 同日有效／流動性通過 {c['feature_valid']} → 報酬評估 {snap['candidate_n']}")
        st.write(f"收益模型參考股票：預先依產業分層抽取 {len(snap.get('reference_tickers',[]))} 檔，不是今天上榜者才拿來訓練。")
        st.caption("參考名冊仍是目前存續股票，未還原下市公司與當時全部成分；不能稱為完整全市場歷史回測。")
        st.json({"source": snap["source_type"], "model_family": snap["settings"]["model_family"],
                 "costs": {k:snap["settings"][k] for k in ("commission","sell_tax","slippage","notional")},
                 "reference_audit": snap["ledger_audit"], "models": snap["model_audits"]}, expanded=False)
        if snap.get("runtime_at_scan"):
            st.caption("掃描時的環境與快取狀態（套件可載入不代表外部服務可連線）")
            st.json(snap["runtime_at_scan"],expanded=False)
        errors = snap["coverage"].get("errors", [])
        if errors:
            st.code("\n".join(errors[:8]), language=None)
        st.download_button("匯出本次完整快照（不含密鑰）", service.snapshot_bytes(snap),
                           file_name=f"radar_{snap['snapshot_id']}.json", mime="application/json",
                           key="export_snapshot_v81")


def main():
    st.set_page_config(page_title="Alpha Radar · Return First", page_icon="📈", layout="centered",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div class="eyebrow">TAIWAN ALPHA RADAR · V8.2.1</div>
<h1>先看收益空間，再決定怎麼進場</h1>
<p>短・中・長線收益研究 × 真實價格條件 × 持股原則<br>不以安全分代替收益，也不再把第一名顯示成100分。</p></div>""", unsafe_allow_html=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if st.session_state.get("v8_version") != service.OPERATIONS_VERSION:
        for key in ("v8_snapshot", "v8_doctor", "v8_error"):
            st.session_state.pop(key, None)
        st.session_state["v8_version"] = service.OPERATIONS_VERSION
        try:
            previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json")
            if previous:
                st.session_state["v8_snapshot"] = service.compact_dashboard(previous)
        except (OSError, ValueError, ModelDataError):
            pass

    with st.sidebar:
        st.markdown("### 資料與研究設定")
        st.caption("這些是研究假設，不是已優化的獲利參數。")
        family_label = st.selectbox("報酬模型的資料範圍", list(FAMILY_LABELS), key="family_v81")
        refs = st.selectbox("歷史參考股票數", [96, 160, 240], key="reference_v81")
        period = st.selectbox("歷史研究長度", ["5y", "8y", "10y", "3y"], key="period_v81")
        if period in ("8y", "10y"):
            st.warning("8y/10y採現存股票名冊，未補回歷史下市成分與當時名冊；有倖存者偏誤，資料長不代表更準。3y/5y也不能排除同類偏誤。")
        with st.expander("費用與深度評估設定", expanded=False):
            commission = st.number_input("單邊手續費（%）", min_value=0., max_value=2., value=.1425, format="%.4f", key="fee_v81")
            tax = st.number_input("賣出證交稅假設（%）", min_value=0., max_value=2., value=.30, format="%.3f", key="tax_v81")
            slip = st.number_input("單邊滑價假設（%）", min_value=0., max_value=2., value=.05, format="%.3f", key="slip_v81")
            amount = st.number_input("每次模擬資金（元）", min_value=1000., value=100000., step=10000., key="amount_v81")
            count = st.selectbox("深度候選數", [160, 300, 600], key="candidates_v81")
        with st.expander("執行政策（研究設定）", expanded=False):
            order_label=st.selectbox("事前選定一套政策", ["原則：次日開盤一次", "實驗：跳空過高後等限價回測"], key="order_mode_v82")
            order_mode="next_open" if order_label.startswith("原則") else "open_or_pullback_limit"
            st.caption("實驗版不會自動取代原政策；獨立建立帳本，不能回測後才挑獲利較高的路徑。日K無法證明實際成交與排隊順序。")
        with st.expander("攻擊型收益研究門檻", expanded=False):
            st.caption("門檻是可修改的研究要求，不是回測最佳參數。低於要求的潛力標的仍可列觀察。")
            ev_short = st.number_input("短線10日最低淨期望報酬（%）", min_value=0., max_value=50., value=1., key="ev_short_v81")
            ev_mid = st.number_input("中線40日最低淨期望報酬（%）", min_value=0., max_value=50., value=3., key="ev_mid_v81")
            ev_long = st.number_input("長波段120日最低淨期望報酬（%）", min_value=0., max_value=50., value=8., key="ev_long_v81")
        with st.expander("上傳已整理的歷史資料", expanded=False):
            st.caption("可上傳結果帳本或基本面／法人歷史。所有成長率用小數，例如0.20代表20%；必須有available_from。")
            external = st.file_uploader("外部因子CSV", type=["csv"], key="external_file_v81")
            ledger_file = st.file_uploader("結果帳本CSV", type=["csv"], key="ledger_file_v81")
            units_label=st.selectbox("營收、EPS成長與ROE單位",["小數（0.20 = 20%）","百分點（20 = 20%）"],key="units_v82")
            ack=st.checkbox("我已核對超過500%等異常值的單位及來源（僅真實極端值才勾選）",key="ack_units_v82")
            st.caption("僅明確選擇百分點時才轉換上列三個欄位；報酬結果和法人比率必須原本就是小數。")
            if st.button("保存上傳檔案，下次掃描生效", key="save_inputs_v81"):
                try:
                    uploads={name:obj.getvalue() for obj,name in ((external,"external.csv"),(ledger_file,"outcome_ledger.csv")) if obj is not None}
                    publish_inputs(DATA_DIR,uploads,"decimal" if units_label.startswith("小數") else "percent_points",acknowledge_extremes=bool(ack))
                    st.success("已驗證單位並保存版本；舊快照不會改寫。")
                except Exception as exc:
                    st.error(safe_error_text(exc,300,(_finmind_token(),)))
        with st.expander("臨時休市公告登錄",expanded=False):
            st.caption("不提供強制認可舊行情。僅依官方公告登錄指定日期休市，並保留來源、公告時間與登錄時間。")
            closure_day=st.text_input("休市日期 YYYY-MM-DD",key="closure_day_v82")
            closure_url=st.text_input("TWSE / TPEx 官方公告 HTTPS 網址",key="closure_source_v82")
            closure_at=st.text_input("公告可用時間（臺北時間）",key="closure_at_v82")
            closure_confirmed=st.checkbox("我已閱讀並確認官方公告",key="closure_ok_v82")
            if st.button("登錄有來源的休市",key="save_closure_v82"):
                try:
                    save_closure_notice(DATA_DIR,closure_day,closure_url,closure_at,now=_taipei_timestamp(),confirmed=bool(closure_confirmed))
                    st.success("已登錄使用者核對的官方休市資訊；不延長原進場計畫。")
                except Exception as exc: st.error(safe_error_text(exc,300,(_finmind_token(),)))
        with st.expander("維護", expanded=False):
            if st.button("更新官方開休市日曆", key="calendar_refresh_v811"):
                ref = calendar_reference(DATA_DIR, now=_taipei_timestamp(), allow_fetch=True)
                st.info("日曆狀態：" + ref["status"])
                if ref.get("warnings"):
                    st.caption("更新有限制；日曆可能使用以前核對的年度排程。")
            if st.button("清除本機行情快取", key="clear_prices_v81"):
                store=DailyPriceStore(DATA_DIR / "daily_prices.sqlite")
                if store.clear():
                    st.success("已清除行情快取；舊快照仍保留。")
                else:
                    st.warning("快取清除未成功；不刪除原檔，請查看環境診斷。")
                    st.code(safe_error_text(store.last_cache_error or "Cache unavailable"),language=None)
            st.caption("雲端本機檔案可能因重部署消失。重要快照／帳本請自行備份。")

        with st.expander("環境與資料來源診斷",expanded=False):
            st.caption("只檢查本機套件與快取狀態，不是行情連線或收益驗證。")
            if st.button("檢查執行環境",key="check_runtime_v821"):
                store=DailyPriceStore(DATA_DIR / "daily_prices.sqlite")
                report=provider_runtime_status(store)
                st.json(report)
                st.download_button("下載環境診斷",json.dumps(report,ensure_ascii=False,indent=2).encode(),
                                   file_name="runtime_diagnostic.json",mime="application/json",key="download_runtime_v821")

    settings = service.RunSettings(reference_size=int(refs), candidate_size=int(count), history_period=period,
                                   model_family=FAMILY_LABELS[family_label], order_mode=order_mode, commission=commission/100.,
                                   sell_tax=tax/100., slippage=slip/100., notional=amount,
                                   min_ev_short=ev_short/100., min_ev_mid=ev_mid/100., min_ev_long=ev_long/100.)
    st.markdown("""<div class="topnote"><b>研究模式：</b>自動日線樣本是保守成交假設的探索性模擬，尚非完整歷史母池或樣本外績效驗證。
基本面、法人沒有資料就顯示缺少；不會用漲幅補成財報分數。</div>""", unsafe_allow_html=True)

    if st.button("更新市場與報酬研究", type="primary", use_container_width=True, key="run_scan_v81"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="建立日線快照／收益樣本：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            st.session_state["v8_snapshot"] = service.compact_dashboard(snap)
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
        except ScanBusyError:
            st.warning("另一個市場掃描正在進行，這次不重複建模。保留目前快照，完成後再讀取。")
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {safe_error_text(exc,300,(_finmind_token(),))}"
            st.error("本次更新未完成；不以空值覆蓋上一份成功快照。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())
    if calendar.get("operator_notice_count",0):
        st.caption(f"\u65e5\u66c6\u542b {calendar['operator_notice_count']} \u7b46\u4f7f\u7528\u8005\u6838\u5c0d\u7684\u5b98\u65b9\u516c\u544a\u767b\u9304\uff1b\u5167\u5bb9\u4e26\u975e\u7cfb\u7d71\u81ea\u52d5\u8a8d\u8b49\u3002")
    if st.session_state.get("v8_error"):
        with st.expander("查看本次失敗原因", expanded=True):
            st.code(st.session_state["v8_error"], language=None)
    if snap:
        st.markdown(f"""<div class="statusline">日線截至 <b>{esc(snap['price_date'])}</b> ·
研究基準 {esc(snap['market']['benchmark'])} · 收益期限 10／40／120交易日<br>
名冊 {snap['coverage']['requested']:,} → 同日有效 {snap['coverage']['feature_valid']:,} → 深度評估 {snap['candidate_n']:,}。
快照 {esc(snap['snapshot_id'][:12])}；不是盤中即時價。</div>""", unsafe_allow_html=True)
        fresh = daily_freshness(snap["price_date"],calendar)
        if not fresh["current_daily"]:
            if fresh["status"] == "UNKNOWN":
                st.warning("交易日曆無法確認；僅供研究，不將舊日線認定為可執行條件。")
            else:
                st.warning("保存快照不是最近完成日線，請更新。預期交易日：" + str(calendar.get("expected_date")))
        elif calendar.get("next_open_passed"):
            st.info("日線是最新已完成資料，但目前已進入下一交易時段；不能當作即時報價。")
        st.caption("新鮮度依證交所年度開休市排程檢查；颱風等臨時休市仍需核對公告。")
    else:
        st.info("第一次更新需要下載行情並建立歷史樣本，可能耗時數分鐘。之後可讀取同日快取，或使用套件的離線預計算指令。")
    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v81")
    if view == VIEW_LABELS[4]:
        st.subheader("個股診斷")
        st.caption("新買收益估計與既有持股規則分開。沒有原始失效點／理由，不假裝可以決定留或賣。")
        with st.form("doctor_form_v81"):
            code = st.text_input("股票代碼", value="2330", key="doctor_code_v81", placeholder="2330 或 6187.TWO")
            horizon_text = st.selectbox("研究週期", list(HORIZON_LABELS.values()), index=1, key="doctor_h_v81")
            own = st.checkbox("已有持股", key="doctor_own_v81")
            c1,c2 = st.columns(2)
            with c1:
                cost = st.number_input("持股成本（選填）", min_value=0., value=0., key="doctor_cost_v81")
                invalid = st.number_input("原始結構失效價（未知填0）", min_value=0., value=0., key="doctor_stop_v81")
            with c2:
                shares = st.number_input("股數（選填）", min_value=0, value=0, step=100, key="doctor_qty_v81")
                trail = st.number_input("既有獲利保護價（未知填0）", min_value=0., value=0., key="doctor_trail_v81")
            basis_date=st.text_input("原失效／保護價的設定日期（選填 YYYY-MM-DD）",key="basis_date_v82")
            basis_current=st.checkbox("我確認輸入價位已核對除權息並換算為目前每股價格",key="basis_current_v82")
            thesis = st.selectbox("原買進理由是否仍成立？", ["尚未確認","仍成立","已不成立"], key="doctor_thesis_v81")
            submitted = st.form_submit_button("立即診斷", type="primary", use_container_width=True)
        if submitted:
            st.session_state.pop("v8_doctor", None)
            try:
                if snap is None:
                    st.warning("先建立一次市場快照，診斷才有相同研究基準。上方更新按鈕不會被表單鎖住。")
                else:
                    with st.spinner("讀取相同快照的個股特徵與收益模型"):
                        dr = service.diagnose(code, snap, DATA_DIR)
                    dr["holding_input"] = {"own":bool(own),"cost":cost,"shares":int(shares),
                                           "stop":invalid or None,"trail":trail or None,"basis_date":basis_date,"basis_current":bool(basis_current),
                                           "broken":{"尚未確認":None,"仍成立":False,"已不成立":True}[thesis]}
                    if own:
                        try:
                            dr["references"]=service.holding_references(code,snap,DATA_DIR,basis_date or None,{"stop":invalid,"trail":trail})
                        except Exception as exc:
                            dr["references"]={"coordinate":{"can_use_original":False,"status":"REFERENCE_UNAVAILABLE"},"technical":{"reference":None},"error":type(exc).__name__}
                    dr["h"] = next(k for k,v in HORIZON_LABELS.items() if v == horizon_text)
                    st.session_state["v8_doctor"] = service.compact_doctor_result(dr)
            except Exception as exc:
                st.error(f"診斷未完成：{type(exc).__name__}: {exc}")
        dr = st.session_state.get("v8_doctor")
        if dr and snap and dr.get("snapshot_id") == snap["snapshot_id"]:
            obj = dr["stock"]
            comparisons = []
            for h in HORIZONS:
                f = obj["horizons"][h].get("forecast") or {}
                supported = f.get("sample_supported", False)
                comparisons.append({"週期":HORIZON_LABELS[h],
                                    "策略淨報酬研究估計":percent((f.get("strategy") or {}).get("mean")) if supported else "樣本不足／未估計",
                                    "相對基準":percent(f.get("alpha_mean")) if supported else "—"})
            st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True, key="doctor_compare_v81")
            st.caption("不同週期不直接互比冠軍，不年化短期估計。沒有新的Hold Edge百分位。")
            card(obj, dr["h"], snap, "doctor", dr.get("chart"), calendar=calendar)
            inp = dr["holding_input"]
            if inp["own"]:
                current_daily = daily_freshness(obj["price_date"],calendar)["current_daily"]
                prices_current = current_daily and not calendar.get("next_open_passed",True)
                refs=dr.get("references") or {}
                coords=refs.get("coordinate") or {}
                coordinate_ok=bool(inp.get("basis_current")) or coords.get("can_use_original") is True
                result = holding_review(obj["price"], original_invalidation=inp["stop"], trailing_protection=inp["trail"],
                                        thesis_broken=inp["broken"], prices_verified=prices_current and coordinate_ok)
                st.markdown("#### 既有持股原則檢查")
                st.info(HOLD_LABELS[result])
                if not coordinate_ok:
                    st.warning("原價位座標尚未確認，或期間有除權息／分割變動；不用舊原始價位直接判定賣出。")
                    if coords.get("mapped_references"):
                        st.write("還原比例映射參考（不自動改寫原計畫）：",coords["mapped_references"])
                technical=refs.get("technical") or {}
                if technical.get("reference") is not None:
                    st.caption(f"系統技術參考線 {money(technical['reference'])}：{technical.get('basis_session')} 已完成日線之2.5ATR，映射目前價格座標。非原設定停損、非自動賣出指令。")

                if inp["cost"]>0 and coords.get("status") == "ACTION_COORDINATE_CHANGE":
                    st.caption("\u671f\u9593\u6709\u50f9\u683c\u5ea7\u6a19\u8b8a\u52d5\uff0c\u4e0d\u4ee5\u820a\u6210\u672c\u76f4\u63a5\u8a08\u7b97\u640d\u76ca\uff1b\u8acb\u6838\u5c0d\u8abf\u6574\u5f8c\u80a1\u6578\u3001\u73fe\u91d1\u80a1\u5229\u8207\u5238\u5546\u5e33\u52d9\u3002")
                elif inp["cost"]>0:
                    pnl = obj["price"]/inp["cost"]-1
                    st.write(f"價格比率參考：{percent(pnl)}（未核對股利、股數變動與稅費，非實際總損益）；成本不參與留／賣規則。")
                if inp["shares"]>0 and inp["stop"] and coordinate_ok:
                    risk = max(0., (obj["price"]-inp["stop"])*inp["shares"])
                    st.caption(f"現價至原失效點的名目差額：約 {risk:,.0f} 元，非最大保證損失。")
    elif snap is None:
        st.info("尚無收益快照。請先更新；不會放入假股票、假行情或假報酬示範。")
    elif view == VIEW_LABELS[0]:
        st.subheader("各週期的合格代表")
        st.caption("每個週期最多1檔。不把10日和120日的收益混成一個總分；同股跨週期並非獨立證據。")
        any_pick = False
        for h in HORIZONS:
            picks = service.select_view(snap, h, True, 1)
            if picks:
                any_pick = True
                card(picks[0], h, snap, "prime", calendar=calendar)
            else:
                st.info(f"{HORIZON_LABELS[h]}：目前沒有通過研究條件的標的。")
        if not any_pick:
            st.caption("可至短／中／長頁查看有收益空間、但尚待進場或資料確認的攻擊觀察名單。")
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)
    if snap:
        diagnostics(snap)
    st.divider()
    st.caption("V8.2.1 報酬導向研究整合版｜本工具不自動下單、不提供即時行情，亦未完成全市場樣本外收益驗證。程式測試通過不等於策略會獲利。")


if __name__ == "__main__":
    main()
