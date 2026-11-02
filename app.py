import html as _html
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="CartWhisperer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODEL
# ============================================================
MODEL_PATH = Path(__file__).resolve().parent / "model" / "cartwhisperer_model.pkl"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_names"], bundle.get("discount_model")


# ============================================================
# DESIGN TOKENS
# ============================================================
AC = "#3ecf8e"
WR = "#e5a74a"
DG = "#e5604d"
TX = "#e6e8eb"
MU = "#8a919c"
LINE = "#1d222b"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --bg:#0a0c0f; --s1:#0f1216; --s2:#141820;
  --line:#1d222b; --line2:#2a303b;
  --tx:#e6e8eb; --mu:#8a919c; --dim:#5c636e;
  --ac:#3ecf8e; --wr:#e5a74a; --dg:#e5604d;
}

html, body, .stApp { background: var(--bg); color: var(--tx); font-family: 'Inter', sans-serif; }
.stApp p, .stApp label, .stApp span { font-family: 'Inter', sans-serif; }

[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 1320px; padding: 1.4rem 2.2rem 3rem 2.2rem; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--s1); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding: 1.4rem 1.2rem; }
[data-testid="stSidebar"] label p { color: var(--mu) !important; font-size: 12px !important; font-weight: 500; }
[data-testid="stSidebar"] hr { border-color: var(--line); margin: 1rem 0; }

.brand { display:flex; align-items:center; gap:10px; margin-bottom: 6px; }
.brand-mark { width:26px; height:26px; border-radius:7px; border:1.5px solid var(--ac); display:flex; align-items:center; justify-content:center; color:var(--ac); font-size:12px; }
.brand-name { font-weight:700; font-size:17px; letter-spacing:-.3px; color:#fff; }
.brand-sub { color:var(--dim); font-size:11px; letter-spacing:.14em; text-transform:uppercase; margin-left:36px; }

.side-h { color: var(--dim); font-size: 10.5px; font-weight: 600; letter-spacing: .16em; text-transform: uppercase; margin: 4px 0 10px 0; }

/* ---------- inputs ---------- */
.stNumberInput input {
  background: var(--s2) !important; border: 1px solid var(--line2) !important; color: var(--tx) !important;
  border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 14px !important;
}
.stNumberInput button { background: var(--s2) !important; border-color: var(--line2) !important; color: var(--mu) !important; }
[data-baseweb="select"] > div { background: var(--s2) !important; border: 1px solid var(--line2) !important; border-radius: 8px !important; }
[data-baseweb="select"] * { color: var(--tx) !important; font-size: 13px !important; }
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
  font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
}

/* ---------- top bar ---------- */
.topbar { display:flex; justify-content:space-between; align-items:center; padding-bottom:16px; border-bottom:1px solid var(--line); margin-bottom: 26px; }
.crumb { color: var(--mu); font-size: 12.5px; }
.crumb b { color: var(--tx); font-weight:600; }
.pills { display:flex; gap:8px; }
.pill { display:inline-flex; align-items:center; gap:7px; border:1px solid var(--line2); background: var(--s1); color: var(--mu); font-size: 11.5px; padding: 5px 11px; border-radius: 999px; }
.dot { width:6px; height:6px; border-radius:50%; background: var(--wr); display:inline-block; }
.dot.g { background: var(--ac); }

.eyebrow { color: var(--ac); font-size: 11px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; }
.title { font-size: 34px; font-weight: 700; letter-spacing: -.9px; color:#fff; margin: 8px 0 6px 0; line-height:1.1; }
.lead { color: var(--mu); font-size: 14px; max-width: 640px; line-height: 1.65; margin-bottom: 4px; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background: transparent; padding: 10px 0; height: auto; color: var(--mu); font-weight: 500; font-size: 13.5px; }
.stTabs [aria-selected="true"] { color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--ac) !important; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display:none; }

/* ---------- cards ---------- */
.card { background: var(--s1); border:1px solid var(--line); border-radius: 14px; padding: 22px 24px; }
.card-h { color: var(--mu); font-size: 11px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; margin-bottom: 14px; }

.decision { background: var(--s1); border:1px solid var(--line); border-radius: 14px; padding: 26px 28px; position: relative; overflow:hidden; height: 100%; }
.decision:before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--c); }
.d-lbl { color: var(--mu); font-size: 11px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }
.d-big { font-family:'JetBrains Mono',monospace; font-size: 84px; font-weight: 500; letter-spacing:-4px; color:#fff; line-height: 1; margin: 16px 0 8px 0; }
.d-big span { font-size: 34px; color: var(--c); letter-spacing: 0; margin-left: 4px; }
.d-verdict { font-size: 17px; font-weight: 600; color: var(--c); margin-bottom: 10px; }
.d-why { color: #b4bac4; font-size: 13.5px; line-height: 1.7; max-width: 500px; margin: 0 0 18px 0; }
.chip { display:inline-block; font-family:'JetBrains Mono',monospace; font-size: 11px; color: var(--mu); border:1px solid var(--line2); padding: 4px 9px; border-radius: 6px; margin: 0 6px 6px 0; }

.gauge { position:relative; height:6px; background: var(--line); border-radius: 6px; margin: 4px 0 8px 0; max-width: 520px; }
.gauge > i { position:absolute; left:0; top:0; bottom:0; border-radius:6px; background: var(--c); }
.gauge > b { position:absolute; top:-5px; width:2px; height:16px; background: var(--mu); }
.gauge-l { display:flex; justify-content:space-between; max-width:520px; color: var(--dim); font-size: 10.5px; font-family:'JetBrains Mono',monospace; margin-bottom: 18px; }

/* ---------- ledger ---------- */
.ledger { width:100%; border-collapse: collapse; }
.ledger td { padding: 11px 0; border-bottom: 1px solid var(--line); font-size: 13.5px; color: var(--mu); }
.ledger tr:last-child td { border-bottom: 0; }
.ledger td.v { text-align:right; font-family:'JetBrains Mono',monospace; color: var(--tx); font-size: 13.5px; }
.ledger tr.total td { color:#fff; font-weight:600; padding-top: 16px; }
.ledger tr.total td.v { font-size: 22px; color: var(--ac); }

/* ---------- KPI ---------- */
.kpi { background: var(--s1); border:1px solid var(--line); border-radius: 14px; padding: 18px 20px; }
.k-l { color: var(--mu); font-size: 11.5px; font-weight: 500; }
.k-v { font-family:'JetBrains Mono',monospace; font-size: 26px; color:#fff; margin: 8px 0 12px 0; font-weight:500; letter-spacing:-.5px; }
.k-v small { font-size: 12px; color: var(--dim); margin-left: 4px; letter-spacing: 0; }
.bar { position:relative; height:4px; background: var(--line); border-radius: 4px; }
.bar > i { position:absolute; left:0; top:0; bottom:0; border-radius:4px; background: var(--c); }
.bar > b { position:absolute; top:-3px; width:1.5px; height:10px; background: var(--dim); }
.k-f { color: var(--dim); font-size: 11px; margin-top: 9px; }

/* ---------- notes ---------- */
.note { border:1px solid var(--line2); border-left: 3px solid var(--wr); background: var(--s1); border-radius: 10px; padding: 14px 18px; color: #b4bac4; font-size: 12.5px; line-height: 1.7; margin-top: 18px; }
.note b { color: var(--tx); }

.feat { display:flex; flex-wrap:wrap; gap:6px; }
.feat span { font-family:'JetBrains Mono',monospace; font-size: 11px; color: var(--mu); border:1px solid var(--line2); padding: 4px 9px; border-radius: 6px; background: var(--s2); }

.errbox { border:1px solid rgba(229,96,77,.45); border-left:3px solid var(--dg); background: var(--s1); border-radius: 12px; padding: 20px 24px; max-width: 760px; margin-top: 30px; }
.errbox h3 { color:#fff; margin: 0 0 8px 0; font-size: 18px; }
.errbox p { color: #b4bac4; font-size: 13.5px; line-height: 1.7; margin: 0; }
.errbox code { font-family:'JetBrains Mono',monospace; color: var(--wr); font-size: 12.5px; }

div.stDownloadButton > button {
  background: transparent; color: var(--tx); border: 1px solid var(--line2); border-radius: 9px;
  font-weight: 500; font-size: 13px; padding: 9px 18px; transition: .15s ease;
}
div.stDownloadButton > button:hover { border-color: var(--ac); color: var(--ac); background: rgba(62,207,142,.05); }

.foot { color: var(--dim); font-size: 11.5px; text-align:center; padding: 28px 0 0 0; border-top: 1px solid var(--line); margin-top: 36px; }
</style>
"""


def html(s: str):
    """Render HTML safely (collapses indentation so Markdown never treats it as a code block)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def esc(x) -> str:
    return _html.escape(str(x))


def usd(x: float) -> str:
    return f"&#36;{x:,.2f}"


def meter(label, value_html, fill_pct, color, foot, ticks=()):
    tick_html = "".join(f'<b style="left:{t}%"></b>' for t in ticks)
    return f"""
    <div class="kpi" style="--c:{color}">
      <div class="k-l">{label}</div>
      <div class="k-v">{value_html}</div>
      <div class="bar"><i style="width:{max(0, min(100, fill_pct)):.0f}%"></i>{tick_html}</div>
      <div class="k-f">{foot}</div>
    </div>"""


def style_fig(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MU, size=12),
        margin=dict(l=4, r=4, t=10, b=4),
        showlegend=False,
        hoverlabel=dict(bgcolor="#141820", bordercolor="#2a303b", font=dict(family="Inter", color=TX)),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(family="JetBrains Mono", size=11))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(family="JetBrains Mono", size=11))
    return fig


html(CSS)

# ============================================================
# LOAD MODEL (fail gracefully, styled)
# ============================================================
try:
    model, feature_names, discount_model = load_model()
except Exception as e:
    html(f"""
    <div class="errbox">
      <h3>Model could not be loaded</h3>
      <p>Expected the file at <code>model/cartwhisperer_model.pkl</code> next to <code>app.py</code>.<br><br>
      Details: {esc(e)}</p>
    </div>
    """)
    st.stop()

POST_DECISION = ["Discount_Percentage", "Discount_Value_USD", "Net_Price_USD",
                 "Post_Discount_Profit_USD", "Converted_Is_Purchased"]
leaky_features = [f for f in feature_names if f in POST_DECISION]

# A model trained with post-decision columns cannot make real decisions (it only
# recognises "discount > 0"). In that case the app runs on transparent rules.
RULE_MODE = bool(leaky_features)
RULE_THRESHOLD = 0.35
ENGINE = "The rules engine" if RULE_MODE else "The trained model"
SCORE_LABEL = "Coupon need score" if RULE_MODE else "Offer probability"
SCORE_WORD = "coupon need score" if RULE_MODE else "model offer probability"

CATEGORIES = ["Apparel", "Beauty", "Books", "Electronics", "Fitness", "Home & Kitchen"]
DEVICES = ["Desktop", "Mobile App", "Mobile Web"]
LOCATIONS = ["Asia-Pacific", "Europe", "Latin America", "Middle East", "North America"]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    html("""
    <div class="brand"><div class="brand-mark">◆</div><div class="brand-name">CartWhisper</div></div>
    <div class="brand-sub">Offer intelligence</div>
    """)
    st.divider()

    html('<div class="side-h">Product</div>')
    price = st.number_input("Product price (USD)", min_value=1.0, value=100.0, step=5.0)
    margin_pct = st.slider("Product margin (%)", 0, 100, 35)
    category = st.selectbox("Category", CATEGORIES)

    st.divider()
    html('<div class="side-h">Cart behaviour</div>')
    cart_age = st.number_input("Cart age (hours)", min_value=0, value=24, step=1)
    session = st.number_input("Session duration (min)", min_value=0, value=6, step=1)
    pages = st.number_input("Pages visited", min_value=0, value=5, step=1)

    st.divider()
    html('<div class="side-h">Customer</div>')
    purchases = st.number_input("Past purchases", min_value=0, value=3, step=1)
    coupon_reliance = st.slider("Historical coupon reliance (%)", 0, 100, 30, 5)
    payday = st.selectbox("Payday weekend", ["No", "Yes"])

    st.divider()
    html('<div class="side-h">Context</div>')
    device = st.selectbox("Device", DEVICES)
    location = st.selectbox("Customer location", LOCATIONS)

    st.divider()
    html('<div class="side-h">Offer strategy</div>')
    max_disc = st.slider("Maximum discount (%)", 5, 40, 20)
    keep_margin = st.slider("Margin to protect (%)", 0, 90, 50)

    st.divider()
    html('<div class="side-h" style="margin-bottom:0">Minor project · v3.1</div>')

# ============================================================
# MODEL INPUT BUILDER  (same logic as the original app)
# ============================================================
margin_usd = price * margin_pct / 100


def build_row(**over):
    p = dict(
        price=price, margin_pct=margin_pct, session=session, pages=pages, cart_age=cart_age,
        purchases=purchases, coupon_reliance=coupon_reliance, payday=payday,
        category=category, device=device, location=location,
    )
    p.update(over)
    m_usd = p["price"] * p["margin_pct"] / 100

    # Pre-coupon baseline values for the post-decision columns the model expects.
    values = {
        "Base_Price_USD": p["price"],
        "Product_Margin_Pct": p["margin_pct"],
        "Absolute_Margin_USD": m_usd,
        "Session_Duration_Mins": p["session"],
        "Pages_Visited": p["pages"],
        "Cart_Staleness_Hours": p["cart_age"],
        "Past_Purchase_Count": p["purchases"],
        "Historical_Coupon_Reliance_Ratio": p["coupon_reliance"],
        "Is_Payday_Weekend": int(p["payday"] == "Yes"),
        "Discount_Percentage": 0,
        "Discount_Value_USD": 0,
        "Net_Price_USD": p["price"],
        "Post_Discount_Profit_USD": m_usd,
        "Converted_Is_Purchased": 0,
    }
    cats = {
        "Product_Category": p["category"],
        "Device_Type": p["device"],
        "User_Location": p["location"],
    }
    for column, selected in cats.items():
        prefix = column + "_"
        for feature in feature_names:
            if feature.startswith(prefix):
                values[feature] = int(feature[len(prefix):] == selected)

    return [values.get(f, 0) for f in feature_names]


def score_parts(age=None, rel=None, sess=None, pg=None):
    """Contribution (0..1 each) of cart age, coupon reliance and low engagement."""
    age = cart_age if age is None else age
    rel = coupon_reliance if rel is None else rel
    sess = session if sess is None else sess
    pg = pages if pg is None else pg
    eng = min(100, sess * 2 + pg * 2)
    return (min(age / 120, 1) * 0.50, rel / 100 * 0.20, (1 - eng / 100) * 0.30)


def rule_score(age=None, rel=None, sess=None, pg=None):
    """0..1 'coupon need' score from cart age, coupon reliance and engagement."""
    return round(sum(score_parts(age, rel, sess, pg)), 4)


def predict_many(rows):
    df = pd.DataFrame(rows, columns=feature_names)
    return model.predict_proba(df)[:, 1]


# ============================================================
# PREDICTION (live)
# ============================================================
prediction, probability, pred_error = None, None, None
try:
    if RULE_MODE:
        base_df = None
        probability = rule_score()
        prediction = int(probability >= RULE_THRESHOLD)
    else:
        base_df = pd.DataFrame([build_row()], columns=feature_names)
        prediction = int(model.predict(base_df)[0])
        probability = float(model.predict_proba(base_df)[0][1])
except Exception as e:
    pred_error = str(e)

# ------------------------------------------------------------
# DISCOUNT SIZING: the model (or the rules engine) decides IF a coupon is
# offered, transparent rules decide HOW MUCH, capped to protect margin.
# ------------------------------------------------------------
MIN_DISCOUNT = 5
MARGIN_CAP = int(margin_pct * (100 - keep_margin) / 100)


def size_discount(score):
    """Map a 0..1 coupon-need score to a discount %, capped to protect margin."""
    frac = max(0.0, min(1.0, (score - RULE_THRESHOLD) / (1 - RULE_THRESHOLD)))
    raw = int(round(MIN_DISCOUNT + (max_disc - MIN_DISCOUNT) * frac))
    return min(raw, MARGIN_CAP), raw, raw > MARGIN_CAP


discount_pct, discount_steps, guard_hit = 0, [], False
need_score = rule_score()
if pred_error is None and prediction == 1:
    if discount_model is not None:
        raw = float(discount_model.predict(base_df)[0])
        d = max(1, min(int(round(raw)), max_disc))
        discount_steps.append(("Discount predicted by the trained model", f"{raw:.1f}%"))
        if d > MARGIN_CAP:
            d, guard_hit = MARGIN_CAP, True
    else:
        pa, pr_, pe = score_parts()
        d, raw, guard_hit = size_discount(need_score)
        discount_steps += [
            ("Cart idle time", f"{pa * 100:.0f} pts"),
            ("Coupon reliance", f"{pr_ * 100:.0f} pts"),
            ("Low engagement", f"{pe * 100:.0f} pts"),
            ("Coupon need score", f"{need_score * 100:.0f}%"),
            (f"Mapped to the {MIN_DISCOUNT}-{max_disc}% range", f"{raw}%"),
        ]
    discount_pct = d

discount_amt = price * discount_pct / 100
final_price = price - discount_amt
profit_after = margin_usd - discount_amt
retained = (profit_after / margin_usd * 100) if margin_usd else 0

# ============================================================
# HEADER
# ============================================================
html(f"""
<div class="topbar">
  <div class="crumb">Workspace &nbsp;/&nbsp; <b>Coupon recommendation</b></div>
  <div class="pills">
    <span class="pill"><span class="dot {'' if RULE_MODE else 'g'}"></span>{'Rules engine · active' if RULE_MODE else 'Trained model · loaded'}</span>
    <span class="pill"><span class="dot g"></span>{'5 signals' if RULE_MODE else str(len(feature_names)) + ' features'}</span>
    <span class="pill"><span class="dot g"></span>Live prediction</span>
  </div>
</div>
<div class="eyebrow">Coupon strategy</div>
<div class="title">Should we offer a coupon?</div>
<div class="lead">Adjust the customer and cart signals in the sidebar. {'The rules engine' if RULE_MODE else 'The trained model'} re-scores
the scenario instantly.</div>
<div style="height:18px"></div>
""")

if pred_error:
    html(f"""
    <div class="errbox">
      <h3>Prediction failed</h3>
      <p>{esc(pred_error)}</p>
    </div>
    """)
    st.stop()

if leaky_features:
    html(f"""
    <div class="note" style="margin:0 0 20px 0"><b>Running on the rules engine.</b>
    The saved model file was trained with post-decision columns ({esc(", ".join(leaky_features))}), which already
    reveal whether a coupon was given, so it cannot make real decisions for a new customer. The coupon decision and
    discount below come from transparent rules instead. After retraining the model without those columns
    (<b>train_model.py</b>), this app switches to the ML model automatically.</div>
    """)

tab_rec, tab_sens, tab_model = st.tabs(["Recommendation", "Sensitivity", "Model &amp; notes"])

offer = prediction == 1
if offer and discount_pct > 0:
    state_color, verdict = AC, f"Give a {discount_pct}% coupon"
    why = (ENGINE + " recommends a coupon for this scenario. The size grows with cart idle time, coupon "
           "reliance and low engagement, and is capped so at least " + str(keep_margin) +
           "% of the unit margin is kept.")
elif offer:
    state_color, verdict = WR, "Coupon blocked by margin guard"
    why = (ENGINE + " recommends a coupon, but the product margin is too thin to give any "
           "discount while keeping " + str(keep_margin) + "% of it.")
else:
    state_color, verdict = MU, "No coupon recommended"
    why = (ENGINE + " does not recommend a coupon for this scenario, so the price "
           "stays unchanged.")

# ============================================================
# TAB 1: RECOMMENDATION
# ============================================================
with tab_rec:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1], gap="medium")

    with c1:
        html(f"""
        <div class="decision" style="--c:{state_color}">
          <div class="d-lbl">Recommended coupon</div>
          <div class="d-big">{discount_pct}<span>% OFF</span></div>
          <div class="d-verdict">{verdict}</div>
          <div class="gauge" style="--c:{state_color}"><i style="width:{probability * 100:.1f}%"></i><b style="left:{RULE_THRESHOLD * 100 if RULE_MODE else 50:.0f}%"></b></div>
          <div class="gauge-l"><span>{SCORE_WORD} {probability * 100:.1f}%</span><span>boundary {RULE_THRESHOLD * 100 if RULE_MODE else 50:.0f}%</span></div>
          <p class="d-why">{why}</p>
          <span class="chip">{esc(category)}</span>
          <span class="chip">{esc(device)}</span>
          <span class="chip">{esc(location)}</span>
          <span class="chip">cart {cart_age}h</span>
        </div>
        """)

    with c2:
        html(f"""
        <div class="card" style="height:100%">
          <div class="card-h">Price ledger</div>
          <table class="ledger">
            <tr><td>List price</td><td class="v">{usd(price)}</td></tr>
            <tr><td>Coupon ({discount_pct}%)</td><td class="v" style="color:{DG if discount_amt else TX}">{'−' if discount_amt else ''}{usd(discount_amt)}</td></tr>
            <tr><td>Unit margin before</td><td class="v">{usd(margin_usd)}</td></tr>
            <tr><td>Unit margin after</td><td class="v">{usd(profit_after)}</td></tr>
            <tr class="total"><td>Customer pays</td><td class="v">{usd(final_price)}</td></tr>
          </table>
        </div>
        """)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    p_color = AC if offer else MU
    age_color = WR if cart_age >= 48 else AC
    rel_color = WR if coupon_reliance >= 60 else AC
    mar_color = AC if margin_pct >= 30 else (WR if margin_pct >= 15 else DG)

    with k1:
        html(meter(SCORE_LABEL, f"{probability * 100:.0f}<small>%</small>", probability * 100, p_color,
                   f"Tick marks the {RULE_THRESHOLD * 100:.0f}% decision boundary" if RULE_MODE else "Tick marks the 50% boundary",
                   ticks=(RULE_THRESHOLD * 100,) if RULE_MODE else (50,)))
    with k2:
        html(meter("Cart age", f"{cart_age}<small>hours</small>", min(100, cart_age / 168 * 100), age_color,
                   "Scale runs 0 to 168 hours (one week)"))
    with k3:
        html(meter("Coupon reliance", f"{coupon_reliance}<small>%</small>", coupon_reliance, rel_color,
                   "Share of past orders that used a coupon"))
    with k4:
        html(meter("Margin retained", f"{retained:.0f}<small>%</small>", retained, AC if retained >= 70 else WR,
                   "Share of unit margin kept after coupon"))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if discount_steps:
        rows_html = "".join(f'<tr><td>{esc(a)}</td><td class="v">{esc(b)}</td></tr>' for a, b in discount_steps)
        cap_row = (f'<tr><td>Margin guard (protect {keep_margin}% of margin)</td><td class="v" style="color:#e5a74a">capped at {discount_pct}%</td></tr>'
                   if guard_hit else "")
        html(f"""
        <div class="card"><div class="card-h">How the {discount_pct}% was built</div>
        <table class="ledger">{rows_html}{cap_row}
        <tr class="total"><td>Final coupon</td><td class="v">{discount_pct}%</td></tr></table></div>
        <div style="height:16px"></div>
        """)

    if RULE_MODE:
        html(f"""
        <div class="note"><b>How to read the score.</b> The coupon need score (0 to 100%) combines cart idle time
        (up to 50 pts, full at 120h), coupon reliance (up to 20 pts) and low engagement (up to 30 pts). A score of
        {RULE_THRESHOLD * 100:.0f}% or more triggers a coupon, and the discount then grows from {MIN_DISCOUNT}% up to your
        maximum as the score rises. It is a rule-based score, not a machine-learning probability.</div>
        """)
    else:
        html("""
        <div class="note"><b>How to read the probability.</b> It is the model's estimate that a coupon would have been
        offered in the historical data for a similar scenario. It is not the chance that a coupon will cause a purchase.</div>
        """)

# ============================================================
# TAB 2: SENSITIVITY (real model what-ifs)
# ============================================================
with tab_sens:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    html("""
    <div class="lead" style="margin-bottom:14px">Each curve changes one signal and holds everything else at the
    sidebar values. The marker is your current scenario.</div>
    """)

    ages = list(range(0, 169, 6))
    rels = list(range(0, 101, 5))
    try:
        if RULE_MODE:
            age_probs = [rule_score(age=a) for a in ages]
            rel_probs = [rule_score(rel=r) for r in rels]
        else:
            age_probs = predict_many([build_row(cart_age=a) for a in ages])
            rel_probs = predict_many([build_row(coupon_reliance=r) for r in rels])
        sens_ok = True
    except Exception as e:
        sens_ok = False
        html(f'<div class="errbox"><h3>Sensitivity unavailable</h3><p>{esc(e)}</p></div>')

    if sens_ok:
        s1, s2 = st.columns(2, gap="medium")

        def curve(xs, ys, cur_x, cur_y, xtitle, xfmt=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=[v * 100 for v in ys], mode="lines",
                line=dict(color=AC, width=2.5, shape="spline", smoothing=0.5),
                fill="tozeroy", fillcolor="rgba(62,207,142,.07)",
                hovertemplate=xtitle + " %{x}<br>Probability %{y:.1f}%<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[cur_x], y=[cur_y * 100], mode="markers",
                marker=dict(size=12, color="#0a0c0f", line=dict(color=state_color, width=3)),
                hovertemplate="Current: %{x}<br>Probability %{y:.1f}%<extra></extra>",
            ))
            fig.add_hline(y=RULE_THRESHOLD * 100 if RULE_MODE else 50, line=dict(color="#2a303b", width=1, dash="dot"))
            fig.update_xaxes(title=xtitle)
            fig.update_yaxes(title=SCORE_LABEL + " (%)", range=[0, 100], ticksuffix="%")
            return style_fig(fig, 320)

        with s1:
            html(f'<div class="card" style="padding-bottom:6px"><div class="card-h">Cart age vs. {SCORE_LABEL.lower()}</div></div>')
            st.plotly_chart(curve(ages, age_probs, cart_age, probability, "Cart age (hours)"),
                            use_container_width=True, config={"displayModeBar": False})
        with s2:
            html(f'<div class="card" style="padding-bottom:6px"><div class="card-h">Coupon reliance vs. {SCORE_LABEL.lower()}</div></div>')
            st.plotly_chart(curve(rels, rel_probs, coupon_reliance, probability, "Coupon reliance (%)"),
                            use_container_width=True, config={"displayModeBar": False})

        # Recommended discount across cart age
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        thr_now = RULE_THRESHOLD if RULE_MODE else 0.5
        disc_curve = [size_discount(rule_score(age=x))[0] if pv >= thr_now else 0
                      for x, pv in zip(ages, age_probs)]
        html('<div class="card" style="padding-bottom:6px"><div class="card-h">Recommended discount vs. cart age</div></div>')
        dfig = go.Figure()
        dfig.add_trace(go.Scatter(
            x=ages, y=disc_curve, mode="lines",
            line=dict(color=AC, width=2.5, shape="hv"),
            fill="tozeroy", fillcolor="rgba(62,207,142,.07)",
            hovertemplate="Cart age %{x}h<br>Discount %{y}%<extra></extra>",
        ))
        dfig.add_trace(go.Scatter(
            x=[cart_age], y=[discount_pct], mode="markers",
            marker=dict(size=12, color="#0a0c0f", line=dict(color=state_color, width=3)),
            hovertemplate="Current: %{x}h<br>Discount %{y}%<extra></extra>",
        ))
        dfig.update_xaxes(title="Cart age (hours)")
        dfig.update_yaxes(title="Discount (%)", ticksuffix="%", range=[0, max(max_disc, 10) + 2])
        st.plotly_chart(style_fig(dfig, 280), use_container_width=True, config={"displayModeBar": False})

        # Context comparison across categories and devices
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        try:
            if RULE_MODE:
                raise RuntimeError("category and device do not affect the rules")
            cat_probs = predict_many([build_row(category=c) for c in CATEGORIES])
            dev_probs = predict_many([build_row(device=d) for d in DEVICES])

            def hbar(names, probs, current):
                fig = go.Figure(go.Bar(
                    y=names, x=[v * 100 for v in probs], orientation="h",
                    marker=dict(color=[AC if n == current else "#2a303b" for n in names]),
                    text=[f"{v * 100:.1f}%" for v in probs], textposition="outside",
                    textfont=dict(family="JetBrains Mono", color=TX, size=12),
                    hovertemplate="%{y}<br>Probability %{x:.1f}%<extra></extra>",
                ))
                fig.update_layout(bargap=0.5)
                fig.update_xaxes(range=[0, 115], ticksuffix="%")
                fig.update_yaxes(autorange="reversed", tickfont=dict(family="Inter", size=12))
                return style_fig(fig, 270)

            b1, b2 = st.columns(2, gap="medium")
            with b1:
                html('<div class="card" style="padding-bottom:6px"><div class="card-h">By product category</div></div>')
                st.plotly_chart(hbar(CATEGORIES, cat_probs, category),
                                use_container_width=True, config={"displayModeBar": False})
            with b2:
                html('<div class="card" style="padding-bottom:6px"><div class="card-h">By device</div></div>')
                st.plotly_chart(hbar(DEVICES, dev_probs, device),
                                use_container_width=True, config={"displayModeBar": False})
        except Exception:
            pass

# ============================================================
# TAB 3: MODEL & NOTES
# ============================================================
with tab_model:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="medium")

    with m1:
        html("""
        <div class="card" style="height:100%">
          <div class="card-h">Objective</div>
          <p style="color:#b4bac4;font-size:13.5px;line-height:1.75;margin:0 0 18px 0">
          CartWhisper is an academic project that uses cart, session and customer behaviour to decide
          whether a coupon should be offered and how large it should be.</p>
          <div class="card-h">Input groups</div>
          <table class="ledger">
            <tr><td>Price, margin, category</td><td class="v">product</td></tr>
            <tr><td>Cart age, session, pages</td><td class="v">behaviour</td></tr>
            <tr><td>Past purchases, coupon reliance, payday</td><td class="v">customer</td></tr>
            <tr><td>Device, location</td><td class="v">context</td></tr>
          </table>
        </div>
        """)

    with m2:
        chips = "".join(f"<span>{esc(f)}</span>" for f in feature_names)
        html(f"""
        <div class="card" style="height:100%">
          <div class="card-h">Model features ({len(feature_names)})</div>
          <div class="feat">{chips}</div>
        </div>
        """)

    html("""
    <div class="note"><b>Interpretation.</b> The model decides whether a coupon is offered. The discount size comes from the trained discount
    model when the model file includes one, otherwise from transparent rules (cart age, engagement, confidence);
    either way it is capped at half of the unit margin. The probability is the model's estimate of a historical
    coupon offer, not the chance that a coupon causes a purchase. Margin figures exclude other costs.</div>
    """)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    report = pd.DataFrame([{
        "Base_Price_USD": price,
        "Product_Margin_Pct": margin_pct,
        "Absolute_Margin_USD": round(margin_usd, 2),
        "Product_Category": category,
        "Cart_Staleness_Hours": cart_age,
        "Session_Duration_Mins": session,
        "Pages_Visited": pages,
        "Past_Purchase_Count": purchases,
        "Historical_Coupon_Reliance_Ratio": coupon_reliance,
        "Is_Payday_Weekend": int(payday == "Yes"),
        "Device_Type": device,
        "User_Location": location,
        "Coupon_Offer_Probability": round(probability, 4),
        "Model_Recommends_Coupon": int(offer),
        "Recommended_Discount_Pct": discount_pct,
        "Discount_Value_USD": round(discount_amt, 2),
        "Final_Price_USD": round(final_price, 2),
        "Unit_Margin_After_USD": round(profit_after, 2),
    }])
    st.download_button(
        "Download recommendation report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="cartwhisper_recommendation.csv",
        mime="text/csv",
    )

html('<div class="foot">CartWhisper · Academic minor project · Interactive demonstration</div>')