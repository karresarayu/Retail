from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None

st.set_page_config(page_title="RetailPulse | Analytics", page_icon="📊", layout="wide")
ROOT = Path(__file__).resolve().parent
KNOWN_ROOT = Path(r"D:\Retail")
COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "Revenue"]

st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,'Segoe UI',sans-serif}
[data-testid="stAppViewContainer"],.stApp{background:#fff;color:#172b4d}
[data-testid="stSidebar"]{background:#f6f8fc;border-right:1px solid #e4eaf2}
h1,h2,h3{color:#172b4d!important}
.rp-hero{padding:1.2rem 1.4rem;border:1px solid #e4eaf2;border-radius:16px;background:linear-gradient(120deg,#fff,#f3f8ff,#effbf8);margin-bottom:1rem}
.rp-kicker{color:#2563eb;font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
.rp-title{font-size:2rem;font-weight:850;margin:.3rem 0}.rp-subtitle{color:#637083}
[data-testid="stMetric"]{border:1px solid #e4eaf2;border-radius:12px;padding:.8rem;background:white}
</style>""", unsafe_allow_html=True)


def locate(rel):
    for base in (ROOT, KNOWN_ROOT, Path.cwd()):
        p = base / rel
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner="Loading sales data…")
def load_sales(path):
    df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[COLS].copy()
    source_rows = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
    for c in ("Quantity", "Price", "Revenue", "Customer ID"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Invoice", "StockCode", "Description", "Country"):
        df[c] = df[c].astype("string").str.strip()
    bad_ids = int(df["Customer ID"].isna().sum())
    df = df[df["Customer ID"].notna()].copy()
    df["Date"] = df["InvoiceDate"].dt.normalize()
    df["Sales_Revenue"] = df["Quantity"] * df["Price"]
    return df, {"source_rows": source_rows, "bad_ids": bad_ids, "retained": len(df)}


@st.cache_data(show_spinner=False)
def artifact(rel):
    p = locate(rel)
    if not p:
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return None


def hero(kicker, title, subtitle):
    st.markdown(f'<div class="rp-hero"><div class="rp-kicker">{kicker}</div><div class="rp-title">{title}</div><div class="rp-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)


def table(df, name, limit=500):
    if df is None or df.empty:
        st.info(f"No data rows are available for {name} in this environment.")
        return
    st.dataframe(df.head(limit), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(limit, len(df)):,} of {len(df):,} rows.")
    st.download_button(f"Download {name} CSV", df.to_csv(index=False).encode("utf-8"), f"{name.lower().replace(' ', '_')}.csv", "text/csv", key=f"dl_{name.lower().replace(' ', '_')}")


def bar(df, x, y, title):
    if df is None or df.empty or x not in df or y not in df:
        st.info(f"Chart unavailable for {title}: required data is not available.")
        return
    if px:
        fig = px.bar(df, x=x, y=y, title=title)
        fig.update_layout(template="plotly_white", height=370)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x)[y])


def line(df, x, y, title):
    if df is None or df.empty or x not in df or y not in df:
        st.info(f"Chart unavailable for {title}: required data is not available.")
        return
    if px:
        fig = px.line(df, x=x, y=y, markers=True, title=title)
        fig.update_layout(template="plotly_white", height=370)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index(x)[y])


sales_path = locate("data/processed/retail_sales_final.csv")
sales = None
stats = {"source_rows": 0, "bad_ids": 0, "retained": 0}
load_error = None
if sales_path:
    try:
        sales, stats = load_sales(str(sales_path))
    except Exception as e:
        load_error = str(e)
if load_error:
    st.error(f"Sales data could not be loaded: {load_error}")
valid = None
rfm = None
if sales is not None:
    valid = sales[sales["Date"].notna() & (sales["Quantity"] > 0) & (sales["Price"] > 0)].copy()
    valid["Sales_Revenue"] = valid["Quantity"] * valid["Price"]
    if not valid.empty:
        rfm = valid.groupby("Customer ID", as_index=False).agg(
            Last_Purchase=("Date", "max"), Invoices=("Invoice", "nunique"),
            Lines=("Invoice", "size"), Monetary=("Sales_Revenue", "sum"),
            Units=("Quantity", "sum"), Country=("Country", "first"))
        rfm["Recency_Days"] = (valid["Date"].max() - rfm["Last_Purchase"]).dt.days

seg_file = artifact("data/processed/customer_segmentation_kmeans_labeled.csv")
churn_file = artifact("data/processed/customer_churn_risk.csv")
churn_importance = artifact("data/processed/churn_feature_importance.csv")
rev_models = artifact("forecast_tradingday_experiment/tradingday_model_comparison.csv")
unit_models = artifact("forecast_units_experiment/daily_units_model_comparison.csv")
forecast_test = artifact("data/processed/forecast_test_30day.csv")
forecast_future = artifact("data/processed/prophet_revenue_forecast_30_days.csv")
inventory_file = artifact("data/processed/inventory_final_recommendations.csv")
abc_file = artifact("reports/inventory/inventory_abc_xyz_summary.csv")

pages = ["Project Overview", "Data Cleaning & EDA", "Customer Segmentation", "Churn Prediction", "Demand Forecasting", "Inventory Optimization", "Requirements & Deliverables"]
descriptions = {
    "Project Overview": "Project purpose and headline results.",
    "Data Cleaning & EDA": "Data quality, sales trends, countries, and products.",
    "Customer Segmentation": "Segment summaries and customer membership.",
    "Churn Prediction": "Churn meaning, model metrics, at-risk customers, and actions.",
    "Demand Forecasting": "Forecast metrics, model comparisons, and actual/predicted charts.",
    "Inventory Optimization": "SKU recommendations, reorder indicators, and simulation results.",
    "Requirements & Deliverables": "Targets, evidence, and submission checklist.",
}
st.sidebar.markdown("## 📊 RetailPulse")
st.sidebar.caption("Customer analytics & demand forecasting")
page = st.sidebar.radio("Project sections", pages)
st.sidebar.caption(descriptions[page])
st.sidebar.divider()
st.sidebar.caption("Source: data/processed/retail_sales_final.csv")
if sales is None:
    st.warning("Master sales CSV was not found. The folder `data/` is excluded by your current .gitignore, so a cloud deployment will not receive these files through Git. Tables/charts needing customer-level data require the dataset or saved artifacts in the deployed environment.")

if page == "Project Overview":
    hero("AI-POWERED CUSTOMER ANALYTICS", "RetailPulse", "Retail analytics for data quality, customer segments, churn risk, demand forecasts, and inventory decisions.")
    st.markdown("### What this platform does")
    st.write("RetailPulse summarizes transaction data, groups customers by buying behaviour, highlights customers who may become inactive, forecasts demand, and provides inventory decision support.")
    a,b,c,d = st.columns(4)
    a.metric("Customer segments", "7")
    b.metric("Churn ROC-AUC", "0.8193")
    c.metric("Top-20% churn precision", "91.04%")
    d.metric("Revenue positive-day MAPE", "26.62%")
    summary = pd.DataFrame([
        {"Area":"Data cleaning & EDA","Purpose":"Check data quality and explore sales patterns","Evidence":"Master dataset and summary charts when data is available"},
        {"Area":"Customer segmentation","Purpose":"Group customers for tailored engagement","Evidence":"7 documented K-Means segments; silhouette 0.254"},
        {"Area":"Churn prediction","Purpose":"Prioritize customers who may become inactive","Evidence":"Gradient Boosting ROC-AUC 0.8193; precision@top 20% 91.04%"},
        {"Area":"Demand forecasting","Purpose":"Estimate future revenue and units","Evidence":"Revenue MAPE 26.62%; units MAPE 25.84%; target ≤12% not met"},
        {"Area":"Inventory optimization","Purpose":"Compare stock availability and unmet demand","Evidence":"Historical simulation; not proven real-world savings"},
    ])
    table(summary, "Project overview")
    if valid is not None and not valid.empty:
        monthly = valid.assign(Month=valid["Date"].dt.to_period("M").astype(str)).groupby("Month", as_index=False).agg(Revenue=("Sales_Revenue","sum"), Units=("Quantity","sum"))
        line(monthly, "Month", "Revenue", "Monthly sales revenue")
        table(monthly, "Monthly sales summary")

elif page == "Data Cleaning & EDA":
    hero("F-01 · DATA FOUNDATION", "Data Cleaning & EDA", "Understand data coverage, cleaning rules, transaction trends, and product/country performance.")
    st.markdown("### What is data cleaning and EDA?")
    st.write("Data cleaning makes fields consistent. Exploratory data analysis summarizes distributions, missing values, trends, and unusual values before modelling.")
    if sales is not None and valid is not None:
        a,b,c,d = st.columns(4)
        a.metric("Rows in CSV", f"{stats['source_rows']:,}")
        b.metric("Missing customer IDs", f"{stats['bad_ids']:,}")
        c.metric("Rows retained", f"{stats['retained']:,}")
        d.metric("Unique customers", f"{valid['Customer ID'].nunique():,}")
        quality = pd.DataFrame([
            {"Check":"Source rows", "Count":stats["source_rows"], "Explanation":"Rows in master CSV"},
            {"Check":"Rows without customer ID", "Count":stats["bad_ids"], "Explanation":"Not included in customer-level analysis"},
            {"Check":"Invalid invoice dates", "Count":int(sales["InvoiceDate"].isna().sum()), "Explanation":"Excluded from date-based charts"},
            {"Check":"Non-positive quantity", "Count":int((sales["Quantity"] <= 0).sum()), "Explanation":"Returns/adjustments, not ordinary sales"},
            {"Check":"Non-positive price", "Count":int((sales["Price"] <= 0).sum()), "Explanation":"Excluded from ordinary positive sales"},
        ])
        table(quality, "Data quality checks")
        monthly = valid.assign(Month=valid["Date"].dt.to_period("M").astype(str)).groupby("Month", as_index=False).agg(Revenue=("Sales_Revenue","sum"), Units=("Quantity","sum"), Invoices=("Invoice","nunique"))
        line(monthly, "Month", "Revenue", "Monthly revenue")
        line(monthly, "Month", "Units", "Monthly units")
        table(monthly, "Monthly sales table")
        countries = valid.groupby("Country", as_index=False).agg(Revenue=("Sales_Revenue","sum"), Units=("Quantity","sum"), Customers=("Customer ID","nunique")).sort_values("Revenue", ascending=False)
        bar(countries.head(15), "Country", "Revenue", "Top 15 countries by revenue")
        table(countries, "Country performance")
        products = valid.groupby(["StockCode","Description"], dropna=False, as_index=False).agg(Revenue=("Sales_Revenue","sum"), Units=("Quantity","sum"), Invoices=("Invoice","nunique")).sort_values("Revenue", ascending=False)
        bar(products.head(15), "StockCode", "Revenue", "Top 15 products by revenue")
        table(products.head(100), "Top product performance")
    else:
        table(pd.DataFrame([
            {"Step":1,"Rule":"Load the nine analytical columns","Purpose":"Consistent schema"},
            {"Step":2,"Rule":"Parse dates and numeric values","Purpose":"Enable time and numeric analysis"},
            {"Step":3,"Rule":"Filter missing customer IDs in memory","Purpose":"Reliable customer-level metrics"},
            {"Step":4,"Rule":"Exclude non-positive quantity/price from ordinary sales charts","Purpose":"Separate returns/adjustments"},
        ]), "Cleaning rules")

elif page == "Customer Segmentation":
    hero("F-02 · CUSTOMER GROUPING", "Customer Segmentation", "Review segment sizes, customer membership, and recommended business actions.")
    st.markdown("### Definition")
    st.write("RFM segmentation groups customers using Recency (days since last purchase), Frequency (purchase activity), and Monetary value (spend). The documented K-Means run produced seven interpretable groups.")
    a,b,c = st.columns(3); a.metric("Segments", "7"); b.metric("Documented customers", "5,852"); c.metric("Silhouette score", "0.254")
    seg = pd.DataFrame([
        ("Champions",474,"Recent, frequent, high-spend customers","Reward loyalty"),
        ("Loyal High-Value Customers",956,"Repeat buyers with strong spend","Personalized benefits"),
        ("Recent Potential Loyalists",785,"Recent customers with repeat-purchase potential","Encourage next purchase"),
        ("Recent Low-Frequency Customers",701,"Recent but infrequent buyers","Onboarding and recommendations"),
        ("At-Risk Customers",1169,"Longer purchase gap despite past value","Win-back campaigns"),
        ("Hibernating Customers",1209,"Long-inactive customers with limited frequency","Low-cost reactivation"),
        ("Lost Low-Value Customers",558,"Long-inactive and low historical spend","Avoid expensive incentives"),
    ], columns=["Business segment","Documented customers","Meaning","Suggested action"])
    table(seg, "Segment summary")
    bar(seg, "Business segment", "Documented customers", "Customers by documented segment")
    if seg_file is not None and not seg_file.empty:
        st.markdown("### Customer-level segment membership")
        table(seg_file, "Customer segment membership")
        cluster = next((c for c in seg_file.columns if any(k in c.lower() for k in ("cluster","segment","label"))), None)
        if cluster:
            counts = seg_file[cluster].astype(str).value_counts().rename_axis("Segment").reset_index(name="Customers")
            bar(counts, "Segment", "Customers", "Saved membership by cluster")
    elif rfm is not None:
        st.markdown("### Customer RFM table")
        st.caption("Saved cluster labels are unavailable; these are actual RFM statistics, not fabricated K-Means labels.")
        table(rfm.sort_values("Monetary", ascending=False), "Customer RFM")
    else:
        st.info("Customer membership needs the master sales CSV or saved segmentation artifact in this environment.")
    table(seg[["Business segment","Suggested action"]], "Segment action guide")

elif page == "Churn Prediction":
    hero("F-04 · CUSTOMER RETENTION", "Churn Prediction", "Understand churn, review model results, and identify customers for retention review.")
    st.markdown("### What is churn prediction?")
    st.write("Churn prediction estimates which customers may stop purchasing. In this historical experiment, a customer was labelled inactive if no eligible purchase occurred during 2011-09-11 to 2011-12-09. This is an inactivity proxy, not confirmed cancellation.")
    a,b,c,d = st.columns(4); a.metric("Customers evaluated", "5,281"); b.metric("Inactivity label rate", "56.60%"); c.metric("Model", "Gradient Boosting"); d.metric("Mean CV ROC-AUC", "0.8051")
    x,y = st.columns(2); x.metric("Held-out ROC-AUC", "0.8193", help="Target ≥ 0.88"); y.metric("Precision@top 20%", "91.04%", help="Target ≥ 75%")
    metrics = pd.DataFrame([
        {"Metric":"ROC-AUC","Result":"0.8193","Target":"≥ 0.88","Meaning":"How well inactive customers rank above active customers"},
        {"Metric":"Precision@top 20%","Result":"91.04%","Target":"≥ 75%","Meaning":"Inactive share among the highest-risk fifth"},
        {"Metric":"Average precision","Result":"0.8476","Target":"Report metric","Meaning":"Summary of precision-recall performance"},
        {"Metric":"Label window","Result":"2011-09-11 to 2011-12-09","Target":"90 days","Meaning":"No eligible purchase in the label window"},
    ])
    table(metrics, "Churn model metrics")
    st.markdown("### Customers who may become inactive")
    if churn_file is not None and not churn_file.empty:
        st.caption("This is the saved churn-risk artifact. Scores are historical and should not be treated as current live predictions.")
        table(churn_file, "Churn risk customers")
        score = next((c for c in churn_file.columns if any(k in c.lower() for k in ("risk","prob","score","churn")) and pd.api.types.is_numeric_dtype(churn_file[c])), None)
        customer = next((c for c in churn_file.columns if "customer" in c.lower() and "id" in c.lower()), None)
        if score:
            top = churn_file.sort_values(score, ascending=False).head(20).copy()
            xcol = customer or top.columns[0]
            top[xcol] = top[xcol].astype(str)
            bar(top, xcol, score, "Top 20 saved churn-risk scores")
    elif rfm is not None and not rfm.empty:
        risk = rfm.copy()
        recency = risk["Recency_Days"].clip(lower=0)
        risk["Risk_Score_Heuristic"] = (0.75 * recency / max(float(recency.max()), 1) + 0.25 * (1 - risk["Invoices"].clip(lower=1) / max(float(risk["Invoices"].max()), 1))).clip(0,1)
        risk["Risk band"] = pd.cut(risk["Risk_Score_Heuristic"], [-0.001,.35,.65,1.001], labels=["Lower","Medium","Higher"]).astype(str)
        risk = risk.sort_values("Risk_Score_Heuristic", ascending=False)
        risk["Heuristic score (%)"] = (risk["Risk_Score_Heuristic"] * 100).round(2)
        risk["Last_Purchase"] = risk["Last_Purchase"].dt.strftime("%Y-%m-%d")
        st.warning("Saved Gradient Boosting risk scores were not found. This list is a transparent recency/frequency heuristic, not the trained model's predictions or calibrated probabilities.")
        table(risk[["Customer ID","Country","Last_Purchase","Recency_Days","Invoices","Monetary","Heuristic score (%)","Risk band"]].head(100), "Heuristic churn review list")
        bands = risk["Risk band"].value_counts().rename_axis("Risk band").reset_index(name="Customers")
        bar(bands, "Risk band", "Customers", "Customers by heuristic risk band")
        bar(risk.head(20), "Customer ID", "Heuristic score (%)", "Top 20 heuristic scores")
    else:
        st.info("No customer-level risk list is available. Include the saved churn-risk CSV or master sales CSV to display customer members.")
    features = pd.DataFrame([
        {"Feature":"Purchases_Last_365d","Importance":0.3452,"Meaning":"Purchases in the previous 365 days"},
        {"Feature":"Revenue_Last_365d","Importance":0.2127,"Meaning":"Revenue in the previous 365 days"},
        {"Feature":"Recency_Days","Importance":0.2016,"Meaning":"Days since last purchase before label window"},
        {"Feature":"Line_Count","Importance":0.0805,"Meaning":"Historical transaction line count"},
        {"Feature":"Max_Gap_Days","Importance":0.0292,"Meaning":"Largest gap between purchases"},
    ])
    table(churn_importance if churn_importance is not None and not churn_importance.empty else features, "Churn feature importance")
    bar(features, "Feature", "Importance", "Documented churn feature importance")
    table(pd.DataFrame([
        {"Risk group":"Higher risk","Suggested action":"Review purchase history and consider personalized win-back"},
        {"Risk group":"Medium risk","Suggested action":"Send relevant recommendations or a reminder"},
        {"Risk group":"Lower risk","Suggested action":"Continue normal loyalty communications"},
    ]), "Churn retention actions")

elif page == "Demand Forecasting":
    hero("F-03 · DEMAND FORECASTING", "Demand Forecasting", "Compare forecast accuracy, inspect model comparison tables, and review forecast values.")
    st.markdown("### Definition")
    st.write("Demand forecasting estimates future revenue and unit sales using historical patterns, lag features, rolling summaries, and calendar effects.")
    a,b,c = st.columns(3); a.metric("Target MAPE", "≤ 12%"); b.metric("Revenue MAPE", "26.62%"); c.metric("Units MAPE", "25.84%")
    summary = pd.DataFrame([
        {"Measure":"Revenue positive-day MAPE","Result":"26.62%","Interpretation":"Target not met; calculated on positive actual revenue days"},
        {"Measure":"Units positive-day MAPE","Result":"25.84%","Interpretation":"Target not met; calculated on positive actual unit days"},
        {"Measure":"Revenue WAPE","Result":"27.81%","Interpretation":"Total absolute error / total actual revenue"},
        {"Measure":"Units WAPE","Result":"25.54%","Interpretation":"Total absolute error / total actual units"},
        {"Measure":"Forecast horizon","Result":"30 calendar days","Interpretation":"Historical series ends 2011-12-09"},
    ])
    table(summary, "Forecast metric summary")
    errors = pd.DataFrame([{"Metric":"Revenue MAPE","Percent":26.62},{"Metric":"Units MAPE","Percent":25.84},{"Metric":"Revenue WAPE","Percent":27.81},{"Metric":"Units WAPE","Percent":25.54},{"Metric":"MAPE target","Percent":12.0}])
    bar(errors, "Metric", "Percent", "Forecast errors compared with 12% MAPE target")
    for label, frame, name in [("Revenue model comparison", rev_models, "Revenue model comparison"),("Units model comparison", unit_models, "Units model comparison")]:
        st.markdown(f"### {label}")
        if frame is not None and not frame.empty:
            table(frame, name)
            model_col = next((c for c in frame.columns if any(k in c.lower() for k in ("model","method","algorithm"))), frame.columns[0])
            metric_col = next((c for c in frame.columns if any(k in c.lower() for k in ("wape","mape","mae","rmse")) and pd.api.types.is_numeric_dtype(frame[c])), None)
            if metric_col:
                bar(frame, model_col, metric_col, f"{label}: {metric_col}")
        else:
            st.info(f"Saved {label.lower()} artifact not found.")
    forecast = forecast_test if forecast_test is not None and not forecast_test.empty else forecast_future
    if forecast is not None and not forecast.empty:
        st.markdown("### Forecast detail table")
        table(forecast, "Forecast detail")
        date_col = next((c for c in forecast.columns if "date" in c.lower()), None)
        pred_col = next((c for c in forecast.columns if any(k in c.lower() for k in ("pred","forecast","yhat")) and pd.api.types.is_numeric_dtype(forecast[c])), None)
        actual_col = next((c for c in forecast.columns if "actual" in c.lower() and pd.api.types.is_numeric_dtype(forecast[c])), None)
        if date_col and pred_col:
            plot = forecast.copy(); plot[date_col] = pd.to_datetime(plot[date_col], errors="coerce")
            if actual_col and px:
                melted = plot.melt(id_vars=[date_col], value_vars=[actual_col,pred_col], var_name="Series", value_name="Value")
                fig = px.line(melted, x=date_col, y="Value", color="Series", markers=True, title="Actual versus predicted")
                fig.update_layout(template="plotly_white", height=370); st.plotly_chart(fig, use_container_width=True)
            else:
                line(plot, date_col, pred_col, "Forecast by date")
    table(pd.DataFrame([
        {"Term":"MAPE","Definition":"Average absolute percentage error; undefined for zero actual values"},
        {"Term":"WAPE","Definition":"Sum of absolute errors divided by sum of actual values"},
        {"Term":"Validation","Definition":"Rolling-origin historical evaluation"},
        {"Term":"Limitation","Definition":"Source ends 2011-12-09; not a current-day forecast"},
    ]), "Forecast definitions")

elif page == "Inventory Optimization":
    hero("F-05 · INVENTORY DECISIONS", "Inventory Optimization", "Review simulation results, SKU-level recommendations, and demand indicators.")
    st.markdown("### Definition")
    st.write("Inventory optimization balances product availability against excess stock. ABC classifies contribution; XYZ classifies demand variability. Real purchasing decisions require stock-on-hand, supplier lead times, and order data.")
    sim = pd.DataFrame([
        {"Policy":"Training-only original safety stock","Products":3094,"Test demand units":745844,"Fulfilled units":469128,"Unmet demand units":276716,"Fill rate (%)":62.90,"Stockout SKU-days":11151,"Ending on-hand units":276592,"Units ordered":745720},
        {"Policy":"Training-only diagnostic safety stock","Products":3094,"Test demand units":745844,"Fulfilled units":443989,"Unmet demand units":301855,"Fill rate (%)":59.53,"Stockout SKU-days":11987,"Ending on-hand units":231613,"Units ordered":675602},
    ])
    a,b,c = st.columns(3); a.metric("Original fill rate", "62.90%"); b.metric("Diagnostic fill rate", "59.53%"); c.metric("SKUs evaluated", "3,094")
    table(sim, "Inventory simulation results")
    if px:
        melt = sim.melt(id_vars="Policy", value_vars=["Fulfilled units","Unmet demand units"], var_name="Outcome", value_name="Units")
        fig = px.bar(melt, x="Policy", y="Units", color="Outcome", barmode="group", title="Fulfilled versus unmet demand")
        fig.update_layout(template="plotly_white", height=370); st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(sim.set_index("Policy")[["Fulfilled units","Unmet demand units"]])
    st.markdown("### SKU-level recommendations")
    if inventory_file is not None and not inventory_file.empty:
        table(inventory_file, "Inventory reorder recommendations")
        sku_col = next((c for c in inventory_file.columns if any(k in c.lower() for k in ("stockcode","sku","product"))), inventory_file.columns[0])
        qty_col = next((c for c in inventory_file.columns if any(k in c.lower() for k in ("reorder","order_qty","recommended","quantity")) and pd.api.types.is_numeric_dtype(inventory_file[c])), None)
        if qty_col:
            bar(inventory_file.sort_values(qty_col, ascending=False).head(20), sku_col, qty_col, "Top 20 reorder quantities")
    elif valid is not None and not valid.empty:
        sku = valid.groupby(["StockCode","Description"], dropna=False, as_index=False).agg(Units=("Quantity","sum"), Revenue=("Sales_Revenue","sum"), Active_Days=("Date","nunique"))
        sku["Average units/day"] = sku["Units"] / sku["Active_Days"].clip(lower=1)
        sku["Illustrative 7-day demand"] = np.ceil(sku["Average units/day"] * 7).astype(int)
        sku = sku.sort_values("Illustrative 7-day demand", ascending=False)
        st.warning("This is a historical 7-day demand indicator, not a stock-aware purchasing recommendation.")
        table(sku.head(200), "Illustrative SKU demand")
        bar(sku.head(20), "StockCode", "Illustrative 7-day demand", "Top 20 SKUs by illustrative 7-day demand")
    else:
        st.info("SKU recommendations need the saved inventory CSV or master sales dataset.")
    st.markdown("### ABC–XYZ explanation / summary")
    if abc_file is not None and not abc_file.empty:
        table(abc_file, "ABC XYZ summary")
    else:
        table(pd.DataFrame([
            {"Classification":"ABC","Meaning":"Ranks SKUs by contribution to sales/value"},
            {"Classification":"XYZ","Meaning":"Groups SKUs by demand variability"},
            {"Required operational data":"Stock-on-hand, lead time, open orders, carrying costs","Meaning":"Needed for real reorder decisions"},
        ]), "ABC XYZ explanation")
    st.warning("The historical simulation assumed seven-day lead time, zero opening stock, and lost sales for unmet demand. It excludes 140 test-only SKUs representing 88,617 units. The 25–40% operational reduction target has not been demonstrated as real-world savings.")

else:
    hero("ZIDIO PROJECT BRIEF", "Requirements & Deliverables", "Review each target alongside the measured result and the evidence still needed.")
    req = pd.DataFrame([
        {"ID":"F-01","Requirement":"ETL and data quality","Target":"Repeatable cleaning and automated checks","Evidence":"Nine analytical columns; missing Customer IDs filtered in memory; historical EDA recorded 6,520 exact duplicates."},
        {"ID":"F-02","Requirement":"Customer segmentation","Target":"6–8 meaningful segments","Evidence":"7 K-Means segments; 5,852 customers; silhouette 0.254."},
        {"ID":"F-03","Requirement":"Demand forecasting","Target":"30-day forecast; MAPE ≤ 12%","Evidence":"Revenue MAPE 26.62%; units MAPE 25.84%; target not met."},
        {"ID":"F-04a","Requirement":"Churn ROC-AUC","Target":"≥ 0.88","Evidence":"Held-out ROC-AUC 0.8193 on 5,281 customers."},
        {"ID":"F-04b","Requirement":"Churn precision@top 20%","Target":"≥ 75%","Evidence":"91.04% in documented held-out experiment."},
        {"ID":"F-05","Requirement":"Inventory optimization","Target":"25–40% reduction in overstock/understock","Evidence":"Historical simulation; original fill rate 62.90%; operational savings not proven."},
        {"ID":"F-06","Requirement":"Interactive dashboard","Target":"Charts, explanations, exports, what-if analysis","Evidence":"Sectioned dashboard with charts, tables, CSV exports when source files are available."},
    ])
    table(req, "Requirements and evidence")
    numeric = pd.DataFrame([
        {"Metric":"Churn ROC-AUC","Actual (%)":81.93,"Target (%)":88.0},
        {"Metric":"Churn precision@top 20%","Actual (%)":91.04,"Target (%)":75.0},
        {"Metric":"Revenue positive-day MAPE","Actual (%)":26.62,"Target (%)":12.0},
        {"Metric":"Units positive-day MAPE","Actual (%)":25.84,"Target (%)":12.0},
    ])
    table(numeric, "Numeric target comparison")
    bar(numeric, "Metric", "Actual (%)", "Measured values; compare with target table")
    table(pd.DataFrame([
        {"Deliverable":"GitHub repository and README","Evidence needed":"Setup steps, data/artifact notes, metrics, and limitations"},
        {"Deliverable":"Public HTTPS demo","Evidence needed":"Verify in private browser and compare against local results"},
        {"Deliverable":"Final report","Evidence needed":"Overview, architecture, results, charts, deployment notes, reflection"},
        {"Deliverable":"Demo video","Evidence needed":"Walk through features, metrics, and limitations"},
        {"Deliverable":"Deployment and operations","Evidence needed":"Clearly distinguish running components from planned components"},
    ]), "Submission checklist")

st.divider()
st.caption("RetailPulse · Source transaction data ends on 2011-12-09. Results are historical. Heuristic churn scores and illustrative inventory indicators are explicitly labelled and are not fitted-model predictions or approved purchasing advice.")
