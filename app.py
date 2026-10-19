from __future__ import annotations



from pathlib import Path

import pandas as pd

import numpy as np

import streamlit as st



try:

          import plotly.express as px

          PLOTLY = True

except ImportError:

          px = None

          PLOTLY = False



st.set_page_config(page_title="RetailPulse | Project Results", page_icon="📊", layout="wide", initial_sidebar_state="expanded")



ROOT = Path(__file__).resolve().parent

KNOWN_ROOT = Path(r"D:\Retail")

ANALYSIS_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "Revenue"]



st.markdown("""

<style>

:root{--ink:#172b4d;--muted:#637083;--blue:#2563eb;--teal:#0f9d91;--border:#e4eaf2;--soft:#f6f8fc}

html,body,[class*="css"]{font-family:Inter,"Segoe UI",sans-serif}

[data-testid="stAppViewContainer"],.stApp{background:#fff;color:var(--ink)}

[data-testid="stSidebar"]{background:#f6f8fc;border-right:1px solid #e4eaf2}

[data-testid="stSidebar"] *{color:#172b4d}

h1,h2,h3{color:#172b4d!important;letter-spacing:-.02em}

.rp-hero{padding:1.25rem 1.45rem;border:1px solid #e4eaf2;border-radius:16px;background:linear-gradient(120deg,#fff,#f5f9ff 65%,#f0fbf9);margin-bottom:1rem}

.rp-kicker{color:#2563eb;font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}

.rp-title{font-size:2.05rem;font-weight:850;line-height:1.15;margin:.3rem 0}

.rp-subtitle{color:#637083;font-size:.98rem;max-width:900px}

.rp-note{padding:.8rem 1rem;border:1px solid #e4eaf2;background:#f6f8fc;border-radius:10px;color:#46566c}

.rp-warn{padding:.8rem 1rem;border:1px solid #f3d28b;background:#fff8e8;border-radius:10px;color:#7c4a03}

.rp-ok{padding:.8rem 1rem;border:1px solid #b7e7d1;background:#ecfdf5;border-radius:10px;color:#065f46}

[data-testid="stMetric"]{border:1px solid #e4eaf2;border-radius:12px;padding:.8rem;background:white}

hr{border-color:#e4eaf2}

</style>

""", unsafe_allow_html=True)





def locate(rel: str):

          for base in (ROOT, KNOWN_ROOT, Path.cwd()):

                    p = base / rel

                    if p.is_file():

                              return p

          return None





@st.cache_data(show_spinner="Loading sales data…")

def load_sales(path: str):

          df = pd.read_csv(path, usecols=lambda c: c in ANALYSIS_COLUMNS, low_memory=False)

          absent = [c for c in ANALYSIS_COLUMNS if c not in df.columns]

          if absent:

                    raise ValueError(f"Missing required columns: {absent}")

          df = df[ANALYSIS_COLUMNS].copy()     # intentionally exclude the three helper flags

          source_rows = len(df)

          df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)

          for c in ["Quantity", "Price", "Revenue", "Customer ID"]:

                    df[c] = pd.to_numeric(df[c], errors="coerce")

          for c in ["Invoice", "StockCode", "Description", "Country"]:

                    df[c] = df[c].astype("string").str.strip()

          valid_id = df["Customer ID"].notna() & np.isfinite(df["Customer ID"])

          missing_id = int((~valid_id).sum())

          df = df.loc[valid_id].copy()

          df["Sales_Revenue"] = df["Quantity"] * df["Price"]

          df["Date"] = df["InvoiceDate"].dt.normalize()

          stats = {"source_rows": int(source_rows), "missing_customer_id_rows": missing_id,

                                  "retained_rows": int(len(df)), "pct_removed": 100 * missing_id / source_rows if source_rows else 0}

          return df, stats





@st.cache_data(show_spinner=False)

def read_csv_artifact(rel: str):

          p = locate(rel)

          if not p:

                    return None

          try:

                    return pd.read_csv(p, low_memory=False)

          except Exception:

                    return None





def hero(kicker, title, subtitle):

          st.markdown(f'<div class="rp-hero"><div class="rp-kicker">{kicker}</div><div class="rp-title">{title}</div><div class="rp-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)





def show_artifact_missing(label, rel):

          st.info(f"{label} is not available at \`{rel}\`. This section shows the documented result where possible; place the corresponding report in the project folder to enable its detailed table or chart.")





SALES_REL = "data/processed/retail_sales_final.csv"

source_path = locate(SALES_REL)

sales = None

sales_stats = {"source_rows": 0, "missing_customer_id_rows": 0, "retained_rows": 0, "pct_removed": 0}

load_error = None

if source_path:

          try:

                    sales, sales_stats = load_sales(str(source_path))

          except Exception as exc:

                    load_error = str(exc)



seg_labels = read_csv_artifact("data/processed/customer_segmentation_kmeans_labeled.csv")

seg_summary = read_csv_artifact("reports/segmentation/customer_segment_business_summary.csv")

churn_cv = read_csv_artifact("churn_experiment/churn_cv_model_comparison.csv")

# The current experiment results below are explicitly tied to the latest rebuilt experiment.

forecast_rev_cmp = read_csv_artifact("forecast_tradingday_experiment/tradingday_model_comparison.csv")

forecast_units_cmp = read_csv_artifact("forecast_units_experiment/daily_units_model_comparison.csv")

forecast_rev_cmp_alt = read_csv_artifact("forecast_tradingday_experiment/tradingday_model_comparison.csv")



st.sidebar.markdown("<div style='font-size:1.35rem;font-weight:850'>📊 RetailPulse</div><div style='color:#637083;font-size:.82rem'>Definitions, targets & measured results</div>", unsafe_allow_html=True)

PAGES = [

          "Project Overview",

          "Data Cleaning & EDA",

          "Customer Segmentation",

          "Churn Prediction",

          "Demand Forecasting",

          "Inventory Optimization",

          "Requirements & Deliverables",

]

page = st.sidebar.radio("Project sections", PAGES, index=0)

section_descriptions = {

          "Project Overview": "Definitions, project purpose, targets, and measured results.",

          "Data Cleaning & EDA": "How source data is filtered and checked, plus what the exploratory analysis tells us.",

          "Customer Segmentation": "Groups customers by purchase behaviour so marketing actions can be tailored.",

          "Churn Prediction": "Estimates which customers are likely to become inactive, with measured model results.",

          "Demand Forecasting": "Forecasts daily revenue and units; compares measured error with the 12% MAPE target.",

          "Inventory Optimization": "Uses demand patterns to explore reorder policies and shows simulation limitations.",

          "Requirements & Deliverables": "Maps each Zidio requirement to its target and the available measured evidence.",

}

st.sidebar.caption(section_descriptions[page])

st.sidebar.divider()

st.sidebar.caption("Source file: data/processed/retail_sales_final.csv")

st.sidebar.caption("The three helper flag columns are excluded from analysis. Missing Customer IDs are filtered in memory only; the master CSV is not modified.")



if load_error:

          st.error(f"Could not load sales data: {load_error}")



if page == "Project Overview":

          hero("AI-POWERED CUSTOMER ANALYTICS", "RetailPulse", "A retail analytics platform combining data quality checks, customer segmentation, churn-risk modelling, demand forecasting, and inventory decision support.")

          st.markdown("### Definition")

          st.write("RetailPulse is an AI-powered retail analytics platform that turns historical transaction data into customer behaviour insights, churn-risk estimates, demand forecasts, and inventory decision support.")

          st.markdown("### What the platform does")

          st.write("It summarizes what sold, groups customers by behaviour, estimates which customers may become inactive, forecasts historical 30-day demand, and compares simulated replenishment policies.")

          st.markdown("### Targets and measured results")

          a,b,c,d = st.columns(4)

          a.metric("Customer segments", "7", help="Target: 6–8 meaningful groups")

          b.metric("Churn ROC-AUC", "0.8193", help="Target: at least 0.88")

          c.metric("Churn precision@top 20%", "91.04%", help="Target: at least 75%")

          d.metric("Revenue MAPE*", "26.62%", help="Target: at most 12%; calculated on positive-sales days")

          st.markdown("### How to use this website")

          st.markdown("- **Data Cleaning & EDA:** understand data coverage and cleaning rules.\n- **Customer Segmentation:** inspect the 7 business groups and their meaning.\n- **Churn Prediction:** read the churn definition, features, model, and measured scores.\n- **Demand Forecasting:** compare revenue and unit forecast errors against the target.\n- **Inventory Optimization:** understand policy simulations and their assumptions.\n- **Requirements & Deliverables:** check every Zidio acceptance target and what remains for submission.")

          st.caption("*MAPE is computed only on days with positive actual sales because ordinary MAPE is undefined when actual sales are zero. The underlying source series ends on 2011-12-09; forecasts are historical demonstrations, not current-day predictions.")



elif page == "Data Cleaning & EDA":

          hero("F-01 · DATA FOUNDATION", "Data Cleaning & EDA", "Shows what is loaded into the dashboard, the filtering rules, and the main quality findings. The source master file is never overwritten.")

          st.markdown("### What this section does")

          st.write("Data cleaning makes fields usable for analysis; exploratory data analysis (EDA) summarizes distributions, time patterns, missingness, duplicates, and unusual values before modelling.")

          if sales is not None:

                    a,b,c,d = st.columns(4)

                    a.metric("Rows in master CSV", f"{sales_stats['source_rows']:,}")

                    b.metric("Rows without usable Customer ID", f"{sales_stats['missing_customer_id_rows']:,}")

                    c.metric("Rows retained in dashboard", f"{sales_stats['retained_rows']:,}")

                    d.metric("Excluded from dashboard", f"{sales_stats['pct_removed']:.2f}%")

                    st.markdown("#### Cleaning rules applied by this website")

                    st.markdown("1. Load only the nine analytical columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country, Revenue.\n2. Exclude rows with missing or nonnumeric Customer IDs from this dashboard's loaded sales view.\n3. Parse invoice dates and numeric fields.\n4. For sales trend charts, use positive Quantity and positive Price; return/negative lines are not treated as ordinary sales.\n5. Do not load or use the three helper flag columns. Do not edit the source CSV.")

                    st.markdown("#### Sales overview")

                    valid = sales[sales["Date"].notna() & (sales["Quantity"] > 0) & (sales["Price"] > 0)].copy()

                    a,b,c,d = st.columns(4)

                    a.metric("Eligible transaction rows", f"{len(valid):,}")

                    b.metric("Known customers", f"{valid['Customer ID'].nunique():,}")

                    c.metric("Products / stock codes", f"{valid['StockCode'].nunique():,}")

                    d.metric("Countries", f"{valid['Country'].nunique():,}")

                    if not valid.empty:

                              monthly = valid.assign(Month=valid["Date"].dt.to_period("M").astype(str)).groupby("Month", as_index=False).agg(Revenue=("Sales_Revenue", "sum"), Units=("Quantity", "sum"))

                              if PLOTLY:

                                        fig = px.line(monthly, x="Month", y="Revenue", markers=True, title="Monthly eligible sales revenue")

                                        fig.update_layout(template="plotly_white", height=350, xaxis_title="Month", yaxis_title="Revenue (source currency units)")

                                        st.plotly_chart(fig, use_container_width=True)

                              else:

                                        st.line_chart(monthly.set_index("Month")["Revenue"])

                              st.caption("Revenue in this chart is calculated as Quantity × Price for eligible positive-quantity, positive-price rows. Currency is shown in source units; no currency conversion is applied.")

                    eda_path = locate("reports/eda/eda_findings.md")

                    if eda_path:

                              with st.expander("Open saved EDA findings"):

                                        st.markdown(eda_path.read_text(encoding="utf-8", errors="replace"))

          else:

                    show_artifact_missing("Master sales CSV", SALES_REL)

          st.markdown("#### Known data-quality findings from the documented EDA")

          st.markdown("- Earlier documented EDA found 6,520 exact duplicate rows across the nine analytical columns; these were not automatically deleted.\n- Negative quantities and zero prices were found and are not counted as ordinary positive sales in sales/forecasting calculations.\n- Some cleaned StockCode values may have had suffix letters stripped in an earlier processed version. The original workbook retains source codes; inventory work should use original StockCodes when SKU identity matters.")

          st.caption("Counts above are dynamically read from the master CSV currently found by this app. Earlier EDA counts may differ if the source file has since changed.")



elif page == "Customer Segmentation":

          hero("F-02 · CUSTOMER GROUPING", "Customer Segmentation", "Groups customers with similar purchase behaviour so each group can receive a more relevant retention or marketing action.")

          st.markdown("### Definition")

          st.write("Customer segmentation is the process of dividing customers into groups using behavioural features. This project uses RFM: Recency (days since last purchase), Frequency (number of purchases/invoices), and Monetary value (spend), with K-Means clustering. The Zidio target is 6–8 meaningful segments.")

          a,b,c = st.columns(3)

          a.metric("Segments produced", "7", help="Zidio target: 6–8 meaningful segments")

          b.metric("Customer population", "5,852", help="Population from the documented RFM/K-Means run")

          c.metric("Best tested silhouette", "0.254", help="Measures cluster separation; higher is generally better")

          st.markdown("### Business interpretation")

          known = [

                    ("Champions",474,"Recent buyers with high purchase frequency and monetary value.","Reward loyalty and encourage advocacy."),

                    ("Loyal High-Value Customers",956,"Repeat buyers with strong monetary value.","Protect the relationship with personalized benefits."),

                    ("Recent Potential Loyalists",785,"Recent customers showing repeat-purchase potential.","Encourage a second or next purchase."),

                    ("Recent Low-Frequency Customers",701,"Recent but infrequent buyers.","Use onboarding and product recommendations."),

                    ("At-Risk Customers",1169,"Customers with a longer gap since purchase, despite historical value.","Test win-back campaigns."),

                    ("Hibernating Customers",1209,"Long-inactive customers with limited frequency.","Use low-cost reactivation messages."),

                    ("Lost Low-Value Customers",558,"Long-inactive customers with low historical spend.","Avoid expensive incentives unless response justifies them."),

          ]

          seg_df = pd.DataFrame(known, columns=["Business segment","Customers","Meaning","Suggested action"])

          if seg_summary is not None and not seg_summary.empty:

                    st.caption("A saved segment-summary artifact is available in the project. The business labels below reflect the documented seven-segment run.")

                    st.dataframe(seg_summary, use_container_width=True, hide_index=True)

          else:

                    st.dataframe(seg_df, use_container_width=True, hide_index=True)

          if PLOTLY:

                    fig = px.bar(seg_df, x="Business segment", y="Customers", title="Customer count by business segment", color="Customers", color_continuous_scale=["#bfdbfe","#2563eb"])

                    fig.update_layout(template="plotly_white", height=380, xaxis_tickangle=-20, coloraxis_showscale=False)

                    st.plotly_chart(fig, use_container_width=True)

          st.markdown("### Important limitation")

          st.write("K-Means produced the seven interpretable groups. The tested DBSCAN configuration did not produce 6–8 useful segments, so it is not the selected business segmentation. Segment names are interpretations of cluster profiles, not ground-truth customer labels.")

          if seg_labels is not None:

                    st.download_button("Download saved customer segments", seg_labels.to_csv(index=False).encode("utf-8"), "retailpulse_customer_segments.csv", "text/csv")



elif page == "Churn Prediction":

          hero("F-04 · CUSTOMER RETENTION", "Churn Prediction", "Scores customers by likelihood of becoming inactive, helping a business prioritize retention outreach.")

          st.markdown("### What churn means in this project")

          st.write("Churn prediction estimates which customers are likely to stop purchasing within a defined timeframe. For this experiment, a customer is labelled churned/inactive if they made no eligible purchase during the final 90-day label window (2011-09-11 to 2011-12-09). This is a historical inactivity proxy, not confirmed cancellation.")

          a,b,c,d = st.columns(4)

          a.metric("Customers evaluated", "5,281")

          b.metric("Churn/inactivity label rate", "56.60%")

          c.metric("Selected model", "Gradient Boosting")

          d.metric("Mean CV ROC-AUC", "0.8051")

          st.markdown("### Target and measured results")

          r1,r2 = st.columns(2)

          with r1:

                    st.metric("Held-out test ROC-AUC", "0.8193", help="Target ≥ 0.88")

                    st.caption("Target: ROC-AUC ≥ 0.88. Measured held-out result: 0.8193.")

          with r2:

                    st.metric("Precision among top 20% risk scores", "91.04%", help="Target ≥ 75%")

                    st.caption("Target: precision@top 20% ≥ 75%. Measured result: 91.04%.")

          st.markdown("### What the metrics mean")

          st.markdown("- **ROC-AUC:** how well the model ranks an inactive customer above an active one across thresholds. 0.5 is roughly random ranking; 1.0 is perfect separation.\n- **Precision@top 20%:** among the 20% of customers assigned the highest risk, the proportion labelled inactive.\n- **Average precision:** summary of precision-recall performance; held-out result was 0.8476.")

          st.markdown("### Features used")

          st.write("Features were calculated only from purchases before the label window. Leading model-specific importance values were Purchases_Last_365d (0.3452), Revenue_Last_365d (0.2127), Recency_Days (0.2016), Line_Count (0.0805), and Max_Gap_Days (0.0292). Importance indicates predictive contribution in this fitted model; it does not prove causation.")

          st.markdown("### Evaluation caveat")

          st.write("The 0.8193 result belongs to the latest rebuilt 5,281-customer experiment and should not be combined with older runs as if they used the same population and split. The held-out split has been evaluated; avoid repeatedly tuning against it. The source data ends in 2011, so this is a historical demonstration rather than a live current-risk score.")

          exp_report = locate("churn_experiment/README_churn_experiment.md")

          if exp_report:

                    with st.expander("Read full churn experiment report"):

                              st.markdown(exp_report.read_text(encoding="utf-8", errors="replace"))



elif page == "Demand Forecasting":

          hero("F-03 · DEMAND FORECASTING", "Demand Forecasting", "Forecasts daily revenue and unit demand, then measures forecast errors against the required accuracy target.")

          st.markdown("### Definition")

          st.write("Demand forecasting estimates future sales using historical patterns. This experiment tests daily revenue (Quantity × Price) and daily units separately, with calendar, lag, rolling-window, and weekday-history features.")

          st.markdown("### Results against Zidio target")

          a,b,c = st.columns(3)

          a.metric("Required MAPE", "≤ 12%")

          b.metric("Best revenue MAPE*", "26.62%")

          c.metric("Best units MAPE*", "25.84%")

          r1,r2 = st.columns(2)

          with r1:

                    st.markdown("#### Revenue forecast")

                    st.metric("Mean WAPE", "27.81%")

                    st.caption("Target: MAPE ≤ 12%. Measured positive-day MAPE: 26.62%.")

                    st.caption("Best tested model: ExtraTrees. Mean WAPE measures total absolute error divided by total actual revenue across each fold.")

          with r2:

                    st.markdown("#### Units forecast")

                    st.metric("Mean WAPE", "25.54%")

                    st.caption("Target: MAPE ≤ 12%. Measured positive-day MAPE: 25.84%.")

                    st.caption("Best tested model: ExtraTrees. This predicts unit quantity, not revenue.")

          st.markdown("### What the metrics mean")

          st.markdown("- **MAPE:** average absolute percentage error; ordinary MAPE is undefined when actual sales are zero. Here it is calculated only on days with positive actual sales.\n- **WAPE:** total absolute error divided by total actual sales across all evaluated days; it includes zero-actual days.\n- **30-day horizon:** the requested prediction period contains 30 calendar days.")

          st.markdown("### Validation method and limits")

          st.write("Three rolling-origin validation windows of 30 days each were used. ExtraTrees was the best tested model for both revenue and units. Forcing Saturdays to zero did not improve results. The data ends on 2011-12-09, and these validation windows were used during iterative experiments; the results are exploratory and need a fresh final holdout or newer data before a definitive claim.")

          rev_path = locate("forecast_tradingday_experiment/tradingday_model_comparison.csv")

          unit_path = locate("forecast_units_experiment/daily_units_model_comparison.csv")

          if rev_path or unit_path:

                    with st.expander("View saved model comparison tables"):

                              if rev_path:

                                        st.markdown("**Revenue models**")

                                        st.dataframe(pd.read_csv(rev_path), use_container_width=True, hide_index=True)

                              if unit_path:

                                        st.markdown("**Unit models**")

                                        st.dataframe(pd.read_csv(unit_path), use_container_width=True, hide_index=True)

          st.caption("*MAPE shown for positive-sales days only; it is not directly comparable with WAPE. Forecast dates are historical because the source dataset ends in 2011.")



elif page == "Inventory Optimization":
    hero("F-05 · INVENTORY DECISIONS", "Inventory Optimization", "Uses historical product demand and ABC–XYZ categories to explore reorder policies and balance stock availability against inventory held.")

    st.markdown("### Definition")
    st.write("Inventory optimization aims to maintain enough stock to meet demand without holding unnecessary excess. ABC classifies products by contribution; XYZ groups them by demand variability. The Zidio target is a 25–40% reduction in overstock/understock. Operational recommendations require actual stock-on-hand and supplier lead-time data.")

    st.markdown("### Latest training-only simulation")
    st.caption("Evaluation period: 10 Nov 2011–9 Dec 2011 · Assumed supplier lead time: 7 days · Policies calculated using training data from 11 Sep–9 Nov 2011.")

    a, b, c = st.columns(3)
    a.metric("Original policy fill rate", "62.90%")
    b.metric("Diagnostic policy fill rate", "59.53%")
    c.metric("Test demand covered", "89.38%", help="The 3,094 SKUs seen in training account for 745,844 of 834,461 test units.")

    inventory_results = pd.DataFrame([
        {"Policy": "Training-only original safety stock", "Products": 3094, "Test demand units": 745844, "Fulfilled units": 469128, "Unmet demand units": 276716, "Fill rate": "62.90%", "Stockout SKU-days": 11151, "Ending on-hand units": 276592, "Units ordered": 745720},
        {"Policy": "Training-only diagnostic safety stock", "Products": 3094, "Test demand units": 745844, "Fulfilled units": 443989, "Unmet demand units": 301855, "Fill rate": "59.53%", "Stockout SKU-days": 11987, "Ending on-hand units": 231613, "Units ordered": 675602},
    ])
    st.dataframe(inventory_results, use_container_width=True, hide_index=True)

    if PLOTLY:
        fig = px.bar(
            inventory_results,
            x="Policy",
            y=["Fulfilled units", "Unmet demand units"],
            barmode="group",
            title="Fulfilled vs. unmet demand (training-only policies)",
            labels={"value": "Units", "variable": "Demand outcome"},
        )
        fig.update_layout(template="plotly_white", height=380, xaxis_title="Policy", yaxis_title="Units", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.write("The original safety-stock policy performed better in this simulation: its fill rate was 62.90% versus 59.53%, with 276,716 versus 301,855 unmet units. Both policies begin with zero on-hand inventory and treat unmet demand as lost sales; these assumptions affect the measured fill rates.")

    st.warning("Coverage limitation: 140 SKUs had no training-period history and contributed 88,617 units (10.62% of test demand). The simulation above evaluates only the 3,094 training SKUs, so its demand totals are 745,844 units rather than the full 834,461 units. The result is a historical simulation, not a measured real-world saving or an approved purchasing recommendation.")

    st.markdown("### Zidio target and operational limits")
    st.write("The 25–40% overstock/understock reduction target is not yet demonstrated by these results. The simulation compares unmet demand and stockout SKU-days, but actual inventory on hand, purchase orders, supplier lead times, carrying costs, and confirmed stockout records were unavailable.")

    inv_path = locate("reports/inventory/inventory_step14_training_only_simulation.csv")
    if inv_path:
        with st.expander("Open saved Step 14 simulation results"):
            st.dataframe(pd.read_csv(inv_path), use_container_width=True, hide_index=True)

    abc = read_csv_artifact("reports/inventory/inventory_abc_xyz_summary.csv")
    if abc is not None:
        st.markdown("#### ABC–XYZ summary")
        st.dataframe(abc, use_container_width=True, hide_index=True)

elif page == "Requirements & Deliverables":

          hero("ZIDIO PROJECT BRIEF", "Requirements & Deliverables", "Each requirement is paired with its target and the measured result or available evidence. Definitions and limitations are stated plainly without status labels.")

          st.markdown("### Main functional requirements")

          req = pd.DataFrame([

                    {"ID":"F-01","Requirement":"ETL and data quality","Definition":"Extract, transform, and load source data into a consistent analytical dataset, with repeatable quality checks.","Target from Zidio brief":"Repeatable cleaning and automated quality checks","Measured result / evidence":"The dashboard loads nine analytical columns, filters missing Customer IDs in memory, and excludes the three helper flags. Historical EDA recorded 6,520 exact duplicate rows; they were not automatically deleted. End-to-end automated ETL checks need to be demonstrated."},

                    {"ID":"F-02","Requirement":"Customer segmentation","Definition":"Group customers with similar purchase behaviour so business actions can be tailored to each group.","Target from Zidio brief":"6–8 meaningful segments with business interpretation","Measured result / evidence":"7 K-Means business segments across 5,852 customers; best tested silhouette score 0.254. DBSCAN did not produce 6–8 useful groups in the tested configuration."},

                    {"ID":"F-03","Requirement":"Demand forecasting","Definition":"Use historical demand patterns to estimate sales for a future period.","Target from Zidio brief":"Prophet + LSTM ensemble; 30-day forecast; MAPE ≤ 12%","Measured result / evidence":"A historical 30-day forecast was generated. Latest daily-model experiments report revenue positive-day MAPE 26.62% and units positive-day MAPE 25.84%; best tested ExtraTrees WAPE was 27.81% for revenue and 25.54% for units. Source data ends 2011-12-09."},

                    {"ID":"F-04a","Requirement":"Churn ROC-AUC","Definition":"ROC-AUC measures how well a model ranks inactive customers above active customers across classification thresholds.","Target from Zidio brief":"ROC-AUC ≥ 0.88","Measured result / evidence":"Latest rebuilt Gradient Boosting experiment: held-out ROC-AUC 0.8193 on 5,281 customers, using 31 numeric features and a final-90-day inactivity label."},

                    {"ID":"F-04b","Requirement":"Churn precision@top 20%","Definition":"Precision@top 20% is the share of customers labelled inactive among the 20% assigned the highest risk scores.","Target from Zidio brief":"Precision@top 20% ≥ 0.75","Measured result / evidence":"91.04% in the latest rebuilt held-out experiment. The held-out split has already been examined during experimentation, so a fresh independent evaluation is recommended for final confirmation."},

                    {"ID":"F-05","Requirement":"Inventory optimization","Definition":"Choose replenishment levels that balance product availability against excess stock and holding cost.","Target from Zidio brief":"25–40% reduction in overstock/understock","Measured result / evidence":"Latest training-only simulation (11 Nov–9 Dec 2011): original safety-stock policy fill rate 62.90% with 276,716 unmet units; diagnostic policy fill rate 59.53% with 301,855 unmet units. The simulation covers 3,094 training SKUs and excludes 140 test-only SKUs contributing 88,617 units. Zero opening stock and lost-sales assumptions apply; the 25–40% operational reduction target has not been demonstrated."},

                    {"ID":"F-06","Requirement":"Interactive dashboard","Definition":"Present analytics through a usable interface with charts, explanations, exports, and scenario exploration.","Target from Zidio brief":"Interactive dashboard with visual insights, exports, and what-if analysis","Measured result / evidence":"The local Streamlit app contains sectioned explanations, charts/tables, and downloads where artifacts are available. Public deployment and full what-if coverage require demonstration."},

          ])

          st.dataframe(req, use_container_width=True, hide_index=True)

          st.markdown("### Production and submission checklist")

          delivery = pd.DataFrame([

                    {"Deliverable / capability":"GitHub repository with README","Zidio expectation":"Source code, setup instructions, data/artifact notes, and limitations","Evidence to include":"Provide the public repository URL and confirm that the README covers installation, file paths, models, and known limitations."},

                    {"Deliverable / capability":"Public HTTPS demo","Zidio expectation":"A deployed web app reachable by the reviewer","Evidence to include":"Add the working public HTTPS URL after deployment and test it in a private/incognito browser."},

                    {"Deliverable / capability":"PDF report (10–18 A4 pages)","Zidio expectation":"Overview, architecture, timeline, technical results, deployment/operations, visuals, and reflection","Evidence to include":"Assemble the final report and verify page count, screenshots, results, and limitations."},

                    {"Deliverable / capability":"4–8 minute demo video","Zidio expectation":"Walkthrough of main features, methods, results, and limitations","Evidence to include":"Record the dashboard walkthrough and include the final video link or file."},

                    {"Deliverable / capability":"Docker / Kubernetes / CI/CD / monitoring","Zidio expectation":"Production-oriented deployment and operational plan described in the brief","Evidence to include":"Document and demonstrate implemented components; distinguish planned architecture from components actually running."},

          ])

          st.dataframe(delivery, use_container_width=True, hide_index=True)

          st.markdown("### How to read the results")

          st.write("Compare each measured result directly with the target in the same row. Where a value is simulated, historically evaluated, or not yet demonstrated, that limitation is written in the evidence column. Do not describe a local app as publicly deployed unless a working public URL has been verified.")



st.divider()

st.caption("RetailPulse · Results are evidence-based and limitations are shown alongside metrics. Source transaction data ends on 2011-12-09; model results are historical and should not be presented as current forecasts or live customer predictions.")

