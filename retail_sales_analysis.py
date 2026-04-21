"""
Retail Sales Performance Dashboard
=====================================
Author: Ronak Parikh
Tools: Python (Pandas, Plotly), Excel-compatible CSV output
Dataset: Synthetic retail sales data (12 months, multiple stores & categories)
Goal: Analyze sales trends, top products, regional performance, and seasonality.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC RETAIL DATASET
# ─────────────────────────────────────────────
months = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
regions = ['Ontario', 'British Columbia', 'Alberta', 'Quebec', 'Manitoba']
categories = ['Electronics', 'Furniture', 'Clothing', 'Appliances', 'Sports']
stores = [f'Store_{i:03d}' for i in range(1, 21)]  # 20 stores

records = []
for month in months:
    for store in np.random.choice(stores, 15, replace=False):
        region = np.random.choice(regions)
        for category in categories:
            base_sales = {
                'Electronics': 45000, 'Furniture': 35000,
                'Clothing': 25000, 'Appliances': 30000, 'Sports': 20000
            }[category]
            seasonality = {
                1: 0.8, 2: 0.75, 3: 0.9, 4: 0.95, 5: 1.0, 6: 1.05,
                7: 1.1, 8: 1.05, 9: 0.95, 10: 1.0, 11: 1.3, 12: 1.5
            }[month.month]
            sales = base_sales * seasonality * np.random.uniform(0.85, 1.15)
            units = int(sales / np.random.uniform(50, 500))
            records.append({
                'date': month,
                'month': month.strftime('%b %Y'),
                'month_num': month.month,
                'store': store,
                'region': region,
                'category': category,
                'sales': round(sales, 2),
                'units_sold': units,
                'transactions': np.random.randint(80, 500),
                'returns': int(units * np.random.uniform(0.02, 0.08)),
                'discount_pct': round(np.random.uniform(0, 25), 1)
            })

df = pd.DataFrame(records)
df['net_sales'] = df['sales'] * (1 - df['discount_pct'] / 100)
df['return_rate'] = (df['returns'] / df['units_sold'] * 100).round(2)
df['avg_transaction_value'] = (df['net_sales'] / df['transactions']).round(2)

print("=" * 60)
print("RETAIL SALES PERFORMANCE ANALYSIS – 2023")
print("=" * 60)
print(f"\n📦 Dataset: {len(df):,} records | {df['store'].nunique()} Stores | 5 Categories | 5 Regions")
print(f"💰 Total Revenue: ${df['net_sales'].sum():,.0f}")
print(f"📦 Total Units Sold: {df['units_sold'].sum():,}")
print(f"🔄 Avg Return Rate: {df['return_rate'].mean():.1f}%")

# ─────────────────────────────────────────────
# 2. MONTHLY SALES TREND
# ─────────────────────────────────────────────
monthly = df.groupby('date').agg(
    total_sales=('net_sales', 'sum'),
    total_units=('units_sold', 'sum'),
    total_transactions=('transactions', 'sum')
).reset_index()
monthly['MoM_growth'] = monthly['total_sales'].pct_change() * 100

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Bar(
    x=monthly['date'], y=monthly['total_sales'],
    name='Net Sales ($)', marker_color='#1565C0', opacity=0.8
), secondary_y=False)
fig1.add_trace(go.Scatter(
    x=monthly['date'], y=monthly['MoM_growth'],
    name='MoM Growth (%)', mode='lines+markers',
    line=dict(color='#FF6F00', width=2),
    marker=dict(size=8)
), secondary_y=True)
fig1.update_layout(
    title='Monthly Sales Revenue & Month-over-Month Growth (2023)',
    xaxis_title='Month',
    title_font_size=18
)
fig1.update_yaxes(title_text="Net Sales ($)", secondary_y=False)
fig1.update_yaxes(title_text="MoM Growth (%)", secondary_y=True)
fig1.write_html('monthly_sales_trend.html')
print("\n✅ Chart saved: monthly_sales_trend.html")

# ─────────────────────────────────────────────
# 3. CATEGORY PERFORMANCE
# ─────────────────────────────────────────────
category_perf = df.groupby('category').agg(
    total_sales=('net_sales', 'sum'),
    total_units=('units_sold', 'sum'),
    avg_return_rate=('return_rate', 'mean'),
    avg_discount=('discount_pct', 'mean')
).reset_index().sort_values('total_sales', ascending=False)

category_perf['sales_share'] = (category_perf['total_sales'] / category_perf['total_sales'].sum() * 100).round(1)

print("\n📊 Category Performance Summary:")
print(category_perf.round(1).to_string(index=False))

fig2 = make_subplots(rows=1, cols=2,
                      subplot_titles=('Revenue by Category', 'Return Rate by Category'),
                      specs=[[{"type": "pie"}, {"type": "bar"}]])

fig2.add_trace(go.Pie(
    labels=category_perf['category'],
    values=category_perf['total_sales'],
    hole=0.4,
    textinfo='label+percent'
), row=1, col=1)

fig2.add_trace(go.Bar(
    x=category_perf['category'],
    y=category_perf['avg_return_rate'],
    marker_color='#EF5350',
    text=category_perf['avg_return_rate'].round(1),
    texttemplate='%{text}%'
), row=1, col=2)

fig2.update_layout(title_text='Category Performance Overview', title_font_size=18, showlegend=False)
fig2.write_html('category_performance.html')
print("✅ Chart saved: category_performance.html")

# ─────────────────────────────────────────────
# 4. REGIONAL ANALYSIS
# ─────────────────────────────────────────────
regional = df.groupby('region').agg(
    total_sales=('net_sales', 'sum'),
    total_units=('units_sold', 'sum'),
    num_transactions=('transactions', 'sum'),
    avg_basket=('avg_transaction_value', 'mean')
).reset_index().sort_values('total_sales', ascending=False)

print("\n🗺️ Regional Performance:")
print(regional.round(1).to_string(index=False))

fig3 = px.bar(
    regional, x='region', y='total_sales',
    title='Total Net Sales by Region (2023)',
    color='total_sales',
    color_continuous_scale='Blues',
    text='total_sales',
    labels={'total_sales': 'Net Sales ($)', 'region': 'Region'}
)
fig3.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
fig3.update_layout(title_font_size=18, showlegend=False)
fig3.write_html('regional_sales.html')
print("✅ Chart saved: regional_sales.html")

# ─────────────────────────────────────────────
# 5. SEASONALITY HEATMAP
# ─────────────────────────────────────────────
pivot = df.groupby(['month_num', 'category'])['net_sales'].sum().reset_index()
pivot_table = pivot.pivot(index='category', columns='month_num', values='net_sales')
pivot_table.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig4 = px.imshow(
    pivot_table,
    title='Sales Heatmap: Category × Month (Seasonality)',
    color_continuous_scale='YlOrRd',
    aspect='auto',
    labels=dict(color='Net Sales ($)')
)
fig4.update_layout(title_font_size=18)
fig4.write_html('seasonality_heatmap.html')
print("✅ Chart saved: seasonality_heatmap.html")

# ─────────────────────────────────────────────
# 6. DISCOUNT vs SALES ANALYSIS
# ─────────────────────────────────────────────
fig5 = px.scatter(
    df, x='discount_pct', y='net_sales',
    color='category', size='units_sold',
    title='Discount % vs Net Sales by Category',
    labels={'discount_pct': 'Discount (%)', 'net_sales': 'Net Sales ($)'},
    opacity=0.6
)
fig5.write_html('discount_vs_sales.html')
print("✅ Chart saved: discount_vs_sales.html")

# ─────────────────────────────────────────────
# 7. EXCEL EXPORT (Power BI ready)
# ─────────────────────────────────────────────
with pd.ExcelWriter('retail_sales_dashboard_data.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    monthly.to_excel(writer, sheet_name='Monthly Trend', index=False)
    category_perf.to_excel(writer, sheet_name='Category Performance', index=False)
    regional.to_excel(writer, sheet_name='Regional Performance', index=False)

print("✅ Excel file saved: retail_sales_dashboard_data.xlsx")
print("   → Ready to connect to Power BI for dashboard creation!")

# ─────────────────────────────────────────────
# 8. KEY INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)
top_cat = category_perf.iloc[0]['category']
top_region = regional.iloc[0]['region']
best_month = monthly.loc[monthly['total_sales'].idxmax(), 'date'].strftime('%B')

print(f"""
1. 📈 SEASONALITY: November–December are peak months, with sales up to 50%
   above baseline. Inventory and staffing must be scaled accordingly.

2. 🏆 TOP CATEGORY: {top_cat} drives the highest revenue share. Focus
   promotional budgets here for maximum ROI.

3. 🗺️ TOP REGION: {top_region} leads all regions in total revenue. Replicate
   its store layout and merchandising strategy in underperforming regions.

4. 💸 DISCOUNTING RISK: High discount rates (>20%) do not consistently
   produce proportionally higher sales — selective discounting is recommended.

5. 🔄 RETURN RATE WATCH: Certain categories show return rates above 6%.
   Product quality reviews and clearer sizing/specs can reduce costly returns.

6. 🌟 BEST MONTH: {best_month} was the highest revenue month of 2023.
   Plan campaign launches and new product releases ahead of this window.
""")

print("✅ Analysis Complete! Files ready for GitHub upload.")
