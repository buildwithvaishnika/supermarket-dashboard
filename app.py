import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# load data
# ─────────────────────────────────────────────

sales = pd.read_csv('outputs/fact_sales_cleaned.csv')
sales['Date'] = pd.to_datetime(sales['Date'])
sales_only = sales[sales['Transaction_Type'] == 'sale'].copy()

monthly_trends = pd.read_csv('outputs/monthly_trends.csv')
top_products = pd.read_csv('outputs/top_products.csv')
discount_impact = pd.read_csv('outputs/discount_impact.csv')
hourly_analysis = pd.read_csv('outputs/hourly_analysis.csv')
yoy_analysis = pd.read_csv('outputs/yoy_analysis.csv')
rfm = pd.read_csv('outputs/rfm_analysis.csv')
clusters = pd.read_csv('outputs/kmeans_clusters.csv')
anomalies_df = pd.read_csv('outputs/anomaly_detection.csv')
anomalies_df['Date'] = pd.to_datetime(anomalies_df['Date'])
forecast = pd.read_csv('outputs/prophet_forecast.csv')
forecast['ds'] = pd.to_datetime(forecast['ds'])
market_basket = pd.read_csv('outputs/market_basket_rules.csv')

# ─────────────────────────────────────────────
# KPI calculations
# ─────────────────────────────────────────────

total_revenue = sales_only['Sales_Amount_USD'].sum()
total_profit = sales_only['Gross_Profit_USD'].sum()
total_transactions = len(sales_only)
total_returns = len(sales[sales['Transaction_Type'] == 'return'])
avg_margin = sales_only['Profit_Margin'].mean()
total_products = sales_only['Item_Name'].nunique()
discount_revenue = sales_only[sales_only['Is_Discounted'] == 1]['Sales_Amount_USD'].sum()
discount_pct = (discount_revenue / total_revenue) * 100

# ─────────────────────────────────────────────
# color palette
# ─────────────────────────────────────────────

BG = '#0f0f1a'
CARD = '#1a1a2e'
CARD2 = '#2d2d44'
GREEN = '#00CC96'
RED = '#EF553B'
BLUE = '#636EFA'
PURPLE = '#AB63FA'
ORANGE = '#FFA15A'
WHITE = '#ffffff'
GRAY = '#aaaaaa'

# ─────────────────────────────────────────────
# app initialization
# ─────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# ─────────────────────────────────────────────
# layout
# ─────────────────────────────────────────────

app.layout = html.Div([

    # header
    html.Div([
        html.H1('Supermarket Sales Analytics', style={
            'color': WHITE, 'margin': '0', 'fontSize': '24px', 'fontWeight': '600'
        }),
        html.P('3 Years of Sales Data — July 2020 to June 2023', style={
            'color': GRAY, 'margin': '4px 0 0 0', 'fontSize': '13px'
        })
    ], style={
        'backgroundColor': CARD, 'padding': '20px 30px',
        'borderBottom': f'1px solid {CARD2}'
    }),

    # navigation
    html.Div([
        dcc.Tabs(id='tabs', value='tab-kpi', children=[
            dcc.Tab(label='Executive Summary', value='tab-kpi'),
            dcc.Tab(label='Sales Trends', value='tab-trends'),
            dcc.Tab(label='Product Analysis', value='tab-products'),
            dcc.Tab(label='Machine Learning', value='tab-ml'),
            dcc.Tab(label='Forecasting', value='tab-forecast'),
        ], style={'backgroundColor': CARD},
        colors={'border': CARD2, 'primary': GREEN, 'background': CARD})
    ], style={'backgroundColor': CARD}),

    # page content
    html.Div(id='page-content', style={
        'backgroundColor': BG, 'minHeight': '100vh', 'padding': '24px'
    })

], style={'backgroundColor': BG, 'fontFamily': 'Inter, system-ui, sans-serif'})


# ─────────────────────────────────────────────
# page routing callback
# ─────────────────────────────────────────────

@callback(Output('page-content', 'children'), Input('tabs', 'value'))
def render_page(tab):
    if tab == 'tab-kpi':
        return kpi_page()
    elif tab == 'tab-trends':
        return trends_page()
    elif tab == 'tab-products':
        return products_page()
    elif tab == 'tab-ml':
        return ml_page()
    elif tab == 'tab-forecast':
        return forecast_page()


# ─────────────────────────────────────────────
# helper: card wrapper
# ─────────────────────────────────────────────

def card(children, style={}):
    base = {
        'backgroundColor': CARD, 'borderRadius': '12px',
        'padding': '20px', 'marginBottom': '20px'
    }
    base.update(style)
    return html.Div(children, style=base)


def section_title(text):
    return html.H3(text, style={
        'color': WHITE, 'margin': '0 0 16px 0',
        'fontSize': '16px', 'fontWeight': '600'
    })


# ─────────────────────────────────────────────
# PAGE 1 — Executive Summary
# ─────────────────────────────────────────────

def kpi_page():
    # KPI gauge figure
    fig_kpi = make_subplots(
        rows=2, cols=4,
        specs=[[{'type': 'indicator'}] * 4] * 2,
    )

    indicators = [
        (total_revenue, 'Total Revenue', '$', '', 1, 1),
        (total_profit, 'Total Profit', '$', '', 1, 2),
        (total_transactions, 'Total Transactions', '', '', 1, 3),
        (total_returns, 'Total Returns', '', '', 1, 4),
        (avg_margin, 'Avg Profit Margin', '', '%', 2, 1),
        (total_products, 'Total Products', '', '', 2, 2),
        (discount_revenue, 'Discounted Revenue', '$', '', 2, 3),
        (discount_pct, 'Discount % of Revenue', '', '%', 2, 4),
    ]

    for value, title, prefix, suffix, row, col in indicators:
        fig_kpi.add_trace(go.Indicator(
            mode="number+delta+gauge",
            value=value,
            title={'text': title, 'font': {'size': 13, 'color': WHITE}},
            number={'prefix': prefix, 'suffix': suffix,
                    'font': {'size': 26, 'color': GREEN}, 'valueformat': ',.0f'},
            delta={'reference': value * 0.85, 'relative': True,
                   'increasing': {'color': GREEN}, 'decreasing': {'color': RED}},
            gauge={'axis': {'visible': False},
                   'bar': {'color': GREEN, 'thickness': 0.3},
                   'bgcolor': CARD2, 'borderwidth': 0}
        ), row=row, col=col)

    fig_kpi.update_layout(
        height=500, paper_bgcolor=CARD,
        font=dict(color=WHITE), margin=dict(t=60, b=20, l=20, r=20),
        title={'text': 'Key Performance Indicators', 'font': {'size': 16, 'color': WHITE}, 'x': 0.5}
    )

    # category revenue pie
    cat_revenue = sales_only.groupby('Category_Name')['Sales_Amount_USD'].sum().reset_index()
    fig_pie = px.pie(
        cat_revenue, values='Sales_Amount_USD', names='Category_Name',
        color_discrete_sequence=[GREEN, BLUE, RED, PURPLE, ORANGE, '#19D3F3'],
        title='Revenue by Category'
    )
    fig_pie.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE), height=350,
        title={'font': {'size': 15, 'color': WHITE}, 'x': 0.5},
        legend=dict(bgcolor=CARD2)
    )

    # discount vs full price
    disc_data = pd.DataFrame({
        'Type': ['Full Price', 'Discounted'],
        'Revenue': [total_revenue - discount_revenue, discount_revenue]
    })
    fig_disc = px.bar(
        disc_data, x='Type', y='Revenue', color='Type',
        color_discrete_map={'Full Price': GREEN, 'Discounted': RED},
        title='Full Price vs Discounted Revenue'
    )
    fig_disc.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE), height=350,
        showlegend=False,
        title={'font': {'size': 15, 'color': WHITE}, 'x': 0.5},
        xaxis=dict(gridcolor=CARD2),
        yaxis=dict(gridcolor=CARD2, title='Revenue (USD)')
    )

    return html.Div([
        card(dcc.Graph(figure=fig_kpi, config={'displayModeBar': False})),
        html.Div([
            html.Div(card(dcc.Graph(figure=fig_pie, config={'displayModeBar': False})),
                     style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
            html.Div(card(dcc.Graph(figure=fig_disc, config={'displayModeBar': False})),
                     style={'width': '48%', 'display': 'inline-block'}),
        ])
    ])


# ─────────────────────────────────────────────
# PAGE 2 — Sales Trends
# ─────────────────────────────────────────────

def trends_page():
    monthly_trends['period'] = monthly_trends['Month_Name'] + ' ' + monthly_trends['Year'].astype(str)

    # monthly trend
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_trends['period'], y=monthly_trends['total_sales'],
        mode='lines+markers', name='Monthly Revenue',
        line=dict(color=GREEN, width=2.5),
        fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)'
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_trends['period'], y=monthly_trends['rolling_3month_avg'],
        mode='lines', name='3 Month Rolling Avg',
        line=dict(color=RED, width=2, dash='dash')
    ))
    fig_trend.add_trace(go.Scatter(
        x=monthly_trends['period'], y=monthly_trends['total_profit'],
        mode='lines+markers', name='Monthly Profit',
        line=dict(color=BLUE, width=2), marker=dict(size=6)
    ))
    fig_trend.update_layout(
        title={'text': 'Monthly Revenue and Profit Trend (Jul 2020 - Jun 2023)',
               'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        height=450, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE), hovermode='x unified',
        xaxis=dict(tickangle=45, gridcolor=CARD2, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=CARD2, title='Amount (USD)'),
        legend=dict(bgcolor=CARD2, borderwidth=1)
    )

    # heatmap
    sales['Hour'] = pd.to_numeric(sales['Hour'], errors='coerce')
    heatmap_data = sales_only[sales_only['Hour'].notna()].groupby(
        ['Day_of_Week', 'Hour'])['Sales_Amount_USD'].sum().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_data.pivot(
        index='Day_of_Week', columns='Hour', values='Sales_Amount_USD').reindex(day_order)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[f"{int(h)}:00" for h in heatmap_pivot.columns],
        y=heatmap_pivot.index,
        colorscale='Viridis',
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Revenue: $%{z:,.0f}<extra></extra>'
    ))
    fig_heat.update_layout(
        title={'text': 'Revenue Heatmap by Day and Hour',
               'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        height=400, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE),
        xaxis_title='Hour of Day', yaxis_title='Day of Week'
    )

    # YoY
    yoy_plot = yoy_analysis[yoy_analysis['current_revenue'].notna()].copy()
    yoy_plot['Year'] = yoy_plot['Year'].astype(str)
    fig_yoy = px.bar(
        yoy_plot, x='Category_Name', y='current_revenue', color='Year',
        barmode='group', text='current_revenue',
        color_discrete_map={'2020': BLUE, '2021': GREEN, '2022': RED, '2023': PURPLE},
        title='Year over Year Revenue by Category'
    )
    fig_yoy.update_traces(texttemplate='$%{text:,.0f}', textposition='outside', textfont_size=9)
    fig_yoy.update_layout(
        height=450, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE),
        title={'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        xaxis=dict(gridcolor=CARD2, tickangle=15),
        yaxis=dict(gridcolor=CARD2, title='Revenue (USD)'),
        legend=dict(bgcolor=CARD2, borderwidth=1, title='Year')
    )

    return html.Div([
        card(dcc.Graph(figure=fig_trend, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_heat, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_yoy, config={'displayModeBar': False})),
    ])


# ─────────────────────────────────────────────
# PAGE 3 — Product Analysis
# ─────────────────────────────────────────────

def products_page():
    # top products
    fig_top = px.bar(
        top_products, x='total_revenue', y='Item_Name', orientation='h',
        color='avg_profit_margin', color_continuous_scale='Teal',
        text='total_revenue',
        hover_data=['Category_Name', 'total_transactions', 'avg_profit_margin'],
        title='Top 10 Products by Revenue (Color = Profit Margin %)'
    )
    fig_top.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_top.update_layout(
        height=500, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE),
        title={'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        xaxis=dict(gridcolor=CARD2, title='Total Revenue (USD)'),
        yaxis=dict(title='', categoryorder='total ascending'),
        coloraxis_colorbar=dict(title='Profit Margin %')
    )

    # treemap
    cat_prod = sales_only.groupby(['Category_Name', 'Item_Name'])['Sales_Amount_USD'].sum().reset_index()
    fig_tree = px.treemap(
        cat_prod, path=['Category_Name', 'Item_Name'],
        values='Sales_Amount_USD', color='Sales_Amount_USD',
        color_continuous_scale='Teal',
        title='Revenue Treemap by Category and Product'
    )
    fig_tree.update_traces(
        textinfo='label+value+percent root',
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>% of Total: %{percentRoot:.1%}<extra></extra>'
    )
    fig_tree.update_layout(
        height=550, paper_bgcolor=CARD,
        font=dict(color=WHITE, size=11),
        title={'font': {'size': 16, 'color': WHITE}, 'x': 0.5}
    )

    # discount impact
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Bar(
        name='Full Price Margin %', x=discount_impact['Category_Name'],
        y=discount_impact['full_price_margin_pct'],
        marker_color=GREEN, text=discount_impact['full_price_margin_pct'],
        texttemplate='%{text:.1f}%', textposition='outside'
    ))
    fig_disc.add_trace(go.Bar(
        name='Discounted Margin %', x=discount_impact['Category_Name'],
        y=discount_impact['discounted_margin_pct'],
        marker_color=RED, text=discount_impact['discounted_margin_pct'],
        texttemplate='%{text:.1f}%', textposition='outside'
    ))
    fig_disc.add_trace(go.Bar(
        name='Margin Loss from Discount', x=discount_impact['Category_Name'],
        y=discount_impact['margin_loss_from_discount'],
        marker_color=BLUE, text=discount_impact['margin_loss_from_discount'],
        texttemplate='-%{text:.1f}%', textposition='outside'
    ))
    fig_disc.update_layout(
        title={'text': 'Discount Impact on Profit Margin by Category',
               'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        barmode='group', height=450,
        paper_bgcolor=CARD, plot_bgcolor=CARD, font=dict(color=WHITE),
        xaxis=dict(gridcolor=CARD2, tickangle=15),
        yaxis=dict(gridcolor=CARD2, title='Profit Margin (%)'),
        legend=dict(bgcolor=CARD2, borderwidth=1)
    )

    return html.Div([
        card(dcc.Graph(figure=fig_top, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_tree, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_disc, config={'displayModeBar': False})),
    ])


# ─────────────────────────────────────────────
# PAGE 4 — Machine Learning
# ─────────────────────────────────────────────

def ml_page():
    # RFM
    segment_colors = {
        'Star Product': GREEN, 'High Performer': BLUE,
        'Average Performer': RED, 'Low Performer': PURPLE,
        'Underperformer': ORANGE
    }
    fig_rfm = px.scatter(
        rfm, x='Frequency', y='Monetary', size='Monetary',
        color='Segment', hover_name='Item_Name',
        color_discrete_map=segment_colors,
        title='RFM Product Segmentation - Frequency vs Monetary Value',
        labels={'Frequency': 'Transaction Frequency', 'Monetary': 'Total Revenue (USD)'}
    )
    fig_rfm.update_layout(
        height=500, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE),
        title={'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        xaxis=dict(gridcolor=CARD2), yaxis=dict(gridcolor=CARD2),
        legend=dict(bgcolor=CARD2, borderwidth=1)
    )

    # K-Means
    fig_km = px.scatter(
        clusters, x='total_revenue', y='avg_profit_margin',
        color='Cluster_Name', size='transaction_count', hover_name='Item_Name',
        color_discrete_map={
            'Low Revenue Products': RED,
            'Mid Revenue Products': BLUE,
            'High Revenue Products': GREEN
        },
        title='K-Means Product Clustering - Revenue vs Profit Margin',
        labels={'total_revenue': 'Total Revenue (USD)', 'avg_profit_margin': 'Avg Profit Margin (%)'}
    )
    fig_km.update_layout(
        height=500, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE),
        title={'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        xaxis=dict(gridcolor=CARD2), yaxis=dict(gridcolor=CARD2),
        legend=dict(bgcolor=CARD2, borderwidth=1)
    )

    # Anomaly Detection
    normal = anomalies_df[anomalies_df['Anomaly'] == 1]
    anomalies = anomalies_df[anomalies_df['Anomaly'] == -1]
    fig_anom = go.Figure()
    fig_anom.add_trace(go.Scatter(
        x=normal['Date'], y=normal['daily_revenue'],
        mode='markers', name='Normal Day',
        marker=dict(color=GREEN, size=5, opacity=0.6)
    ))
    fig_anom.add_trace(go.Scatter(
        x=anomalies['Date'], y=anomalies['daily_revenue'],
        mode='markers', name='Anomaly Detected',
        marker=dict(color=RED, size=12, symbol='star')
    ))
    fig_anom.update_layout(
        title={'text': 'Sales Anomaly Detection using Isolation Forest',
               'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        height=450, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE),
        xaxis=dict(gridcolor=CARD2, title='Date'),
        yaxis=dict(gridcolor=CARD2, title='Daily Revenue (USD)'),
        legend=dict(bgcolor=CARD2, borderwidth=1)
    )

    # Market Basket
    if 'product_a' in market_basket.columns and 'product_b' in market_basket.columns:
        pivot = market_basket.pivot_table(
            index='product_a', columns='product_b', values='lift', fill_value=0)
        fig_mb = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale='Teal',
            hovertemplate='%{y} + %{x}<br>Lift: %{z:.2f}<extra></extra>'
        ))
        fig_mb.update_layout(
            title={'text': 'Market Basket - Product Association Heatmap',
                   'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
            height=550, paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(color=WHITE, size=10), xaxis=dict(tickangle=45)
        )
        mb_chart = card(dcc.Graph(figure=fig_mb, config={'displayModeBar': False}))
    else:
        mb_chart = html.Div()

    return html.Div([
        card(dcc.Graph(figure=fig_rfm, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_km, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_anom, config={'displayModeBar': False})),
        mb_chart
    ])


# ─────────────────────────────────────────────
# PAGE 5 — Forecasting
# ─────────────────────────────────────────────

def forecast_page():
    daily_sales = sales_only.groupby('Date')['Sales_Amount_USD'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])

    # forecast chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=daily_sales['ds'], y=daily_sales['y'],
        mode='lines', name='Actual Sales',
        line=dict(color=GREEN, width=1.5), opacity=0.8
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Forecast',
        line=dict(color=RED, width=2, dash='dash')
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
        y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(239, 85, 59, 0.15)',
        line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'
    ))
    forecast_start = str(daily_sales['ds'].max().date())
    fig_fc.add_vline(x=forecast_start, line_dash='dot', line_color=WHITE)
    fig_fc.add_annotation(
        x=forecast_start, y=3500, text='Forecast Start',
        showarrow=False, font=dict(color=WHITE, size=12)
    )
    fig_fc.update_layout(
        title={'text': 'Sales Forecast - Next 6 Months using Prophet',
               'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        height=500, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE), hovermode='x unified',
        xaxis=dict(gridcolor=CARD2, title='Date'),
        yaxis=dict(gridcolor=CARD2, title='Daily Revenue (USD)'),
        legend=dict(bgcolor=CARD2, borderwidth=1)
    )

    # seasonality
    fig_season = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Overall Trend', 'Weekly Seasonality', 'Yearly Seasonality')
    )
    fig_season.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['trend'],
        mode='lines', name='Trend', line=dict(color=GREEN, width=2)
    ), row=1, col=1)
    fig_season.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['weekly'],
        mode='lines', name='Weekly', line=dict(color=RED, width=2)
    ), row=2, col=1)
    fig_season.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yearly'],
        mode='lines', name='Yearly', line=dict(color=BLUE, width=2)
    ), row=3, col=1)
    fig_season.update_layout(
        title={'text': 'Sales Decomposition - Trend, Weekly and Yearly Seasonality',
               'font': {'size': 16, 'color': WHITE}, 'x': 0.5},
        height=650, paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=WHITE)
    )
    for i in range(1, 4):
        fig_season.update_xaxes(gridcolor=CARD2, row=i, col=1)
        fig_season.update_yaxes(gridcolor=CARD2, row=i, col=1)

    return html.Div([
        card(dcc.Graph(figure=fig_fc, config={'displayModeBar': False})),
        card(dcc.Graph(figure=fig_season, config={'displayModeBar': False})),
    ])


# ─────────────────────────────────────────────
# run
# ─────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)
