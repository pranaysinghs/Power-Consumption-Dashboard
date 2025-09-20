import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# ------------------------------- 
# PAGE CONFIG
# ------------------------------- 
st.set_page_config(
    page_title=" Power Analytics Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        margin: 5px 0;
        opacity: 0.9;
    }
    
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 50%, #ff9ff3 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .stSelectbox label {
        color: #000000 !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .element-container h2,
    .element-container h3,
    .element-container h4,
    .element-container h5,
    .element-container h6 {
        color: #ffffff !important;
    }
    
    div[data-testid="metric-container"] > label {
        color: #ffffff !important;
    }
    
    .sidebar .element-container h1,
    .sidebar .element-container h2,
    .sidebar .element-container h3,
    .sidebar .element-container h4,
    .sidebar .element-container h5,
    .sidebar .element-container h6 {
        color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #000000 !important;
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------- 
# LOAD & PREPROCESS DATA
# ------------------------------- 
@st.cache_data
def load_data():
    """Load and preprocess the power consumption data"""
    try:
        df = pd.read_csv("Power_Consumption_2019_2020.csv")
        
        # Rename and clean
        df.rename(columns={"Dates": "Date"}, inplace=True)
        
        # Convert Date column explicitly with correct format (day-month-year hour:minute)
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y %H:%M")
        
        # Set Date as index
        df.set_index("Date", inplace=True)
        
        return df
    except FileNotFoundError:
        st.error(" Power_Consumption_2019_2020.csv file not found! Please upload the data file.")
        st.stop()

# Load data
df = load_data()
states = df.columns.tolist()

# Calculate additional metrics
daily_df = df.resample('D').mean()
monthly_df = df.resample('ME').mean()

# ------------------------------- 
# HELPER FUNCTIONS
# ------------------------------- 
@st.cache_data
def calculate_metrics(data, state):
    """Calculate key metrics for the selected state"""
    current_consumption = data[state].iloc[-1]
    avg_consumption = data[state].mean()
    peak_consumption = data[state].max()
    min_consumption = data[state].min()
    
    # Calculate growth rate (last month vs previous month)
    monthly_data = data[state].resample('ME').mean()
    if len(monthly_data) >= 2:
        growth_rate = ((monthly_data.iloc[-1] - monthly_data.iloc[-2]) / monthly_data.iloc[-2]) * 100
    else:
        growth_rate = 0
    
    return {
        'current': current_consumption,
        'average': avg_consumption,
        'peak': peak_consumption,
        'minimum': min_consumption,
        'growth_rate': growth_rate
    }

def create_gauge_chart(value, title, max_val, color):
    """Create a gauge chart for KPIs"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': 'white'}},
        delta = {'reference': max_val * 0.7, 'font': {'color': 'white'}},
        gauge = {
            'axis': {'range': [None, max_val], 'tickcolor': 'white'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_val * 0.5], 'color': "rgba(0, 255, 0, 0.2)"},
                {'range': [max_val * 0.5, max_val * 0.8], 'color': "rgba(255, 255, 0, 0.2)"},
                {'range': [max_val * 0.8, max_val], 'color': "rgba(255, 0, 0, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        },
        number = {'font': {'size': 40, 'color': 'white'}}
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

@st.cache_data
def generate_forecast(data, state, forecast_days):
    """Generate Prophet forecast for the selected state"""
    forecast_data = data[[state]].reset_index()
    forecast_data = forecast_data.rename(columns={"Date": "ds", state: "y"})
    
    model = Prophet(
        daily_seasonality=True, 
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    model.fit(forecast_data)
    
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    return forecast_data, forecast

# ------------------------------- 
# MAIN DASHBOARD
# ------------------------------- 

# Header
st.markdown("""
<h1> POWER ANALYTICS COMMAND CENTER</h1>
<p style='text-align: center; font-size: 1.2rem; color: rgba(255,255,255,0.9); margin-top: -20px;'>

</p>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("## <span style='color: black;'> Control Panel</span>", unsafe_allow_html=True)

selected_state = st.sidebar.selectbox(
    " Select State",
    options=states,
    index=0,
    help="Choose a state to analyze power consumption"
)

forecast_period = st.sidebar.selectbox(
    " Forecast Period", 
    options=[30, 90, 180, 365],
    format_func=lambda x: f" {x} Days" if x == 30 else f" {x} Days ({x//30} Months)" if x < 365 else " 365 Days (1 Year)",
    index=1,
    help="Select the forecast horizon"
)

# Calculate metrics
metrics = calculate_metrics(df, selected_state)

# KPI Cards
st.markdown("### <span style='color: white;'> Key Performance Indicators</span>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['current']:.1f} MW</div>
        <div class="metric-label">CURRENT LOAD</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['average']:.1f} MW</div>
        <div class="metric-label">AVERAGE LOAD</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['peak']:.1f} MW</div>
        <div class="metric-label">PEAK LOAD</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['growth_rate']:+.1f}%</div>
        <div class="metric-label">GROWTH RATE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Charts Section
col_left, col_right = st.columns([3, 1])

with col_left:
    st.markdown("### <span style='color: white;'> Historical Power Consumption Analysis</span>", unsafe_allow_html=True)
    
    # Enhanced Time Series Plot
    fig1 = go.Figure()
    
    # Add main line with gradient fill
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df[selected_state],
        mode='lines',
        name='Power Consumption',
        line=dict(color='#00f2fe', width=3),
        fill='tonexty',
        fillcolor='rgba(0, 242, 254, 0.1)'
    ))
    
    # Add moving average
    ma_30 = df[selected_state].rolling(window=30*24).mean()  # 30-day moving average
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=ma_30,
        mode='lines',
        name='30-Day Moving Average',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    
    fig1.update_layout(
        title=f" Historical Power Consumption - {selected_state}",
        title_font=dict(size=20, color='white'),
        xaxis_title="Time Period",
        yaxis_title="Power Consumption (MW)",
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        height=500,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("### <span style='color: white;'> Current Load Status</span>", unsafe_allow_html=True)
    
    # Gauge Chart
    gauge_fig = create_gauge_chart(
        metrics['current'], 
        "Current Load (MW)", 
        metrics['peak'], 
        '#00f2fe'
    )
    
    st.plotly_chart(gauge_fig, use_container_width=True)

st.markdown("---")

# Forecast Section
col_forecast, col_insights = st.columns([3, 1])

with col_forecast:
    st.markdown("### <span style='color: white;'> AI-Powered Forecast Analysis</span>", unsafe_allow_html=True)
    
    # Generate forecast
    with st.spinner("Generating AI forecast..."):
        forecast_data, forecast = generate_forecast(df, selected_state, forecast_period)
    
    # Advanced Forecast Plot
    fig2 = go.Figure()
    
    # Historical data
    fig2.add_trace(go.Scatter(
        x=forecast_data["ds"], 
        y=forecast_data["y"], 
        mode="lines", 
        name="Historical",
        line=dict(color='#00f2fe', width=3)
    ))
    
    # Forecast line
    forecast_start = len(forecast_data)
    fig2.add_trace(go.Scatter(
        x=forecast["ds"][forecast_start:], 
        y=forecast["yhat"][forecast_start:], 
        mode="lines", 
        name="Forecast",
        line=dict(color='#ff6b6b', width=3)
    ))
    
    # Confidence interval
    fig2.add_trace(go.Scatter(
        x=forecast["ds"][forecast_start:],
        y=forecast["yhat_upper"][forecast_start:],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig2.add_trace(go.Scatter(
        x=forecast["ds"][forecast_start:],
        y=forecast["yhat_lower"][forecast_start:],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 107, 107, 0.2)',
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig2.update_layout(
        title=f" AI-Powered Forecast - {selected_state} ({forecast_period} Days)",
        title_font=dict(size=20, color='white'),
        xaxis_title="Date",
        yaxis_title="Power Consumption (MW)",
        template='plotly_dark',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with col_insights:
    st.markdown("### <span style='color: white;'> Analytics Insights</span>", unsafe_allow_html=True)
    
    # Insights Panel
    st.markdown(f"""
    <div style='color: white; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
        <div style='margin-bottom: 20px;'>
            <h4 style='color: #00f2fe; margin-bottom: 5px;'> Peak Usage</h4>
            <p style='font-size: 1.3rem; margin-bottom: 0;'>{metrics['peak']:.1f} MW</p>
        </div>
        
        <div style='margin-bottom: 20px;'>
            <h4 style='color: #00f2fe; margin-bottom: 5px;'> Minimum Usage</h4>
            <p style='font-size: 1.3rem; margin-bottom: 0;'>{metrics['minimum']:.1f} MW</p>
        </div>
        
        <div style='margin-bottom: 20px;'>
            <h4 style='color: #00f2fe; margin-bottom: 5px;'> Volatility</h4>
            <p style='font-size: 1.3rem; margin-bottom: 0;'>{df[selected_state].std():.1f} MW</p>
        </div>
        
        <div style='margin-bottom: 0;'>
            <h4 style='color: #00f2fe; margin-bottom: 5px;'> Load Factor</h4>
            <p style='font-size: 1.3rem; margin-bottom: 0;'>{(metrics['average']/metrics['peak']*100):.1f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Heatmap Comparison
st.markdown("### <span style='color: white;'> State-wise Power Consumption Comparison</span>", unsafe_allow_html=True)

# Heatmap Comparison (Top 10 states)
top_states = df.mean().nlargest(10).index.tolist()
heatmap_data = daily_df[top_states].T

fig5 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Viridis',
    colorbar=dict(title="Power Consumption (MW)")
))

fig5.update_layout(
    title=" State-wise Power Consumption Heatmap (Top 10 States)",
    title_font=dict(size=18, color='white'),
    template='plotly_dark',
    height=400
)

st.plotly_chart(fig5, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 40px;'>
     Powered by Advanced ML Analytics | Real-time Data Processing<br>
    <small>Built with Streamlit & Prophet AI</small>
</div>
""", unsafe_allow_html=True)

# Sidebar Additional Info
st.sidebar.markdown("---")
st.sidebar.markdown("### <span style='color: black;'> Current Analysis</span>", unsafe_allow_html=True)
st.sidebar.info(f"""
**Selected State:** {selected_state}  
**Forecast Period:** {forecast_period} days  
**Data Points:** {len(df):,}  
**Date Range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
""")

st.sidebar.markdown("### <span style='color: black;'> Quick Stats</span>", unsafe_allow_html=True)
st.sidebar.success(f"""
**Peak State:** {df.mean().idxmax()}  
**Peak Consumption:** {df.mean().max():.1f} MW  
**Total States:** {len(states)}  
**Analysis Ready!** 
""")