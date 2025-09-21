# Power Analytics Dashboard

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://power-consumption-dashboard.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

An interactive **Streamlit dashboard** for analyzing and forecasting **power consumption trends** across Indian states.  
It combines **time-series analysis, KPI metrics, AI forecasting (Prophet), and rich visualizations** into a single professional analytics tool.


## 🌍 Live Demo
🔗 [Try the Dashboard](https://power-consumption-dashboard.streamlit.app/)



## 🚀 Features
- 📊 **KPI Cards** → Current, Average, Peak, Growth rate  
- 📉 **Historical Trends** → Line chart with 30-day moving average  
- 🔵 **Gauge Chart** → Visualizes current load vs. peak load  
- 🤖 **Forecasting** → Prophet-based forecasting (30/90/180/365 days)  
- 📑 **Insights Cards** → Peak, Minimum, Volatility, Load Factor  
- 🌡 **Heatmap** → Top 10 states daily consumption comparison  
- 🎨 **Modern UI** → Dark theme, gradient styling, responsive layout  

---

## 🛠 Tech Stack
- [Streamlit](https://streamlit.io/) — Web dashboard framework  
- [Pandas](https://pandas.pydata.org/) — Data manipulation & resampling  
- [Prophet](https://facebook.github.io/prophet/) — Time-series forecasting  
- [Plotly](https://plotly.com/) — Interactive visualizations  
- [NumPy](https://numpy.org/), [Datetime](https://docs.python.org/3/library/datetime.html)  

---

## 📂 Project Structure
- app.py # Main Streamlit application code
- Power_Consumption_2019_2020.csv # Dataset
- requirements.txt # Python dependencies
