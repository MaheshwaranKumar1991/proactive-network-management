# Proactive Network Management
### Forecasting Jitter, Loss and Latency for Optimal Performance

## Project Overview
This project monitors Microsoft Teams call quality metrics and predicts 
network degradation BEFORE it affects users using a CNN-LSTM model.

## Data Source
Microsoft Teams Graph API (Mock — same schema as real API)

## Pipeline
MS Teams API → Apache NiFi → Prometheus → CNN-LSTM → Spike Detection → NOC Alerts → Grafana

## Model
CNN-LSTM with Attention — Unified Multivariate Forecasting

## How to Run
1. Open Google Colab — colab.research.google.com
2. Upload teams_noc_pipeline.py
3. Run: pip install numpy scikit-learn ruptures
4. Run: exec(open("/content/teams_noc_pipeline.py").read())

## Key Results
- Precision: 0.83
- Recall: 0.89  
- Latency RMSE: 20ms
- Warns NOC team 2 hours before spike

## Tech Stack
Python | CNN-LSTM | Apache NiFi | Prometheus | Redis | Grafana
