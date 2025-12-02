# ✈️ Aviation Turnaround Delay Root-Cause Miner & Optimizer

A comprehensive data science project that predicts flight turnaround delays, identifies root causes, and provides actionable recommendations to reduce delays by 6-12%.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Business Impact](#business-impact)
- [Key Performance Indicators](#key-performance-indicators)
- [Technical Architecture](#technical-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Future Enhancements](#future-enhancements)

## 🎯 Problem Statement

### Business Scenario
Airline X loses millions of dollars annually due to turnaround delays—the time spent preparing an aircraft for its next flight (cleaning, fueling, boarding, baggage handling, etc.). These delays cause a ripple effect, leading to:
- Missed passenger connections
- Compensation costs
- Reduced fleet utilization
- Decreased customer satisfaction
- Regulatory penalties

**Goal:** Identify the root causes of these delays and build an optimizer that can recommend operational changes to reduce delays by 6–12%.

## 💰 Business Impact

### Current State (Based on Analysis)
- **Delay Rate:** 37.33% of flights experience delays >15 minutes
- **Average Delay:** ~25 minutes per delayed flight
- **Estimated Annual Cost:** $50 per delayed flight × 3,733 delays (per 10,000 flights) = **$186,650** per 10,000 flights

### Potential Savings
With a 6-12% reduction in delays:
- **Conservative (6%):** $11,199 savings per 10,000 flights
- **Optimistic (12%):** $22,398 savings per 10,000 flights
- **Annual Savings (for 100,000 flights/year):** $111,990 - $223,980

## 📊 Key Performance Indicators

### Primary KPIs
1. **On-Time Performance (OTP):** 62.67% (flights departing within 15 minutes of schedule)
2. **Delay Minutes:** Average 25.1 minutes for delayed flights
3. **Delay Cause Frequency:**
   - Arrival Delay Propagation: 34.9%
   - Total Ground Time: 30.0%
   - Gate Delays: 6.5%
   - Weather Delays: 5.5%
   - Gate Availability: 4.9%

## 🏗️ Technical Architecture

### Data Pipeline
```
Synthetic Data Generation → Feature Engineering → Model Training → SHAP Analysis → Streamlit Dashboard
```

### Technology Stack
- **Language:** Python 3.13
- **ML Libraries:** scikit-learn, XGBoost, SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy

### Models Implemented
1. **Random Forest Classifier**
   - Feature importance analysis
   - Robust to outliers
   - Interpretable results

2. **XGBoost Classifier**
   - Superior prediction accuracy
   - SHAP value integration
   - Production-ready performance

## 🎉 Results

### Model Performance

#### Random Forest
- **Accuracy:** 98%
- **Precision:** 98% (Class 0), 98% (Class 1)
- **Recall:** 99% (Class 0), 96% (Class 1)
- **ROC-AUC:** 0.9986

#### XGBoost (Production Model)
- **Accuracy:** 100%
- **Precision:** 100% (Class 0), 99% (Class 1)
- **Recall:** 100% (Class 0), 100% (Class 1)
- **ROC-AUC:** 0.9999

### Top Predictive Features
1. **Arrival Delay** (34.9%) - Incoming flight delays propagate
2. **Total Ground Time** (30.0%) - Sum of all turnaround operations
3. **Gate Delay** (6.5%) - Gate availability issues
4. **Weather Delay** (5.5%) - Adverse weather conditions
5. **Gate Available** (4.9%) - Binary gate availability flag

### Key Insights
- **Delay Propagation:** Incoming delays are the #1 predictor
- **Weather Impact:** Snowy conditions increase delay rate by 3x
- **Gate Management:** Gate unavailability adds 15-30 minutes
- **Critical Path:** Boarding time is the longest single operation

## 🚀 Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/rengavk/aviation-turnaround-optimizer.git
cd aviation-turnaround-optimizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate synthetic data**
```bash
python src/data_generator.py
```

4. **Train models**
```bash
python src/model_training.py
```

5. **Launch dashboard**
```bash
streamlit run app.py
```

## 📖 Usage

### Running the Dashboard
```bash
streamlit run app.py
```

The dashboard provides three main views:

1. **Performance Dashboard**
   - KPI metrics (OTP, Average Delay, Total Delays, Cost)
   - Delay distribution visualization
   - Root cause analysis charts
   - Weather impact analysis

2. **Delay Predictor**
   - Interactive parameter inputs
   - Real-time delay probability prediction
   - Risk assessment (Low/Moderate/High)
   - Actionable recommendations

3. **Root Cause Analysis**
   - Detailed delay factor analysis
   - Correlation heatmap
   - Individual flight turnaround Gantt charts

### Generating New Data
```bash
python src/data_generator.py
```

### Retraining Models
```bash
python src/model_training.py
```

## 📁 Project Structure

```
aviation-turnaround-optimizer/
│
├── data/                           # Data files and outputs
│   ├── aviation_turnaround_data.csv
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── src/                            # Source code
│   ├── data_generator.py          # Synthetic data generation
│   └── model_training.py          # Model training pipeline
│
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## 🔬 Methodology

### Step 1: Problem Framing
- Defined business scenario and impact
- Established KPIs (OTP, Delay Minutes, Delay Causes)
- Set target: 6-12% delay reduction

### Step 2: Data Setup
- Generated 10,000 synthetic flight records
- Variables include:
  - Flight operations: cleaning, fueling, boarding, baggage times
  - External factors: weather, gate availability, crew readiness
  - Temporal features: hour, day of week, month

### Step 3: Feature Engineering
Created advanced features:
- **Buffer Times:** Deviation from expected operation times
- **Congestion Level:** Flights per hour at the airport
- **Weather Index:** Numeric severity encoding
- **Delay Propagation:** Incoming flight delay impact
- **Critical Path Time:** Maximum of parallel operations

### Step 4: Modeling
- **Random Forest:** Feature importance and robust predictions
- **XGBoost:** High-accuracy predictions with SHAP interpretability
- **Evaluation:** Classification metrics, ROC-AUC, confusion matrices

### Step 5: Visualization & Deployment
- SHAP summary plots for model interpretability
- Feature importance charts
- Interactive Streamlit dashboard
- Real-time delay prediction interface

## 🔮 Future Enhancements

### Short-term
- [ ] Integrate real BTS (Bureau of Transportation Statistics) data
- [ ] Add time-series forecasting for delay trends
- [ ] Implement A/B testing framework for interventions

### Medium-term
- [ ] Develop optimization engine for resource allocation
- [ ] Add multi-airport analysis
- [ ] Create mobile-responsive dashboard

### Long-term
- [ ] Real-time data integration via APIs
- [ ] Reinforcement learning for dynamic scheduling
- [ ] Predictive maintenance integration
- [ ] Cost-benefit analysis module

## 📈 Business Recommendations

Based on the analysis, we recommend:

1. **Improve Arrival Punctuality**
   - Focus on upstream delay prevention
   - Implement buffer time management

2. **Optimize Gate Management**
   - Pre-assign gates for high-risk flights
   - Implement dynamic gate allocation

3. **Weather Preparedness**
   - Pre-position resources during adverse weather
   - Develop weather-specific protocols

4. **Crew Coordination**
   - Ensure crew readiness before aircraft arrival
   - Implement crew tracking systems

5. **Process Optimization**
   - Parallelize ground operations where possible
   - Reduce boarding time through better procedures

## 👨‍💻 Author

**Renganayaki Venkatakrishnan**
- Data Scientist | Power BI Developer | Statistician
- [LinkedIn](https://www.linkedin.com/in/renganayaki-venkatakrishnan-349a61186/)
- [GitHub](https://github.com/rengavk)
- [Portfolio](https://rengavk.github.io/)

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by real-world aviation operations research
- Data generation methodology based on industry standards
- SHAP library for model interpretability

---

**Note:** This project uses synthetic data for demonstration purposes. For production deployment, integrate with real operational data sources.
