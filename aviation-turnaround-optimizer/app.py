import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Aviation Turnaround Optimizer",
    page_icon="✈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the aviation data"""
    df = pd.read_csv('data/aviation_turnaround_data.csv')
    df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])
    df['actual_arrival'] = pd.to_datetime(df['actual_arrival'])
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['actual_departure'] = pd.to_datetime(df['actual_departure'])
    return df

@st.cache_resource
def load_models():
    """Load trained models"""
    with open('data/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('data/xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    return rf_model, xgb_model

def create_delay_distribution(df):
    """Create delay distribution chart"""
    fig = px.histogram(
        df, 
        x='departure_delay_minutes',
        nbins=50,
        title='Distribution of Departure Delays',
        labels={'departure_delay_minutes': 'Delay (minutes)'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.add_vline(x=15, line_dash="dash", line_color="red", 
                  annotation_text="15-min threshold")
    return fig

def create_root_cause_chart(df):
    """Create root cause analysis chart"""
    # Calculate average contribution of each factor
    causes = {
        'Gate Delay': df['gate_delay'].mean(),
        'Weather Delay': df['weather_delay'].mean(),
        'Cleaning Time': df['cleaning_time'].mean() - 20,
        'Fueling Time': df['fueling_time'].mean() - 15,
        'Boarding Time': df['boarding_time'].mean() - 25,
        'Crew Delay': df[df['crew_ready_offset'] > 0]['crew_ready_offset'].mean()
    }
    
    causes_df = pd.DataFrame(list(causes.items()), columns=['Cause', 'Avg Impact (min)'])
    causes_df = causes_df.sort_values('Avg Impact (min)', ascending=True)
    
    fig = px.bar(
        causes_df,
        x='Avg Impact (min)',
        y='Cause',
        orientation='h',
        title='Root Cause Analysis: Average Delay Impact',
        color='Avg Impact (min)',
        color_continuous_scale='Reds'
    )
    return fig

def create_gantt_chart(flight_data):
    """Create a Gantt chart for turnaround visualization"""
    tasks = []
    start_time = flight_data['actual_arrival']
    
    # Define tasks
    task_list = [
        ('Cleaning', flight_data['cleaning_time']),
        ('Fueling', flight_data['fueling_time']),
        ('Baggage', flight_data['baggage_time']),
        ('Boarding', flight_data['boarding_time'])
    ]
    
    current_time = start_time
    for task_name, duration in task_list:
        tasks.append(dict(
            Task=task_name,
            Start=current_time,
            Finish=current_time + timedelta(minutes=duration),
            Duration=duration
        ))
        if task_name != 'Boarding':  # Boarding happens after others
            current_time = start_time
        else:
            current_time = current_time + timedelta(minutes=duration)
    
    df_gantt = pd.DataFrame(tasks)
    
    fig = px.timeline(
        df_gantt,
        x_start='Start',
        x_end='Finish',
        y='Task',
        title='Turnaround Process Timeline',
        color='Task'
    )
    fig.update_yaxes(categoryorder='total ascending')
    return fig

def predict_delay(model, features):
    """Predict delay probability"""
    prediction = model.predict_proba([features])[0]
    return prediction[1]  # Probability of delay

# Main app
def main():
    st.markdown('<div class="main-header">✈️ Aviation Turnaround Delay Optimizer</div>', 
                unsafe_allow_html=True)
    
    # Load data and models
    try:
        df = load_data()
        rf_model, xgb_model = load_models()
    except FileNotFoundError:
        st.error("⚠️ Data or models not found. Please run data generation and model training first.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Delay Predictor", "Root Cause Analysis"])
    
    if page == "Dashboard":
        st.header("📊 Performance Dashboard")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            otp = (df['is_delayed'] == 0).mean() * 100
            st.metric("On-Time Performance", f"{otp:.1f}%")
        
        with col2:
            avg_delay = df[df['is_delayed'] == 1]['departure_delay_minutes'].mean()
            st.metric("Avg Delay (Delayed Flights)", f"{avg_delay:.1f} min")
        
        with col3:
            total_delays = df['is_delayed'].sum()
            st.metric("Total Delayed Flights", f"{total_delays:,}")
        
        with col4:
            delay_cost = total_delays * 50  # Assume $50 per delayed flight
            st.metric("Estimated Cost", f"${delay_cost:,}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_delay_distribution(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_root_cause_chart(df), use_container_width=True)
        
        # Weather impact
        st.subheader("Weather Impact on Delays")
        weather_delays = df.groupby('weather_condition')['is_delayed'].mean() * 100
        fig = px.bar(
            x=weather_delays.index,
            y=weather_delays.values,
            labels={'x': 'Weather Condition', 'y': 'Delay Rate (%)'},
            title='Delay Rate by Weather Condition',
            color=weather_delays.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Delay Predictor":
        st.header("🔮 Delay Predictor")
        st.write("Enter flight parameters to predict delay probability")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cleaning_time = st.slider("Cleaning Time (min)", 10, 40, 20)
            fueling_time = st.slider("Fueling Time (min)", 5, 30, 15)
            boarding_time = st.slider("Boarding Time (min)", 15, 40, 25)
        
        with col2:
            baggage_time = st.slider("Baggage Time (min)", 10, 35, 20)
            gate_available = st.selectbox("Gate Available", [1, 0])
            crew_ready_offset = st.slider("Crew Ready Offset (min)", -30, 30, -10)
        
        with col3:
            weather = st.selectbox("Weather", ['Sunny', 'Rainy', 'Foggy', 'Snowy'])
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)
        
        # Calculate derived features
        weather_map = {'Sunny': 0, 'Rainy': 5, 'Foggy': 15, 'Snowy': 30}
        weather_delay = weather_map[weather]
        gate_delay = 0 if gate_available == 1 else 20
        
        weather_index = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}[weather]
        
        critical_path = max(cleaning_time, fueling_time, baggage_time)
        total_ground = critical_path + boarding_time + gate_delay + weather_delay
        
        # Feature vector
        features = [
            cleaning_time, fueling_time, boarding_time, baggage_time,
            gate_available, crew_ready_offset, gate_delay, weather_delay,
            hour, day_of_week, 1,  # month
            cleaning_time - 20, fueling_time - 15, boarding_time - 25,  # buffers
            10,  # flights_per_hour (default)
            weather_index, 0,  # arrival_delay (default)
            critical_path, total_ground
        ]
        
        if st.button("Predict Delay Risk"):
            prob = predict_delay(xgb_model, features)
            
            st.subheader("Prediction Result")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Delay Probability (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            if prob > 0.7:
                st.error(f"⚠️ HIGH RISK: {prob*100:.1f}% probability of delay")
                st.write("**Recommendations:**")
                st.write("- Expedite ground operations")
                st.write("- Ensure crew is ready early")
                st.write("- Pre-assign gate if possible")
            elif prob > 0.3:
                st.warning(f"⚡ MODERATE RISK: {prob*100:.1f}% probability of delay")
                st.write("**Recommendations:**")
                st.write("- Monitor turnaround closely")
                st.write("- Have backup resources ready")
            else:
                st.success(f"✅ LOW RISK: {prob*100:.1f}% probability of delay")
    
    elif page == "Root Cause Analysis":
        st.header("🔍 Root Cause Analysis")
        
        # Filter delayed flights
        delayed_flights = df[df['is_delayed'] == 1]
        
        st.write(f"Analyzing {len(delayed_flights)} delayed flights out of {len(df)} total flights")
        
        # Top delay causes
        st.subheader("Primary Delay Contributors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_root_cause_chart(df), use_container_width=True)
        
        with col2:
            # Correlation heatmap
            delay_factors = delayed_flights[[
                'cleaning_time', 'fueling_time', 'boarding_time', 
                'baggage_time', 'gate_delay', 'weather_delay', 
                'departure_delay_minutes'
            ]].corr()
            
            fig = px.imshow(
                delay_factors,
                title='Correlation Matrix: Delay Factors',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample flight analysis
        st.subheader("Sample Flight Turnaround Analysis")
        flight_idx = st.selectbox("Select Flight", delayed_flights.index[:20])
        
        if flight_idx is not None:
            flight = delayed_flights.loc[flight_idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(create_gantt_chart(flight), use_container_width=True)
            
            with col2:
                st.write("**Flight Details:**")
                st.write(f"Flight ID: {flight['flight_id']}")
                st.write(f"Delay: {flight['departure_delay_minutes']:.1f} min")
                st.write(f"Weather: {flight['weather_condition']}")
                st.write(f"Gate Available: {'Yes' if flight['gate_available'] == 1 else 'No'}")
                st.write(f"Total Turnaround: {flight['actual_turnaround_time']:.1f} min")

if __name__ == "__main__":
    main()
