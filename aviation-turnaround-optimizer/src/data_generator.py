import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_flight_data(n_records=10000):
    np.random.seed(42)
    
    # 1. Flight IDs and Schedule
    flight_ids = [f"AX{np.random.randint(100, 999)}" for _ in range(n_records)]
    
    start_date = datetime(2024, 1, 1)
    # Generate random dates within a 3-month period
    scheduled_arrivals = [start_date + timedelta(minutes=np.random.randint(0, 129600)) for _ in range(n_records)]
    
    # 2. Operational Variables (Turnaround Tasks)
    # Normal distributions with some skew for delays
    cleaning_time = np.random.normal(loc=20, scale=5, size=n_records) # Mean 20 mins
    cleaning_time = np.maximum(cleaning_time, 10) # Min 10 mins
    
    fueling_time = np.random.normal(loc=15, scale=4, size=n_records)
    fueling_time = np.maximum(fueling_time, 5)
    
    boarding_time = np.random.normal(loc=25, scale=6, size=n_records)
    boarding_time = np.maximum(boarding_time, 15)
    
    baggage_time = np.random.normal(loc=20, scale=5, size=n_records)
    baggage_time = np.maximum(baggage_time, 10)
    
    # 3. External Factors
    weather_conditions = np.random.choice(['Sunny', 'Rainy', 'Snowy', 'Foggy'], size=n_records, p=[0.7, 0.2, 0.05, 0.05])
    
    # Gate Availability (1 = Available, 0 = Not Available/Occupied)
    # 90% chance gate is ready
    gate_available = np.random.choice([1, 0], size=n_records, p=[0.9, 0.1])
    
    # Crew Ready Time (relative to scheduled arrival)
    # Negative means ready before arrival, Positive means late
    crew_ready_offset = np.random.normal(loc=-10, scale=15, size=n_records)
    
    # 4. Calculate Actual Arrival and Turnaround
    # Actual Arrival = Scheduled + Random Arrival Delay (Air traffic, etc.)
    arrival_delay_noise = np.random.exponential(scale=10, size=n_records) # Most on time, some long delays
    # Add some random early arrivals too
    arrival_offset = arrival_delay_noise - 5 
    
    actual_arrivals = [sched + timedelta(minutes=int(off)) for sched, off in zip(scheduled_arrivals, arrival_offset)]
    
    # 5. Construct DataFrame
    df = pd.DataFrame({
        'flight_id': flight_ids,
        'scheduled_arrival': scheduled_arrivals,
        'actual_arrival': actual_arrivals,
        'cleaning_time': cleaning_time.round(1),
        'fueling_time': fueling_time.round(1),
        'boarding_time': boarding_time.round(1),
        'baggage_time': baggage_time.round(1),
        'weather_condition': weather_conditions,
        'gate_available': gate_available,
        'crew_ready_offset': crew_ready_offset.round(1)
    })
    
    # 6. Logic for Delays
    # If gate is not available, add 15-30 mins delay
    df['gate_delay'] = df['gate_available'].apply(lambda x: 0 if x == 1 else np.random.randint(15, 30))
    
    # Weather impact
    weather_map = {'Sunny': 0, 'Rainy': 5, 'Foggy': 15, 'Snowy': 30}
    df['weather_delay'] = df['weather_condition'].map(weather_map) * np.random.uniform(0.8, 1.2, n_records)
    
    # Total Turnaround Time Required
    # Assuming parallel processes but critical path is usually max(cleaning, fueling) + boarding
    # Simplified: Max of ground ops + boarding + delays
    
    # Let's define Turnaround Time as the time from Actual Arrival to "Ready for Departure"
    # We simulate the "Actual Departure" based on these factors
    
    # Scheduled Turnaround Time (Standard) = 45 mins
    scheduled_turnaround = 45
    
    # Actual Turnaround Calculation
    # Critical path: max(cleaning, fueling, baggage) + boarding
    # (This is a simplification)
    ground_ops_time = df[['cleaning_time', 'fueling_time', 'baggage_time']].max(axis=1)
    total_ops_time = ground_ops_time + df['boarding_time']
    
    # Add external delays
    total_turnaround_needed = total_ops_time + df['gate_delay'] + df['weather_delay']
    
    # Crew delay: if crew is late (offset > 0), it might add to turnaround if it exceeds other tasks
    # If crew arrives 10 mins after plane, but ops take 40 mins, no delay.
    # If crew arrives 50 mins after plane, and ops take 40 mins, 10 mins delay.
    df['crew_delay_impact'] = df['crew_ready_offset'].apply(lambda x: max(0, x))
    
    # Final Turnaround Time
    df['actual_turnaround_time'] = np.maximum(total_turnaround_needed, df['crew_delay_impact'])
    
    # Calculate Departure Delays
    # Scheduled Departure = Scheduled Arrival + Scheduled Turnaround
    df['scheduled_departure'] = df['scheduled_arrival'] + timedelta(minutes=scheduled_turnaround)
    
    # Actual Departure = Actual Arrival + Actual Turnaround Time
    df['actual_departure'] = df['actual_arrival'] + pd.to_timedelta(df['actual_turnaround_time'], unit='m')
    
    # Target Variable: Departure Delay (in minutes)
    df['departure_delay_minutes'] = (df['actual_departure'] - df['scheduled_departure']).dt.total_seconds() / 60
    
    # Label: Delayed if > 15 mins
    df['is_delayed'] = (df['departure_delay_minutes'] > 15).astype(int)
    
    return df

if __name__ == "__main__":
    import os
    print("Generating synthetic aviation data...")
    df = generate_flight_data(10000)
    
    # Get the correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    output_path = os.path.join(data_dir, 'aviation_turnaround_data.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Data generated successfully! Saved to {output_path}")
    print(df.head())
    print(f"\nDelay Rate: {df['is_delayed'].mean():.2%}")
