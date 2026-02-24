import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_data(num_rows=1000):
    start_date = datetime(2023, 1, 1)
    data = []
    
    for i in range(num_rows):
        dt = start_date + timedelta(hours=i)
        junction = np.random.randint(1, 5)
        # Higher traffic during day hours (8-20)
        hour = dt.hour
        base_vehicles = 20 + np.random.randint(0, 30)
        if 8 <= hour <= 20:
            base_vehicles += 50 + np.random.randint(0, 100)
            
        data.append({
            'DateTime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'Junction': junction,
            'Vehicles': base_vehicles,
            'ID': i
        })
        
    df = pd.DataFrame(data)
    df.to_csv('sample_traffic.csv', index=False)
    print("Generated sample_traffic.csv with 1000 rows.")

if __name__ == "__main__":
    generate_traffic_data()
