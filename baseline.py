import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
import multiprocessing as mp
import functools
import warnings
import logging
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_SEARCH_RADIUS = 15  # 5 km
TAXI_SPEED_BASE = 15  # km/hours
GRID_SIZE = 0.01  # 1 Km

class SingleTaxiCompany:
    def __init__(self, name, taxis):
        self.name = name
        self.taxis = taxis
        self.revenue = 0
        self.successful_matches = 0
        self.operational_costs = 0

@functools.lru_cache(maxsize=10000)
def get_travel_time_distance(pickup, dropoff, speed):
    try:
        distance = geodesic(pickup, dropoff).kilometers
        travel_time = (distance / speed) * 3600  # Convert to seconds
        return travel_time, distance * 1000  # Convert to meters
    except Exception as e:
        logging.error(f"Error calculating travel time: {e}")
        return float('inf'), float('inf')

def get_max_wait_time(hour, num_requests, day_of_week):
    base_wait_time = 600  # 10 minutes
    if num_requests > 1000:  # High demand
        base_wait_time *= 1.3 
    elif num_requests < 500:  # Low demand
        base_wait_time *= 0.9  
    if 6 <= hour < 10 or 16 <= hour < 20:  # Rush/Peak hours
        base_wait_time *= 1.1  
    elif 22 <= hour < 6:  # Night hours
        base_wait_time *= 1.0 
    if day_of_week in [5, 6]:  # Saturday and Sunday
        base_wait_time *= 1.05  
    return base_wait_time

def get_taxi_speed(hour, day_of_week):
    speed = TAXI_SPEED_BASE
    if 6 <= hour < 10 or 16 <= hour < 20:  # Rush/Peak hours
        speed *= 0.85
    elif 22 <= hour < 6:  # Night hours
        speed *= 1.2
    if day_of_week in [5, 6]:  # Saturday and Sunday
        speed *= 1.05
    return speed

def get_grid_key(lat, lon):
    return (int(lat / GRID_SIZE), int(lon / GRID_SIZE))

def process_batch(batch_data):
    batch_requests, taxi_grid, taxi_speed, max_wait_time, taxi_locations = batch_data
    assignments = []

    for _, request in batch_requests.iterrows():
        pickup = (request['pickup_latitude'], request['pickup_longitude'])
        pickup_key = get_grid_key(pickup[0], pickup[1])
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                grid_key = (pickup_key[0] + dx, pickup_key[1] + dy)
                if grid_key in taxi_grid and taxi_grid[grid_key]:
                    taxi_idx = taxi_grid[grid_key].pop(0)
                    taxi_location = taxi_locations.iloc[taxi_idx]
                    dist = geodesic(pickup, (taxi_location['latitude'], taxi_location['longitude'])).kilometers
                    
                    if dist <= MAX_SEARCH_RADIUS:
                        travel_time = max(1, (dist / taxi_speed) * 3600)
                        wait_time = travel_time
                        
                        if wait_time <= max_wait_time:
                            assignments.append((_, taxi_idx, wait_time))
                            taxi_grid[get_grid_key(pickup[0], pickup[1])].append(taxi_idx)
                            break
                    else:
                        taxi_grid[grid_key].append(taxi_idx)
            if assignments and assignments[-1][0] == _:
                break
        
    return assignments

def match_requests(requests, company, pool, taxi_speed, max_wait_time, batch_size):
    total_requests = len(requests)
    total_served = 0
    total_wait_time = 0
    
    taxi_grid = defaultdict(list)
    for idx, taxi in company.taxis.iterrows():
        grid_key = get_grid_key(taxi['latitude'], taxi['longitude'])
        taxi_grid[grid_key].append(idx)
    
    batch_results = []
    for i in range(0, len(requests), batch_size):
        batch_requests = requests.iloc[i:i+batch_size]
        batch_data = (batch_requests, taxi_grid, taxi_speed, max_wait_time, company.taxis[['latitude', 'longitude']])
        batch_results.append(pool.apply_async(process_batch, (batch_data,)))

    assigned_requests = set()
    for result in batch_results:
        try:
            assignments = result.get()
            for request_idx, taxi_idx, wait_time in assignments:
                if request_idx not in assigned_requests:
                    assigned_requests.add(request_idx)
                    total_served += 1
                    total_wait_time += wait_time
                    
                    request = requests.loc[request_idx]
                    taxi = company.taxis.iloc[taxi_idx]
                    pickup = (request['pickup_latitude'], request['pickup_longitude'])
                    dropoff = (request['dropoff_latitude'], request['dropoff_longitude'])
                    
                    company.taxis.at[taxi_idx, 'latitude'] = pickup[0]
                    company.taxis.at[taxi_idx, 'longitude'] = pickup[1]
                    
                    dropoff_travel_time, distance = get_travel_time_distance(pickup, dropoff, taxi_speed)
                    fare = distance / 1000 * 2  # $2 per km
                    company.revenue += fare
                    company.successful_matches += 1
                    
                    operational_cost = calculate_operational_cost(wait_time, dropoff_travel_time)
                    company.operational_costs += operational_cost
        except Exception as e:
            logging.error(f"Error processing batch: {e}")

    avg_wait_time = total_wait_time / total_served if total_served > 0 else float('nan')
    return total_requests, total_served, avg_wait_time, assigned_requests

def calculate_operational_cost(travel_time, dropoff_travel_time):
    fuel_cost_per_hour = 5
    maintenance_cost_per_hour = 2
    total_time_hours = (travel_time + dropoff_travel_time) / 3600
    return (fuel_cost_per_hour + maintenance_cost_per_hour) * total_time_hours

def main():
    requests = pd.read_csv('150k2024.csv')
    taxi_data = pd.read_csv('1000.csv')
    
    requests['request_datetime'] = pd.to_datetime(requests['request_datetime'])
    requests['pickup_datetime'] = pd.to_datetime(requests['pickup_datetime'])
    requests['dropoff_datetime'] = pd.to_datetime(requests['dropoff_datetime'])
    requests['hour'] = requests['request_datetime'].dt.hour
    requests['date'] = requests['request_datetime'].dt.date
    requests['trip_time'] = (requests['dropoff_datetime'] - requests['pickup_datetime']).dt.total_seconds()
    requests = requests[requests['hour'] < 25].sort_values('request_datetime')
    
    requests['weekday'] = requests['request_datetime'].dt.weekday
    requests['is_weekend'] = requests['weekday'].isin([5, 6]).astype(int)

    company = SingleTaxiCompany("SingleCompany", taxi_data)
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=ctx.cpu_count()) as pool:
        results = []
        for (day, hour), group in tqdm(requests.groupby(['date', 'hour'])):
            day_of_week = group['weekday'].iloc[0]
            taxi_speed = get_taxi_speed(hour, day_of_week)
            max_wait_time = get_max_wait_time(hour, len(group), day_of_week)
            batch_size = min(1000, max(100, len(group) // (ctx.cpu_count() * 2)))
            result = match_requests(group, company, pool, taxi_speed, max_wait_time, batch_size)
            results.append(((day, hour), result))
    
    total_requests = sum(r for _, (r, _, _, _) in results)
    total_served = sum(s for _, (_, s, _, _) in results)
    total_wait_time = sum(w * s for _, (_, s, w, _) in results if not np.isnan(w))
    
    served_percentage = (total_served / total_requests) * 100
    avg_wait_time = total_wait_time / total_served if total_served > 0 else float('nan')
    
    print(f"Total Requests: {total_requests}")
    print(f"Total Served: {total_served}")
    print(f"Percentage Served: {served_percentage:.2f}%")
    print(f"Average Wait Time: {avg_wait_time:.2f} seconds")
    print(f"Total Revenue: ${company.revenue:.2f}")
    print(f"Total Operational Costs: ${company.operational_costs:.2f}")
    print(f"Net Profit: ${company.revenue - company.operational_costs:.2f}")

if __name__ == '__main__':
    main()