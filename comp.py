import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import functools
import warnings
import logging
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_SEARCH_RADIUS = 15  # 15 km
REPOSITIONING_WINDOW = 2  # hours 
TAXI_SPEED_BASE = 15  # km/hours

class TaxiCompany:
    def __init__(self, name, taxis, pricing_strategy):
        self.name = name
        self.taxis = taxis
        self.pricing_strategy = pricing_strategy
        if 'satisfaction_modifier' not in self.pricing_strategy:
            self.pricing_strategy['satisfaction_modifier'] = 1.0
        self.revenue = 0
        self.successful_matches = 0
        self.customer_satisfaction = []
        self.market_share = 0
        self.operational_costs = 0

    def add_customer_satisfaction(self, satisfaction):
        self.customer_satisfaction.append(satisfaction)

    def update_pricing_strategy(self, competitors_data):
        avg_competitor_price = np.mean([comp['price_multiplier'] for comp in competitors_data])
        if self.market_share < 0.4:
            self.pricing_strategy['price_multiplier'] = max(0.8, min(avg_competitor_price * 0.95, self.pricing_strategy['price_multiplier'] * 0.98))
        elif self.market_share > 0.6:
            self.pricing_strategy['price_multiplier'] = min(1.2, max(avg_competitor_price * 1.05, self.pricing_strategy['price_multiplier'] * 1.02))
        
        # Adjust satisfaction modifier based on recent performance
        recent_satisfaction = np.mean(self.customer_satisfaction[-100:]) if self.customer_satisfaction else 1.0
        self.pricing_strategy['satisfaction_modifier'] = max(0.9, min(1.1, recent_satisfaction))

class TaxiRepositioningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q

class CompetitiveMatching:
    def __init__(self, companies):
        self.companies = companies

    def competitive_bidding(self, request, taxi_trees, taxi_speed, max_wait_time, hour, day_of_week, demand, supply):
        pickup = (request['pickup_latitude'], request['pickup_longitude'])
        dropoff = (request['dropoff_latitude'], request['dropoff_longitude'])
        best_bid = float('inf')
        best_company = None
        best_taxi_idx = None

        for company in self.companies:
            distances, taxi_indices = taxi_trees[company.name].query([pickup], k=50, distance_upper_bound=MAX_SEARCH_RADIUS)
            base_price = request['trip_miles'] * 2  # Base price: $2 per mile
            dynamic_price = calculate_dynamic_price(base_price, demand, supply, company.pricing_strategy)

            for dist, taxi_idx in zip(distances[0], taxi_indices[0]):
                if taxi_idx >= len(company.taxis):
                    continue
                travel_time = max(1, (dist / taxi_speed) * 3600)
                wait_time = travel_time
                pickup_delay = max(0, wait_time - max_wait_time)
                if wait_time <= max_wait_time * 1.5:
                    request_utility = calculate_request_utility(travel_time, wait_time, demand, request['trip_miles'], request['trip_time'], dynamic_price, pickup_delay)
                    taxi_utility = calculate_taxi_utility(travel_time, request['trip_miles'], request['trip_time'], hour, day_of_week, dynamic_price, company)
                    combined_utility = (request_utility + taxi_utility) / 2
                    bid = 1 / (combined_utility + 1e-6)
                    if bid < best_bid:
                        best_bid = bid
                        best_company = company
                        best_taxi_idx = taxi_idx

        return best_company, best_taxi_idx, best_bid

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
    base_wait_time = 600 
    if num_requests > 1000:  # High demand
        base_wait_time *= 1.5 
    elif num_requests < 500:  # Low demand
        base_wait_time *= 0.8  
    if 6 <= hour < 10 or 16 <= hour < 20:  # Rush/Peak hours
        base_wait_time *= 1.2  
    elif 22 <= hour < 6:  # Night hours
        base_wait_time *= 0.9 
    if day_of_week in [5, 6]:  # Saturday and Sunday
        base_wait_time *= 1.1  
    return base_wait_time

def calculate_dynamic_price(base_price, demand, supply, company_strategy):
    price_multiplier = company_strategy.get('price_multiplier', 1.0)
    surge_factor = min(3.0, 1 + price_multiplier * (demand / max(1, supply) - 1))
    return base_price * surge_factor

def calculate_customer_satisfaction(wait_time, max_wait_time, company):
    base_satisfaction = max(0, 1 - (wait_time / max_wait_time))
    company_modifier = company.pricing_strategy.get('satisfaction_modifier', 1.0)
    return base_satisfaction * company_modifier

def calculate_operational_efficiency(successful_matches, total_taxis, time_period):
    return successful_matches / (total_taxis * time_period)

def get_taxi_speed(hour, day_of_week):
    speed = TAXI_SPEED_BASE
    if 6 <= hour < 10 or 16 <= hour < 20:  # Rush/Peak hours
        speed *= 0.8
    elif 22 <= hour < 6:  # Night hours
        speed *= 1.3
    if day_of_week in [5, 6]:  # Saturday and Sunday
        speed *= 1.1
    return speed

def calculate_request_utility(travel_time, wait_time, demand, trip_miles, trip_time, price, pickup_delay):
    trip_value = max(1, trip_miles * trip_time / 3600) * price
    return trip_value / max(1, travel_time + wait_time * 1.2 + 0.15 * demand + pickup_delay * 0.5)

def calculate_taxi_utility(travel_time, trip_miles, trip_time, hour, day_of_week, price, company):
    base_utility = max(1, trip_miles * trip_time / 3600) * price
    
    time_factor = 1.1 if 6 <= hour < 10 or 16 <= hour < 20 else 0.95 if 22 <= hour < 6 else 1.0
    day_factor = 1.05 if day_of_week in [5, 6] else 1.0 
    trip_length_factor = 1.0 if trip_time <= 1500 else 0.95 
    company_factor = company.pricing_strategy.get('utility_multiplier', 1.0)
    
    return base_utility * time_factor * day_factor * trip_length_factor * company_factor / max(1, travel_time * 1.1)

def predict_high_demand_areas(requests, current_time, window_size, demand_model):
    future_requests = requests[(requests['request_datetime'] > current_time) & 
                               (requests['request_datetime'] <= current_time + pd.Timedelta(hours=window_size))]
    if len(future_requests) == 0:
        return None, None

    X = future_requests[['hour', 'weekday', 'is_weekend']]
    predicted_demand = demand_model.predict(X)
    future_requests['predicted_demand'] = predicted_demand

    coords = future_requests[['pickup_latitude', 'pickup_longitude']].values
    db = DBSCAN(eps=0.01, min_samples=5).fit(coords)
    
    labels = db.labels_
    unique_labels = set(labels)
    
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise points
    
    if not unique_labels:
        return None, None

    best_cluster = max(unique_labels, key=lambda label: future_requests[labels == label]['predicted_demand'].sum())
    
    cluster_requests = future_requests[labels == best_cluster]
    weighted_lat = (cluster_requests['pickup_latitude'] * cluster_requests['predicted_demand']).sum() / cluster_requests['predicted_demand'].sum()
    weighted_lon = (cluster_requests['pickup_longitude'] * cluster_requests['predicted_demand']).sum() / cluster_requests['predicted_demand'].sum()
    
    return weighted_lat, weighted_lon

def redistribute_taxi(taxi, requests, all_requests, current_time, demand_model, rl_agent):
    state = get_state(taxi, requests, all_requests, current_time)
    action = rl_agent.get_action(state)

    new_lat, new_lon = taxi['latitude'], taxi['longitude']

    if action == 1:  # Move towards high demand area
        high_demand_lat, high_demand_lon = predict_high_demand_areas(all_requests, current_time, REPOSITIONING_WINDOW, demand_model)
        if high_demand_lat is not None and high_demand_lon is not None:
            new_lat += (high_demand_lat - taxi['latitude']) * 0.3
            new_lon += (high_demand_lon - taxi['longitude']) * 0.3
    elif action == 2:  # Move towards mean request location
        if len(requests) > 0:
            total_miles = requests['trip_miles'].sum()
            mean_lat = (requests['pickup_latitude'] * requests['trip_miles']).sum() / total_miles
            mean_lon = (requests['pickup_longitude'] * requests['trip_miles']).sum() / total_miles
            new_lat += (mean_lat - taxi['latitude']) * 0.4
            new_lon += (mean_lon - taxi['longitude']) * 0.4

    if np.isnan(new_lat) or np.isnan(new_lon):
        return taxi

    taxi['latitude'] = new_lat
    taxi['longitude'] = new_lon
    return taxi

def get_state(taxi, requests, all_requests, current_time):
    try:
        lat_bin = int(taxi['latitude'] * 10) if not np.isnan(taxi['latitude']) else 0
        lon_bin = int(taxi['longitude'] * 10) if not np.isnan(taxi['longitude']) else 0
        time_bin = current_time.hour

        nearby_requests = len(requests[(np.abs(requests['pickup_latitude'] - taxi['latitude']) < 0.1) &
                                       (np.abs(requests['pickup_longitude'] - taxi['longitude']) < 0.1)])

        return (lat_bin, lon_bin, time_bin, nearby_requests)
    except Exception as e:
        return (0, 0, current_time.hour, 0)

def train_demand_model(requests):
    X = requests[['hour', 'weekday', 'is_weekend']]
    y = requests['trip_miles']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    
    rf_mse = mean_squared_error(y_test, rf_pred)
    gb_mse = mean_squared_error(y_test, gb_pred)
    
    if rf_mse < gb_mse:
        logging.info("RandomForest model selected for demand prediction")
        return rf_model, scaler
    else:
        logging.info("GradientBoosting model selected for demand prediction")
        return gb_model, scaler

def calculate_multi_objective_reward(wait_time, travel_time, trip_miles, trip_time, pickup_delay):
    w_wait = 0.25
    w_travel = 0.15
    w_revenue = 0.4
    w_pickup_delay = 0.2

    max_wait = 1800  # 30 minutes
    max_travel = 3600  # 1 hour
    max_pickup_delay = 600  # 10 minutes
    wait_score = 1 - min(wait_time / max_wait, 1)
    travel_score = 1 - min(travel_time / max_travel, 1)
    pickup_delay_score = 1 - min(pickup_delay / max_pickup_delay, 1)

    revenue = trip_miles * 2 
    max_revenue = 100 
    revenue_score = min(revenue / max_revenue, 1)

    return w_wait * wait_score + w_travel * travel_score + w_revenue * revenue_score + w_pickup_delay * pickup_delay_score

def process_batch(batch_data):
    batch_requests, companies, taxi_trees, taxi_speed, max_wait_time, hour, day_of_week, demand, supply = batch_data
    matching_engine = CompetitiveMatching(companies)
    assignments = []

    for _, request in batch_requests.iterrows():
        best_company, best_taxi_idx, best_bid = matching_engine.competitive_bidding(
            request, taxi_trees, taxi_speed, max_wait_time, hour, day_of_week, demand, supply
        )
        if best_company is not None:
            assignments.append((_, best_company.name, best_taxi_idx, best_bid))

    return assignments

def match_requests(requests, companies, pool, taxi_speed, max_wait_time, batch_size, hour, day_of_week, all_requests, demand_model, rl_agent, scaler):
    total_requests = len(requests)
    total_served = 0
    total_wait_time = 0
    taxi_trees = {company.name: cKDTree(company.taxis[['latitude', 'longitude']].values) for company in companies}
    requests['priority_score'] = requests['trip_miles'] * requests['trip_time'] * (1.1 if day_of_week in [5, 6] else 1.0)
    requests = requests.sort_values('priority_score', ascending=False)
    unassigned_requests = requests.copy()
    current_time = requests['request_datetime'].min()

    for attempt in range(1):
        batch_results = []
        demand = len(unassigned_requests)
        supply = sum(len(company.taxis) for company in companies)

        for i in range(0, len(unassigned_requests), batch_size):
            batch_requests = unassigned_requests.iloc[i:i+batch_size]
            batch_data = (batch_requests, companies, taxi_trees, taxi_speed, max_wait_time * (1 + 0.5 * attempt), hour, day_of_week, demand, supply)
            batch_results.append(pool.apply_async(process_batch, (batch_data,)))

        newly_assigned = []
        for result in batch_results:
            assignments = result.get()
            for request_idx, company_name, taxi_idx, _ in assignments:
                try:
                    request = unassigned_requests.loc[request_idx]
                    company = next(c for c in companies if c.name == company_name)
                    taxi = company.taxis.iloc[taxi_idx]
                    pickup = (request['pickup_latitude'], request['pickup_longitude'])
                    dropoff = (request['dropoff_latitude'], request['dropoff_longitude'])
                    travel_time, _ = get_travel_time_distance((taxi['latitude'], taxi['longitude']), pickup, taxi_speed)
                    wait_time = travel_time
                    pickup_delay = max(0, wait_time - max_wait_time)
                    
                    if wait_time <= max_wait_time * (1 + 0.5 * attempt):
                        total_served += 1
                        total_wait_time += wait_time
                        newly_assigned.append(request.name)
                        taxi['latitude'] = pickup[0]
                        taxi['longitude'] = pickup[1]
                        dropoff_travel_time, _ = get_travel_time_distance(pickup, dropoff, taxi_speed)
                        dropoff_time = current_time + pd.Timedelta(seconds=travel_time + dropoff_travel_time)

                        # RL agent for repositioning
                        old_state = get_state(taxi, batch_requests, all_requests, current_time)
                        taxi = redistribute_taxi(taxi, batch_requests[batch_requests.index > request_idx], all_requests, dropoff_time, demand_model, rl_agent)
                        new_state = get_state(taxi, batch_requests, all_requests, dropoff_time)

                        # Calculate reward based on multiple objectives
                        reward = calculate_multi_objective_reward(wait_time, travel_time, request['trip_miles'], request['trip_time'], pickup_delay)

                        # Update RL agent
                        rl_agent.update(old_state, rl_agent.get_action(old_state), reward, new_state)

                        # Update company stats/metrics
                        company.taxis.iloc[taxi_idx] = taxi
                        fare = request['trip_miles'] * 2 * company.pricing_strategy['price_multiplier']
                        company.revenue += fare
                        company.successful_matches += 1
                        satisfaction = calculate_customer_satisfaction(wait_time, max_wait_time, company)
                        company.add_customer_satisfaction(satisfaction)

                        # Calculate and update operational costs
                        operational_cost = calculate_operational_cost(travel_time, dropoff_travel_time)
                        company.operational_costs += operational_cost

                        # Update taxi tree for the company
                        taxi_trees[company.name] = cKDTree(company.taxis[['latitude', 'longitude']].values)
                        
                    else:
                        continue
                    
                except Exception as e:
                    logging.error(f"Error in match_requests: {e}")
                    continue

        unassigned_requests = unassigned_requests[~unassigned_requests.index.isin(newly_assigned)]
        if len(unassigned_requests) == 0:
            break

    # Update market share and pricing strategies
    total_matches = sum(company.successful_matches for company in companies)
    for company in companies:
        company.market_share = company.successful_matches / total_matches if total_matches > 0 else 0
        company.update_pricing_strategy([{'price_multiplier': c.pricing_strategy['price_multiplier']} for c in companies if c != company])

    avg_wait_time = total_wait_time / total_served if total_served > 0 else float('nan')
    return total_requests, total_served, avg_wait_time, set(requests.index) - set(unassigned_requests.index)

def calculate_operational_cost(travel_time, dropoff_travel_time):
    # Simplified cost calculation
    fuel_cost_per_hour = 5  # Assumed fuel cost per hour
    maintenance_cost_per_hour = 2  # Assumed maintenance cost per hour
    total_time_hours = (travel_time + dropoff_travel_time) / 3600
    return (fuel_cost_per_hour + maintenance_cost_per_hour) * total_time_hours

def main():
    # Load and preprocess data
    requests = pd.read_csv('1m2024.csv')
    taxi_data = pd.read_csv('20000.csv')
    
    requests['request_datetime'] = pd.to_datetime(requests['request_datetime'])
    requests['pickup_datetime'] = pd.to_datetime(requests['pickup_datetime'])
    requests['dropoff_datetime'] = pd.to_datetime(requests['dropoff_datetime'])
    requests['hour'] = requests['request_datetime'].dt.hour
    requests['date'] = requests['request_datetime'].dt.date
    requests['trip_time'] = (requests['dropoff_datetime'] - requests['pickup_datetime']).dt.total_seconds()
    requests = requests[requests['hour'] < 25].sort_values('request_datetime')
    
    requests['weekday'] = requests['request_datetime'].dt.weekday
    requests['is_weekend'] = requests['weekday'].isin([5, 6]).astype(int)
    requests['time_of_day'] = pd.cut(requests['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])

    uber_taxis = int(0.75 * len(taxi_data))
    uber = TaxiCompany("Uber", taxi_data.iloc[:uber_taxis], {
        "price_multiplier": 1.0, 
        "utility_multiplier": 1.0,
        "satisfaction_modifier": 1.05
    })
    lyft = TaxiCompany("Lyft", taxi_data.iloc[uber_taxis:], {
        "price_multiplier": 0.9, 
        "utility_multiplier": 1.1,
        "satisfaction_modifier": 1.0
    })
    companies = [uber, lyft]
    
    demand_model, scaler = train_demand_model(requests)
    rl_agent = TaxiRepositioningAgent(1000, 3)

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=ctx.cpu_count()) as pool:
        results = []
        for (day, hour), group in tqdm(requests.groupby(['date', 'hour'])):
            day_of_week = group['weekday'].iloc[0]
            taxi_speed = get_taxi_speed(hour, day_of_week)
            max_wait_time = get_max_wait_time(hour, len(group), day_of_week)
            batch_size = min(1000, max(100, len(group) // (ctx.cpu_count() * 2)))
            result = match_requests(group, companies, pool, taxi_speed, max_wait_time, batch_size, hour, day_of_week, requests, demand_model, rl_agent, scaler)
            results.append(((day, hour), result))
    
    # Process results
    total_requests = sum(r for _, (r, _, _, _) in results)
    total_served = sum(s for _, (_, s, _, _) in results)
    total_wait_time = sum(w * s for _, (_, s, w, _) in results if not np.isnan(w))
    
    served_indices = set().union(*[served_set for _, (_, _, _, served_set) in results])
    requests['served'] = requests.index.isin(served_indices)
    
    # Print analysis results
    print_analysis_results(requests, companies, total_requests, total_served, total_wait_time)
    
    # Create visualizations
    create_visualizations(requests, companies)

def print_analysis_results(requests, companies, total_requests, total_served, total_wait_time):
    if total_requests > 0:
        logging.info(f"\nTotal Requests: {total_requests}, Total Served: {total_served}, Percentage Served: {total_served / total_requests * 100:.2f}%")
        logging.info(f"Average Wait Time: {total_wait_time / total_served:.2f} seconds")
    else:
        logging.info(f"Total Requests: {total_requests}, Total Served: {total_served}")
    
    print("\nAnalysis of served vs. unserved requests:")
    print(requests.groupby('served')[['trip_miles', 'trip_time']].mean())
    
    print("\nService rate by time of day:")
    print(requests.groupby('time_of_day')['served'].mean())
    
    print("\nService rate for weekdays vs. weekends:")
    print(requests.groupby('is_weekend')['served'].mean())

    total_time_hours = (requests['request_datetime'].max() - requests['request_datetime'].min()).total_seconds() / 3600
    for company in companies:
        print(f"\n{company.name} Performance:")
        print(f"Total Revenue: ${company.revenue:.2f}")
        print(f"Total Operational Costs: ${company.operational_costs:.2f}")
        print(f"Net Profit: ${company.revenue - company.operational_costs:.2f}")
        print(f"Successful Matches: {company.successful_matches}")
        print(f"Average Revenue per Match: ${company.revenue / company.successful_matches:.2f}")
        
        avg_satisfaction = sum(company.customer_satisfaction) / len(company.customer_satisfaction)
        print(f"Average Customer Satisfaction: {avg_satisfaction:.2f}")
        
        efficiency = calculate_operational_efficiency(company.successful_matches, len(company.taxis), total_time_hours)
        print(f"Operational Efficiency: {efficiency:.4f} matches per taxi per hour")

    print("\nCompetition Analysis:")
    for company in companies:
        print(f"{company.name}:")
        print(f"  Market Share: {company.market_share:.2%}")
        print(f"  Current Price Multiplier: {company.pricing_strategy['price_multiplier']:.2f}")
        print(f"  Average Revenue per Match: ${company.revenue / company.successful_matches:.2f}")
        print(f"  Profit Margin: {(company.revenue - company.operational_costs) / company.revenue:.2%}")

def create_visualizations(requests, companies):
    # Original visualizations
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    requests.groupby('time_of_day')['served'].mean().plot(kind='bar')
    plt.title('Service Rate by Time of Day')
    plt.ylabel('Service Rate')

    plt.subplot(2, 2, 2)
    requests.groupby('is_weekend')['served'].mean().plot(kind='bar')
    plt.title('Service Rate: Weekdays vs Weekends')
    plt.xticks([0, 1], ['Weekday', 'Weekend'])
    plt.ylabel('Service Rate')

    plt.subplot(2, 2, 3)
    for company in companies:
        plt.bar(company.name, company.market_share)
    plt.title('Market Share')
    plt.ylabel('Market Share')

    plt.subplot(2, 2, 4)
    for company in companies:
        profit = company.revenue - company.operational_costs
        plt.bar(company.name, profit)
    plt.title('Net Profit')
    plt.ylabel('Profit ($)')

    plt.tight_layout()
    plt.savefig('taxi_service_analysis.png')
    plt.close()

    # New visualizations
    # Time Series Plot of Service Rate
    plt.figure(figsize=(12, 6))
    requests.set_index('request_datetime').resample('H')['served'].mean().plot()
    plt.title('Hourly Service Rate Over Time')
    plt.xlabel('Date and Time')
    plt.ylabel('Service Rate')
    plt.savefig('hourly_service_rate.png')
    plt.close()

    # Heatmap of Request Density
    plt.figure(figsize=(12, 10))
    plt.hexbin(requests['pickup_longitude'], requests['pickup_latitude'], gridsize=20, cmap='YlOrRd')
    plt.colorbar(label='Number of Requests')
    plt.title('Heatmap of Request Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('request_density_heatmap.png')
    plt.close()

    # Company Performance Comparison
    metrics = ['Revenue', 'Operational Costs', 'Net Profit', 'Successful Matches']
    company_data = {company.name: [company.revenue, company.operational_costs, 
                                   company.revenue - company.operational_costs, 
                                   company.successful_matches] for company in companies}
    df = pd.DataFrame(company_data, index=metrics)
    df.plot(kind='bar', figsize=(12, 6))
    plt.title('Company Performance Comparison')
    plt.ylabel('Value')
    plt.legend(title='Company')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('company_performance_comparison.png')
    plt.close()

    logging.info("Analysis complete. Visualizations saved.")

if __name__ == '__main__':
    main()
