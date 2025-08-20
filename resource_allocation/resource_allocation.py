import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Data Generation and Simulation ---
# This function creates a more complex, multi-dimensional dataset
def generate_multivariate_workload_data(num_records=2000):
    """
    Simulates a multivariate time-series dataset for cloud resources.
    Includes CPU, Memory, and Network I/O.
    """
    timestamps = pd.to_datetime(pd.Series(range(num_records)), unit='h', origin='2025-01-01')

    # Base trends for each resource
    cpu_base = 50 + 25 * np.sin(np.linspace(0, 4 * np.pi, num_records))
    mem_base = 60 + 15 * np.cos(np.linspace(0, 4 * np.pi, num_records))
    net_base = 30 + 10 * np.sin(np.linspace(0, 4 * np.pi, num_records))
    
    # Introduce random noise and periodic spikes to simulate real-world workloads
    cpu_noise = np.random.normal(0, 5, num_records)
    mem_noise = np.random.normal(0, 4, num_records)
    net_noise = np.random.normal(0, 3, num_records)

    # Simulate peak hours (e.g., business day)
    peak_hours = np.zeros(num_records)
    for i in range(num_records):
        if (i % 24) >= 9 and (i % 24) <= 17:  # Simulating 9 AM to 5 PM
            peak_hours[i] = 10
    
    cpu_usage = np.clip(cpu_base + cpu_noise + peak_hours, 0, 100)
    mem_usage = np.clip(mem_base + mem_noise + peak_hours, 0, 100)
    net_usage = np.clip(net_base + net_noise + peak_hours, 0, 100)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu_usage,
        'mem_usage': mem_usage,
        'net_usage': net_usage
    })
    
    return data

# Generate the simulated data
df = generate_multivariate_workload_data()

print("--- Simulated Multi-Resource Workload Snapshot ---")
print(df.head())
print("\n--- Data Visualization ---")

# Plot the generated data
plt.figure(figsize=(15, 8))
plt.plot(df['timestamp'], df['cpu_usage'], label='CPU Usage (%)', color='red')
plt.plot(df['timestamp'], df['mem_usage'], label='Memory Usage (%)', color='blue')
plt.plot(df['timestamp'], df['net_usage'], label='Network I/O (%)', color='green')
plt.title('Simulated Cloud Resource Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Resource Usage (%)')
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Data Preprocessing for Multiple Resources ---
# Create lag features for each resource metric to use for prediction.
def create_lag_features(data, lags):
    """
    Generates lag features for each resource column in the DataFrame.
    """
    for col in ['cpu_usage', 'mem_usage', 'net_usage']:
        for i in range(1, lags + 1):
            data[f'{col}_lag_{i}'] = data[col].shift(i)
    return data

num_lags = 5  # Use the past 5 hours of data
df_processed = create_lag_features(df, num_lags)
df_processed.dropna(inplace=True)

# Define features (X) and targets (y)
feature_cols = [f'{col}_lag_{i}' for col in ['cpu_usage', 'mem_usage', 'net_usage'] for i in range(1, num_lags + 1)]
target_cols = ['cpu_usage', 'mem_usage', 'net_usage']

X = df_processed[feature_cols]
y = df_processed[target_cols]

# Split the data into training and testing sets (time-based split)
split_point = int(len(df_processed) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 3. Model Training (Random Forest and XGBoost) ---
print("\n--- Training Machine Learning Models ---")

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Initialize and train the XGBoost Regressor
# XGBoost is highly efficient and often performs better on structured data.
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# --- Evaluate the Models ---
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"XGBoost RMSE: {xgb_rmse:.2f}")

# Choose the best model
best_model = xgb_model if xgb_rmse < rf_rmse else rf_model
print(f"\nSelected Model for Allocation: {'XGBoost' if best_model == xgb_model else 'Random Forest'}")

# --- 4. Intelligent Resource Allocation Logic ---
# A simplified function to simulate resource allocation based on model predictions.
def allocate_resources_from_prediction(predicted_resources, vm_config):
    """
    Calculates the number of VMs to allocate based on predicted resource needs.
    Args:
        predicted_resources (list/array): A list of predicted values [cpu, memory, network].
        vm_config (dict): Configuration of a single VM (e.g., {'cpu': 25, 'mem': 16, 'net': 20}).
    Returns:
        int: The number of VMs to allocate.
    """
    # Define safety buffer in percentage points
    safety_buffer_cpu = 10
    safety_buffer_mem = 5
    safety_buffer_net = 5
    
    # Add buffer to the prediction
    required_cpu = predicted_resources[0] + safety_buffer_cpu
    required_mem = predicted_resources[1] + safety_buffer_mem
    required_net = predicted_resources[2] + safety_buffer_net
    
    # Calculate VMs needed for each resource type, rounding up.
    vms_needed_cpu = np.ceil(required_cpu / vm_config['cpu'])
    vms_needed_mem = np.ceil(required_mem / vm_config['mem'])
    vms_needed_net = np.ceil(required_net / vm_config['net'])
    
    # The total number of VMs allocated must satisfy the highest demand.
    # We take the maximum of the three calculated values.
    return int(max(vms_needed_cpu, vms_needed_mem, vms_needed_net))

print("\n--- Dynamic Resource Allocation Demonstration ---")

# Define a standard VM's resource capacity in percentage
vm_resource_config = {'cpu': 25, 'mem': 20, 'net': 20}
print(f"Assuming each VM provides: {vm_resource_config['cpu']}% CPU, {vm_resource_config['mem']}% Memory, {vm_resource_config['net']}% Network I/O.")

# Let's take the very last data point from our test set
last_observation = X_test.iloc[-1].values.reshape(1, -1)

# Get the multi-resource prediction for the next time step
next_hour_prediction = best_model.predict(last_observation)[0]

print(f"Predicted resource usage for the next hour:")
print(f"  CPU: {next_hour_prediction[0]:.2f}%")
print(f"  Memory: {next_hour_prediction[1]:.2f}%")
print(f"  Network: {next_hour_prediction[2]:.2f}%")

# Use the prediction to allocate resources
vms_to_allocate = allocate_resources_from_prediction(next_hour_prediction, vm_resource_config)

print(f"\nBased on the prediction and the VM config, the system should allocate {vms_to_allocate} VMs.")

# --- Evaluate the effectiveness on the test set (simulation) ---
print("\n--- Simulating Allocation on Test Data ---")
total_overprovisioning_cost = 0
total_underprovisioning_count = 0
total_predicted_vms = 0
vm_cost_per_hour = 1  # Example cost unit

for i in range(len(X_test)):
    # Get prediction and actual values
    prediction = best_model.predict(X_test.iloc[[i]])[0]
    actual_resources = y_test.iloc[i].values
    
    # Calculate allocated VMs based on prediction
    allocated_vms = allocate_resources_from_prediction(prediction, vm_resource_config)
    
    # Calculate required VMs based on actual usage
    required_vms_cpu = np.ceil(actual_resources[0] / vm_resource_config['cpu'])
    required_vms_mem = np.ceil(actual_resources[1] / vm_resource_config['mem'])
    required_vms_net = np.ceil(actual_resources[2] / vm_resource_config['net'])
    required_vms = int(max(required_vms_cpu, required_vms_mem, required_vms_net))
    
    # Calculate overprovisioning and underprovisioning
    if allocated_vms > required_vms:
        overprovisioned_vms = allocated_vms - required_vms
        total_overprovisioning_cost += overprovisioned_vms * vm_cost_per_hour
    elif allocated_vms < required_vms:
        total_underprovisioning_count += 1
    
    total_predicted_vms += allocated_vms

print(f"Total simulated time steps: {len(X_test)}")
print(f"Total cost of overprovisioned VMs: ${total_overprovisioning_cost}")
print(f"Number of underprovisioning events (potential SLA violations): {total_underprovisioning_count}")
print(f"Average number of VMs allocated per time step: {total_predicted_vms / len(X_test):.2f}")