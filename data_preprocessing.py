import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Cleans the raw data, removes outliers, extracts features and standardizes values
print("Part 2: Data Preprocessing")

# Define file paths
data_dir = 'C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset'
processed_dir = os.path.join(data_dir, 'processed')
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir) # Creates directory if it doesn't exist

# Load combined dataset from exploration phase 
file_path = os.path.join(data_dir, 'C:/Users/csmaogun/OneDrive - Liverpool John Moores University/6219COMP_Technology/AI Technique/Transportation_Dataset/combined_raw_uber_data.csv')
print("Loading combined dataset...")
try:
    df = pd.read_csv(file_path)
    print(f"Loaded dataset successfully! Total records: {len(df)}")
except FileNotFoundError:
    print("Dataset not found. Please ensure the exploration step was completed.")
    raise

# Extracts useful time-based features from the date column
print("\n Temporal Feature Extraction")

df['Date/Time'] = pd.to_datetime(df['Date/Time'])
df['hour'] = df['Date/Time'].dt.hour
df['day_of_week'] = df['Date/Time'].dt.dayofweek
df['day_of_month'] = df['Date/Time'].dt.day
df['month_num'] = df['Date/Time'].dt.month

print("Temporal features added: hour, day_of_week, day_of_month, month_num")

# Checks and handles any missing value in the dataset
print("\n Handling Missing Values")

missing = df.isnull().sum()
print("Missing values per column:")
print(missing)

if missing.sum() == 0:
    print("No missing values found.")
else:
    # Fill numerical columns with median, categorical columns with mode
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values filled.")

# Removes geographical outliers that are outside NYC boundaries
print("\n Removing Geographical Outliers")

initial_count = len(df)
df = df[(df['Lat'] >= 40.5) & (df['Lat'] <= 41.0) & 
        (df['Lon'] >= -74.3) & (df['Lon'] <= -73.7)]
final_count = len(df)

print(f"Removed outliers. Records removed: {initial_count - final_count}")
print(f"Remaining records: {final_count}")

# Scale features for clustering technique
print("\n Feature Scaling for Clustering")

# 1. Spatial features only - for pure location-based clustering
X_spatial = df[['Lat', 'Lon']]
X_spatial_scaled = StandardScaler().fit_transform(X_spatial)

# 2. Spatial + Temporal features - for time-aware clustering
X_temp_spatial = df[['Lat', 'Lon', 'hour', 'day_of_week']]
X_temp_spatial_scaled = StandardScaler().fit_transform(X_temp_spatial)

# 3. Spatial + Temporal + Month - checks monthly pattern trends
X_with_month = df[['Lat', 'Lon', 'hour', 'day_of_week', 'month_num']]
X_with_month_scaled = StandardScaler().fit_transform(X_with_month)

print("Features scaled and ready for clustering")

# Saves all processed data for the clustering phase
print("\n Saving Preprocessed Data")

# Save data as CSV file
processed_csv_path = os.path.join(processed_dir, 'preprocessed_uber_data.csv')
df.to_csv(processed_csv_path, index=False)
print(f"Cleaned data saved to: {processed_csv_path}")

# Save NumPy arrays
np.save(os.path.join(processed_dir, 'X_spatial_scaled.npy'), X_spatial_scaled)
np.save(os.path.join(processed_dir, 'X_temporal_spatial_scaled.npy'), X_temp_spatial_scaled)
np.save(os.path.join(processed_dir, 'X_with_month_scaled.npy'), X_with_month_scaled)
np.save(os.path.join(processed_dir, 'month_data.npy'), df['month_num'].values)

print("Scaled features saved as .npy files")
print("\nData preprocessing complete.")