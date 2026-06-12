import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Function to generate synthetic data
def generate_data(num_samples=1000):
    np.random.seed(42)
    data = []
    # Profiles: [Temp_mean, Temp_std, Humid_mean, Humid_std, Moisture_mean, Moisture_std, Crop]
    profiles = [
        (28, 5, 80, 10, 70, 10, "Rice"),
        (24, 6, 60, 15, 45, 10, "Maize"),
        (18, 4, 75, 10, 50, 15, "Potato"),
        (25, 5, 65, 10, 60, 10, "Tomato"),
        (30, 6, 50, 15, 30, 10, "Sunflower"),
        (20, 5, 55, 15, 40, 10, "Wheat")
    ]
    
    for _ in range(num_samples):
        profile = profiles[np.random.randint(len(profiles))]
        temp = np.random.normal(profile[0], profile[1])
        humid = np.random.normal(profile[2], profile[3])
        moist = np.random.normal(profile[4], profile[5])
        data.append([max(0, temp), max(0, min(100, humid)), profile[6], max(0, min(100, moist))])
        
    df = pd.DataFrame(data, columns=["Temperature", "Humidity", "Crop", "Target_Water_Level"])
    return df

def run_training():
    print("Generating synthetic crop data...")
    df = generate_data(2000)

    le = LabelEncoder()
    df["Crop_Encoded"] = le.fit_transform(df["Crop"])

    X = df[["Temperature", "Humidity", "Crop_Encoded"]]
    y = df["Target_Water_Level"]

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    score = model.score(X, y)
    print(f"Model Training R^2 Score: {score:.2f}")

    print("Saving model to crop_water_model.pkl...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "crop_water_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoder": le}, f)
        
    print("Success! Model trained and saved.")
    return score

if __name__ == "__main__":
    run_training()
