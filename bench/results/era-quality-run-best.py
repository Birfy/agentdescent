import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def train_and_predict(train_path, test_path):
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop id column
    train = train.drop(columns=['id'], errors='ignore')
    test = test.drop(columns=['id'], errors='ignore')

    X = train.drop('MedHouseVal', axis=1)
    y = train['MedHouseVal']

    # Feature Engineering
    for df in [X, test]:
        df['MedInc_times_HouseAge'] = df['MedInc'] * df['HouseAge']
        df['MedInc_times_AveRooms'] = df['MedInc'] * df['AveRooms']
        df['MedInc_times_AveOccup'] = df['MedInc'] * df['AveOccup']
        df['AveRooms_minus_AveBedrms'] = df['AveRooms'] - df['AveBedrms']
        df['Population_times_AveOccup'] = df['Population'] * df['AveOccup']
        df['AveRooms_div_AveOccup'] = df['AveRooms'] / (df['AveOccup'] + 1e-5)
        
        # Geographic features
        df['Location'] = df['Longitude'] * df['Latitude']
        df['Distance_to_center'] = np.sqrt(df['Longitude']**2 + df['Latitude']**2)
        
        # Target capped at 5.0, so flagging high value instances helps the model
        df['is_high_MedInc'] = (df['MedInc'] > 8.0).astype(int)

    # Split data for early stopping validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Training with Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=15,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-4
    )
    
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(test)
    
    # Clip predictions to known target bounds [0.0, 5.0]
    predictions = np.clip(predictions, 0.0, 5.0)
    
    return predictions
