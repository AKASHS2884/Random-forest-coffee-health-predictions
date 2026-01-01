import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nColumn Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    return df

def clean_data(df):
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            if len(df[col].mode()) > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    return df

def engineer_features(df):
    if 'Caffeine_mg' in df.columns and 'BMI' in df.columns:
        estimated_weight = df['BMI'] * (1.7 ** 2)
        df['caffeine_per_kg'] = df['Caffeine_mg'] / estimated_weight
    
    if 'Coffee_Intake' in df.columns and 'Caffeine_mg' in df.columns:
        df['coffee_intensity'] = df['Coffee_Intake'] * df['Caffeine_mg']
    
    if 'Age' in df.columns:
        df['age_group'] = pd.cut(df['Age'], 
                                  bins=[0, 25, 35, 45, 55, 100], 
                                  labels=[0, 1, 2, 3, 4])
        df['age_group'] = df['age_group'].fillna(2).astype(int)
    
    if 'Sleep_Hours' in df.columns:
        df['sleep_deficit'] = (8 - df['Sleep_Hours']).clip(lower=0)
    
    if 'Smoking' in df.columns and 'Alcohol_Consumption' in df.columns:
        df['health_risk_score'] = df['Smoking'] + df['Alcohol_Consumption']
    
    if 'Physical_Activity_Hours' in df.columns:
        df['activity_level'] = pd.cut(df['Physical_Activity_Hours'],
                                       bins=[-0.1, 3, 7, 12, 20],
                                       labels=[0, 1, 2, 3])
        df['activity_level'] = df['activity_level'].fillna(1).astype(int)
    
    return df

def encode_categorical_features(df, target_column):
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != target_column]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    return df, label_encoders

def prepare_features_target(df, target_column):
    if df[target_column].dtype == 'object':
        stress_map = {'Low': 1, 'Medium': 2, 'High': 3}
        df[target_column] = df[target_column].map(stress_map)
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    return X, y

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='neg_mean_squared_error')
    print(f"Cross-validation RMSE: {np.sqrt(-cv_scores.mean()):.4f}")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"\nTraining Set:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Set:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return {
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'feature_importance': feature_importance
    }

def save_model_and_artifacts(model, scaler, feature_columns, metrics, label_encoders):
    output_dir = os.path.dirname(__file__)
    
    joblib.dump(model, os.path.join(output_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))
    joblib.dump(metrics, os.path.join(output_dir, 'model_metrics.pkl'))
    joblib.dump(label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
    
    print("\nAll artifacts saved successfully")

def main():
    print("Starting Coffee Health Prediction Model Training\n")
    
    data_path = os.path.join('..', 'dataset', 'global_coffee_health_data.csv')
    target_column = 'Stress_Level'
    
    print("Step 1: Loading data...")
    df = load_and_explore_data(data_path)
    
    print("\nStep 2: Cleaning data...")
    df = clean_data(df)
    
    print("\nStep 3: Engineering features...")
    df = engineer_features(df)
    
    print("\nStep 4: Encoding categorical features...")
    df, label_encoders = encode_categorical_features(df, target_column)
    
    print("\nStep 5: Preparing features and target...")
    X, y = prepare_features_target(df, target_column)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    print("\nStep 6: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    print("\nStep 7: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print("\nStep 8: Training model...")
    model = train_random_forest(X_train_scaled, y_train)
    
    print("\nStep 9: Evaluating model...")
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\nStep 10: Saving artifacts...")
    save_model_and_artifacts(model, scaler, X_train.columns.tolist(), metrics, label_encoders)
    
    print("\n" + "="*50)
    print("Training completed successfully")
    print("="*50)

if __name__ == "__main__":
    main()
