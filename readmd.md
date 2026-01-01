# Coffee Health Prediction System

Predict stress levels based on coffee consumption and lifestyle factors using a machine learning model served through a Django web app.

## 1. Project Structure

CoffeeHealthProject/
│
├── dataset/
│ └── global_coffee_health_data.csv # Provided dataset
│
├── ml_logic/
│ ├── train_model.py # End‑to‑end training pipeline
│ ├── random_forest_model.pkl # Trained model (created after training)
│ ├── scaler.pkl # Feature scaler
│ ├── feature_columns.pkl # Feature names used during training
│ ├── label_encoders.pkl # Encoders for categorical features
│ └── model_metrics.pkl # Saved evaluation metrics
│
├── coffee_app/
│ ├── manage.py
│ ├── coffee_app/
│ │ ├── init.py
│ │ ├── settings.py
│ │ ├── urls.py
│ │ ├── wsgi.py
│ │ └── asgi.py
│ │
│ └── api/
│ ├── init.py
│ ├── apps.py
│ ├── urls.py
│ ├── views.py
│ ├── serializers.py
│ ├── templates/
│ │ └── index.html # Web UI
│ └── static/
│ └── style.css # UI styling
│
├── venv/ # Python virtualenv (can be recreated)
└── requirements.txt

text

## 2. Prerequisites

- Python 3.11 (or any compatible 3.10+ version)
- pip

Use PowerShell or Command Prompt on Windows.

## 3. Setup Instructions

### 3.1. Clone or unzip the project

Unzip the project and note the folder path, for example:

C:\Users<you>\Desktop\CoffeeHealthProject

text

Open a terminal in that folder.

### 3.2. Create and activate virtual environment

If `venv/` is not already present or you want a fresh environment:

python -m venv venv
.\venv\Scripts\activate

text

You should see `(venv)` at the start of your prompt.

### 3.3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

text

## 4. Train the Machine Learning Model

This step reads `dataset/global_coffee_health_data.csv`, trains the RandomForest model, evaluates it, and saves all artifacts into `ml_logic/`.

cd ml_logic
python train_model.py

text

You should see logs ending with:

All artifacts saved successfully
Training completed successfully
text

Files created or updated in `ml_logic/`:

- `random_forest_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`
- `label_encoders.pkl`
- `model_metrics.pkl`

If these files already exist and you trust them, you can skip retraining.

## 5. Run the Django Web Application

From the project root:

cd coffee_app
..\venv\Scripts\activate # if not already active
python manage.py migrate # one‑time DB setup
python manage.py runserver

text

You should see:

Django version 4.2.x, using settings 'coffee_app.settings'
Starting development server at http://127.0.0.1:8000/

text

Open a browser and go to:  
`http://127.0.0.1:8000/`

## 6. Using the App

On the **Coffee Health Prediction System** page:

1. Fill in all fields:
   - Age  
   - Gender  
   - Country  
   - Coffee Cups Per Day  
   - Caffeine (mg)  
   - Sleep Hours  
   - Sleep Quality (Poor / Fair / Good / Excellent)  
   - BMI  
   - Heart Rate  
   - Exercise Hours/Week  
   - Health Issues (None / Mild / Severe)  
   - Occupation (Student / Office / Service / Healthcare / Other)  
   - Smoking (Yes / No)  
   - Alcohol (Yes / No)

2. Click **Predict Stress Level**.

The app will display:
- Predicted numeric stress score
- Stress category (Low / Medium / High)
- Model confidence percentage
- A text recommendation

The dataset is synthetic and encodes a strong relationship between sleep quality and stress level, so predictions are very consistent with that variable. [file:22]

## 7. Restarting After Shutdown

Any time you reboot or close the terminal:

cd C:\Users<you>\Desktop\CoffeeHealthProject
.\venv\Scripts\activate
cd coffee_app
python manage.py runserver

text

Then open `http://127.0.0.1:8000/` again.

## 8. Common Issues

- **ModuleNotFoundError (Django or other libraries)**  
  Make sure the virtual environment is active and dependencies are installed:

.\venv\Scripts\activate
pip install -r requirements.txt

text

- **Dataset missing**  
Ensure `dataset/global_coffee_health_data.csv` exists. If not, place the CSV there before running `train_model.py`. [file:22]

- **Port already in use**  
If `runserver` reports port 8000 in use:

python manage.py runserver 8001

text

Then open `http://127.0.0.1:8001/`.

## 9. Tech Stack

- Python, pandas, NumPy, scikit‑learn [file:1]
- Django 4.2, Django REST Framework [file:1]
- HTML/CSS frontend with vanilla JavaScript
- RandomForestRegressor model with engineered lifestyle features based on the Global Coffee Health Dataset [file:22]