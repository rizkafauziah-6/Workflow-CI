import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dagshub

# Inisialisasi DagsHub
dagshub.init(repo_owner='rizkafauziah-6', repo_name='Eksperimen_SML_RizkaAulia', mlflow=True)

# Load Data
df = pd.read_csv('dataset_preprocessing/curn_clean_preprocessing.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Eksperimen_Basic_Churn")

with mlflow.start_run():
    # Gunakan Autolog untuk kriteria basic
    mlflow.sklearn.autolog()

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    print("Model basic berhasil dilatih dan dicatat di DagsHub.")