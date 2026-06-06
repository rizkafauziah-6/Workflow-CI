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

    # Log model secara manual ke DagsHub (tetap dicoba secara online)
    try:
        mlflow.sklearn.log_model(rf, "model")
    except Exception as e:
        print(f"Warning: Gagal log model ke DagsHub: {e}")

    # Simpan model secara lokal di path GITHUB_WORKSPACE agar persisten dan bisa digunakan oleh Docker build
    import os
    import shutil
    workspace_dir = os.environ.get("GITHUB_WORKSPACE", "..")
    local_model_path = os.path.join(workspace_dir, "local_model")
    if os.path.exists(local_model_path):
        shutil.rmtree(local_model_path)
    mlflow.sklearn.save_model(rf, local_model_path)
    print(f"Model basic berhasil dilatih, disimpan lokal di {local_model_path}, dan dicatat di DagsHub.")