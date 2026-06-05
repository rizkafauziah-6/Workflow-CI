import mlflow
import subprocess
import sys

def main():
    print("Mencari run terbaru dari eksperimen...")
    # Set tracking URI ke DagsHub (sudah dikonfigurasi lewat env var oleh GitHub Actions)
    # Cari run terbaru dari eksperimen "Eksperimen_Basic_Churn"
    experiment_name = "Eksperimen_Basic_Churn"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Eksperimen '{experiment_name}' tidak ditemukan!")
        sys.exit(1)
        
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        print("Tidak ada run yang ditemukan dalam eksperimen ini!")
        sys.exit(1)
        
    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/model"
    image_name = "alya6/churn-prediction-mlops:latest"
    
    print(f"Run ID Terakhir: {latest_run_id}")
    print(f"Model URI: {model_uri}")
    print(f"Mulai melakukan build Docker image dengan nama: {image_name}...")
    
    # Menjalankan mlflow models build-docker
    # Kita menggunakan --env-manager=local agar tidak memerlukan Conda pada saat build image di runner
    build_cmd = [
        "mlflow", "models", "build-docker",
        "-m", model_uri,
        "-n", image_name,
        "--env-manager", "local"
    ]
    
    try:
        subprocess.run(build_cmd, check=True)
        print("Docker image berhasil dibangun!")
    except subprocess.CalledProcessError as e:
        print(f"Gagal membangun Docker image: {e}")
        sys.exit(1)
        
    print(f"Melakukan push Docker image '{image_name}' ke Docker Hub...")
    try:
        subprocess.run(["docker", "push", image_name], check=True)
        print("Docker image berhasil di-push ke Docker Hub!")
    except subprocess.CalledProcessError as e:
        print(f"Gagal melakukan push Docker image ke Docker Hub: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
