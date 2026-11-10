# ===== 1. Importer les bibliothèques =====

import mlflow
import mlflow.sklearn


# ===== 2. Se connecter à la base du Registry =====

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Charger le champion depuis le Registry =====

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# ===== 4. Exporter le modèle dans un dossier local =====

# serialization_format="cloudpickle" :
# Même format qu'aux TP3 et TP4.
# Ce dossier sera copié dans l'image Docker.
# L'image n'aura donc plus besoin d'accéder au Registry.

mlflow.sklearn.save_model(
    model,
    "model",
    serialization_format="cloudpickle"
)


# ===== 5. Confirmer l'exportation =====

print("Champion exporté dans le dossier model/")