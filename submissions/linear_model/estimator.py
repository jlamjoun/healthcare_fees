import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def get_estimator():
    # Colonnes catégorielles et numériques
    categorical_cols = ["annee", "profession_sante", "region", "departement"]
    numerical_cols = []  # Il n'y a pas de colonnes numériques dans vos données

    # Création du préprocesseur pour transformer les colonnes
    preprocessor = make_column_transformer(
        (OneHotEncoder(), categorical_cols),  # OneHotEncoder pour les colonnes catégorielles
        remainder='passthrough'  # Pas de transformation pour les colonnes numériques
    )

    # Modèle de régression RandomForest
    regressor = LinearRegression()

    # Construction de la pipeline
    pipeline = make_pipeline(preprocessor, regressor)

    return pipeline
