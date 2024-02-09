import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

def get_estimator():

    categorical_encoder = OrdinalEncoder()
    categorical_cols = ["annee", "profession_sante", "region", "departement"]

    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
        remainder='passthrough',  # passthrough numerical columns as they are
    )

    regressor = RandomForestRegressor(
        n_estimators=10, max_depth=10, max_features=10, n_jobs=4
    )

    return make_pipeline(preprocessor, regressor)
