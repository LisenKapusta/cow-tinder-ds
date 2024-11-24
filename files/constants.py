from pathlib import Path
import joblib

current_path = Path(__file__).resolve()

ROOT_DIRECTORY = current_path.parent.parent


MODEL_GROWTH_PATH = ROOT_DIRECTORY / "models" / "model_growth.pkl"
MODEL_GENETIC_VALU_PATH = ROOT_DIRECTORY / "models" / "model_genetic_value.pkl"
MODEL_UPITANNOST_PATH = ROOT_DIRECTORY / "models" / "model_upitannost.pkl"
MODEL_INBREEDING_PATH = ROOT_DIRECTORY / "models" / "model_inbreeding.pkl"
MODEL_HEALTH_PATH = ROOT_DIRECTORY / "models" / "model_health.pkl"
MODEL_FERTILITY_PATH = ROOT_DIRECTORY / "models" / "model_fertility.pkl"
MODEL_UDOY_PATH = ROOT_DIRECTORY / "models" / 'model_udoy.pkl'


MERGED_DF_PATH = ROOT_DIRECTORY / "datasets" / "merged_df.csv"
DF_ANIMALS_PATH = ROOT_DIRECTORY / "datasets" / "df_animals.csv"