from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "loan_prediction.csv"

MODEL_PATH = BASE_DIR / "models" / "loan_model.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"

PLOTS_DIR = OUTPUT_DIR / "plots"

TARGET = "Loan_Status"

RANDOM_STATE = 42

TEST_SIZE = 0.20