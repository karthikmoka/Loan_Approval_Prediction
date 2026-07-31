from __future__ import annotations

from data_analysis import run_eda
from train_model import train_model
from predict import run_sample_prediction


def main():
    """
    Run complete Loan Approval Prediction project.
    """

    print("\n========================================")
    print("   LOAN APPROVAL PREDICTION PROJECT")
    print("========================================")

    print("\nStep 1: Exploratory Data Analysis")
    run_eda()

    print("\nStep 2: Model Training")
    train_model()

    print("\nStep 3: Sample Prediction")
    run_sample_prediction()

    print("\n========================================")
    print("      PROJECT COMPLETED SUCCESSFULLY")
    print("========================================")

    print("\nTo open the website, run:")

    print("\nstreamlit run app.py")


if __name__ == "__main__":
    main()