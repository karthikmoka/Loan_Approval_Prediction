from __future__ import annotations

import joblib
import pandas as pd

from config import MODEL_PATH


def load_model():
    """
    Load the trained loan approval model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nModel file not found:\n{MODEL_PATH}\n\n"
            "First run: python train_model.py"
        )

    model = joblib.load(MODEL_PATH)

    return model


def predict_loan(
    gender: str,
    married: str,
    dependents: str,
    education: str,
    self_employed: str,
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_amount_term: float,
    credit_history: float,
    property_area: str
):
    """
    Predict loan approval for one applicant.
    """

    model = load_model()

    applicant_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }
        ]
    )

    prediction = model.predict(
        applicant_data
    )[0]

    probability = model.predict_proba(
        applicant_data
    )[0]

    rejection_probability = probability[0]
    approval_probability = probability[1]

    result = (
        "Loan Approved"
        if prediction == 1
        else "Loan Rejected"
    )

    return {
        "prediction": int(prediction),
        "result": result,
        "approval_probability": float(
            approval_probability
        ),
        "rejection_probability": float(
            rejection_probability
        )
    }


def display_prediction(
    result: dict
) -> None:
    """
    Display prediction result.
    """

    print("\n========================================")
    print("          LOAN PREDICTION RESULT")
    print("========================================")

    print(
        f"\nFinal Prediction: {result['result']}"
    )

    print(
        "Approval Probability: "
        f"{result['approval_probability'] * 100:.2f}%"
    )

    print(
        "Rejection Probability: "
        f"{result['rejection_probability'] * 100:.2f}%"
    )

    print("\n========================================")


def run_sample_prediction() -> None:
    """
    Run prediction using sample applicant data.
    """

    sample_result = predict_loan(
        gender="Male",
        married="Yes",
        dependents="1",
        education="Graduate",
        self_employed="No",
        applicant_income=5000,
        coapplicant_income=2000,
        loan_amount=150,
        loan_amount_term=360,
        credit_history=1,
        property_area="Semiurban"
    )

    display_prediction(
        sample_result
    )


if __name__ == "__main__":
    run_sample_prediction()