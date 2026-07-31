from __future__ import annotations

import streamlit as st

from predict import predict_loan


st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #eef2ff 0%,
                #f8fafc 50%,
                #ecfeff 100%
            );
        }

        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #475569;
            margin-bottom: 30px;
        }

        .info-card {
            background-color: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
        }

        .result-approved {
            background: linear-gradient(
                135deg,
                #dcfce7,
                #bbf7d0
            );
            padding: 25px;
            border-radius: 16px;
            border-left: 8px solid #16a34a;
            text-align: center;
        }

        .result-rejected {
            background: linear-gradient(
                135deg,
                #fee2e2,
                #fecaca
            );
            padding: 25px;
            border-radius: 16px;
            border-left: 8px solid #dc2626;
            text-align: center;
        }

        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 35px;
            padding: 15px;
        }

        div.stButton > button {
            width: 100%;
            height: 52px;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            );
            color: white;
            border: none;
        }

        div.stButton > button:hover {
            background: linear-gradient(
                90deg,
                #1d4ed8,
                #4338ca
            );
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine Learning based system to estimate
        whether a loan application may be approved or rejected.
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:

    st.header("Project Information")

    st.info(
        """
        Model: Random Forest Classifier

        Dataset: Loan Prediction Dataset

        Features:
        - Missing value handling
        - IQR outlier clipping
        - One-hot encoding
        - Feature scaling
        - Approval probability
        """
    )

    st.warning(
        """
        This application is created for educational
        and internship project purposes only.
        """
    )


st.markdown(
    """
    <div class="info-card">
        <h3>Enter Applicant Details</h3>
        <p>
            Fill all the details below and click
            <b>Predict Loan Status</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.form("loan_prediction_form"):

    column1, column2, column3 = st.columns(3)

    with column1:

        gender = st.selectbox(
            "Gender",
            options=[
                "Male",
                "Female"
            ]
        )

        married = st.selectbox(
            "Married",
            options=[
                "Yes",
                "No"
            ]
        )

        dependents = st.selectbox(
            "Number of Dependents",
            options=[
                "0",
                "1",
                "2",
                "3+"
            ]
        )

        education = st.selectbox(
            "Education",
            options=[
                "Graduate",
                "Not Graduate"
            ]
        )

    with column2:

        self_employed = st.selectbox(
            "Self Employed",
            options=[
                "No",
                "Yes"
            ]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        coapplicant_income = st.number_input(
            "Co-applicant Income",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=1.0,
            value=150.0,
            step=10.0,
            help="Loan amount is generally represented in thousands."
        )

    with column3:

        loan_amount_term = st.selectbox(
            "Loan Amount Term",
            options=[
                12,
                36,
                60,
                84,
                120,
                180,
                240,
                300,
                360,
                480
            ],
            index=8
        )

        credit_history_label = st.selectbox(
            "Credit History",
            options=[
                "Good Credit History",
                "Poor Credit History"
            ]
        )

        property_area = st.selectbox(
            "Property Area",
            options=[
                "Urban",
                "Semiurban",
                "Rural"
            ]
        )

    submit_button = st.form_submit_button(
        "🔍 Predict Loan Status"
    )


if submit_button:

    credit_history = (
        1.0
        if credit_history_label == "Good Credit History"
        else 0.0
    )

    try:

        result = predict_loan(
            gender=gender,
            married=married,
            dependents=dependents,
            education=education,
            self_employed=self_employed,
            applicant_income=applicant_income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_amount_term=float(
                loan_amount_term
            ),
            credit_history=credit_history,
            property_area=property_area
        )

        st.markdown("---")

        if result["prediction"] == 1:

            st.markdown(
                f"""
                <div class="result-approved">
                    <h2>✅ Loan Approved</h2>
                    <p>
                        The machine learning model predicts
                        that this loan may be approved.
                    </p>
                    <h3>
                        Approval Probability:
                        {result["approval_probability"] * 100:.2f}%
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Applicant has a positive loan approval prediction."
            )

        else:

            st.markdown(
                f"""
                <div class="result-rejected">
                    <h2>❌ Loan Rejected</h2>
                    <p>
                        The machine learning model predicts
                        that this loan may be rejected.
                    </p>
                    <h3>
                        Rejection Probability:
                        {result["rejection_probability"] * 100:.2f}%
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.error(
                "Applicant has a negative loan approval prediction."
            )

        st.subheader("Prediction Probabilities")

        probability_column1, probability_column2 = (
            st.columns(2)
        )

        with probability_column1:

            st.metric(
                label="Approval Probability",
                value=(
                    f"{result['approval_probability'] * 100:.2f}%"
                )
            )

            st.progress(
                int(
                    result[
                        "approval_probability"
                    ] * 100
                )
            )

        with probability_column2:

            st.metric(
                label="Rejection Probability",
                value=(
                    f"{result['rejection_probability'] * 100:.2f}%"
                )
            )

            st.progress(
                int(
                    result[
                        "rejection_probability"
                    ] * 100
                )
            )

        with st.expander(
            "View Entered Applicant Details"
        ):

            st.write(
                {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self Employed": self_employed,
                    "Applicant Income": applicant_income,
                    "Co-applicant Income": coapplicant_income,
                    "Loan Amount": loan_amount,
                    "Loan Amount Term": loan_amount_term,
                    "Credit History": credit_history_label,
                    "Property Area": property_area
                }
            )

    except FileNotFoundError as error:

        st.error(str(error))

        st.info(
            "First run this command in terminal: "
            "`python train_model.py`"
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


st.markdown(
    """
    <div class="footer">
        Loan Approval Prediction using Machine Learning
        <br>
        Internship Project
    </div>
    """,
    unsafe_allow_html=True
)