from __future__ import annotations

import pandas as pd
import plotly.express as px

from config import DATA_PATH, PLOTS_DIR, TARGET


def load_dataset() -> pd.DataFrame:
    """
    Load the loan prediction dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset file not found:\n{DATA_PATH}\n\n"
            "loan_prediction.csv file ni data folder lo pettandi."
        )

    dataframe = pd.read_csv(DATA_PATH)

    return dataframe


def save_chart(figure, filename: str) -> None:
    """
    Save a Plotly chart as an HTML file.
    """

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PLOTS_DIR / filename

    figure.write_html(output_path)

    print(f"Chart saved: {output_path}")


def print_dataset_information(dataframe: pd.DataFrame) -> None:
    """
    Print basic dataset information.
    """

    print("\n========================================")
    print("       LOAN DATASET INFORMATION")
    print("========================================")

    print("\n1. Dataset shape:")
    print(dataframe.shape)

    print("\n2. Column names:")
    print(dataframe.columns.tolist())

    print("\n3. First five rows:")
    print(dataframe.head())

    print("\n4. Last five rows:")
    print(dataframe.tail())

    print("\n5. Dataset data types:")
    print(dataframe.dtypes)

    print("\n6. Missing values:")
    print(dataframe.isnull().sum())

    print("\n7. Duplicate rows:")
    print(dataframe.duplicated().sum())

    print("\n8. Numerical summary:")
    print(dataframe.describe())

    categorical_columns = dataframe.select_dtypes(
        include="object"
    ).columns

    if len(categorical_columns) > 0:
        print("\n9. Categorical summary:")
        print(
            dataframe[categorical_columns].describe()
        )

    if TARGET in dataframe.columns:
        print("\n10. Loan status distribution:")
        print(
            dataframe[TARGET].value_counts()
        )


def create_loan_status_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create loan approval status distribution chart.
    """

    if TARGET not in dataframe.columns:
        print(
            f"Skipping loan status chart. "
            f"Column '{TARGET}' not found."
        )
        return

    status_data = (
        dataframe[TARGET]
        .map({
            "Y": "Approved",
            "N": "Denied"
        })
        .value_counts()
        .rename_axis("Loan Status")
        .reset_index(name="Count")
    )

    figure = px.bar(
        status_data,
        x="Loan Status",
        y="Count",
        text="Count",
        title="Loan Approval Status Distribution"
    )

    figure.update_traces(
        textposition="outside"
    )

    save_chart(
        figure,
        "01_loan_status_distribution.html"
    )


def create_gender_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create gender-wise loan status chart.
    """

    required_columns = {
        "Gender",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="Gender",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Gender",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "02_gender_analysis.html"
    )


def create_married_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create marital-status loan chart.
    """

    required_columns = {
        "Married",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="Married",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Marital Status",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "03_marital_status_analysis.html"
    )


def create_dependents_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create dependents-wise loan status chart.
    """

    required_columns = {
        "Dependents",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="Dependents",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Number of Dependents",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "04_dependents_analysis.html"
    )


def create_education_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create education-wise loan status chart.
    """

    required_columns = {
        "Education",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="Education",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Education",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "05_education_analysis.html"
    )


def create_self_employed_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create employment-wise loan status chart.
    """

    required_columns = {
        "Self_Employed",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="Self_Employed",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Employment",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "06_self_employed_analysis.html"
    )


def create_property_area_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create property-area loan status chart.
    """

    required_columns = {
        "Property_Area",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="Property_Area",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Property Area",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "07_property_area_analysis.html"
    )


def create_applicant_income_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create applicant-income distribution chart.
    """

    required_columns = {
        "ApplicantIncome",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="ApplicantIncome",
        color=TARGET,
        nbins=40,
        marginal="box",
        title="Applicant Income Distribution",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "08_applicant_income_distribution.html"
    )


def create_coapplicant_income_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create co-applicant-income distribution chart.
    """

    required_columns = {
        "CoapplicantIncome",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.histogram(
        dataframe,
        x="CoapplicantIncome",
        color=TARGET,
        nbins=40,
        marginal="box",
        title="Co-applicant Income Distribution",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "09_coapplicant_income_distribution.html"
    )


def create_loan_amount_boxplot(
    dataframe: pd.DataFrame
) -> None:
    """
    Create loan-amount box plot for outlier analysis.
    """

    required_columns = {
        "LoanAmount",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    figure = px.box(
        dataframe,
        x=TARGET,
        y="LoanAmount",
        points="outliers",
        title="Loan Amount Outlier Analysis",
        labels={
            TARGET: "Loan Status",
            "LoanAmount": "Loan Amount"
        }
    )

    save_chart(
        figure,
        "10_loan_amount_boxplot.html"
    )


def create_credit_history_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create credit-history loan status chart.
    """

    required_columns = {
        "Credit_History",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    temporary_dataframe = dataframe.copy()

    temporary_dataframe[
        "Credit_History_Label"
    ] = temporary_dataframe[
        "Credit_History"
    ].map({
        1.0: "Good Credit History",
        0.0: "Poor Credit History"
    })

    figure = px.histogram(
        temporary_dataframe,
        x="Credit_History_Label",
        color=TARGET,
        barmode="group",
        title="Loan Approval Status by Credit History",
        labels={
            TARGET: "Loan Status",
            "Credit_History_Label": "Credit History"
        }
    )

    save_chart(
        figure,
        "11_credit_history_analysis.html"
    )


def create_total_income_chart(
    dataframe: pd.DataFrame
) -> None:
    """
    Create total-income feature and chart.
    """

    required_columns = {
        "ApplicantIncome",
        "CoapplicantIncome",
        TARGET
    }

    if not required_columns.issubset(
        dataframe.columns
    ):
        return

    temporary_dataframe = dataframe.copy()

    temporary_dataframe[
        "TotalIncome"
    ] = (
        temporary_dataframe[
            "ApplicantIncome"
        ].fillna(0)
        +
        temporary_dataframe[
            "CoapplicantIncome"
        ].fillna(0)
    )

    figure = px.histogram(
        temporary_dataframe,
        x="TotalIncome",
        color=TARGET,
        nbins=40,
        marginal="box",
        title="Total Household Income Distribution",
        labels={
            TARGET: "Loan Status"
        }
    )

    save_chart(
        figure,
        "12_total_income_distribution.html"
    )


def create_correlation_heatmap(
    dataframe: pd.DataFrame
) -> None:
    """
    Create correlation heatmap for numerical columns.
    """

    numerical_dataframe = (
        dataframe.select_dtypes(
            include="number"
        )
    )

    if numerical_dataframe.empty:
        return

    correlation_matrix = (
        numerical_dataframe.corr(
            numeric_only=True
        )
    )

    figure = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Numerical Feature Correlation Heatmap"
    )

    save_chart(
        figure,
        "13_correlation_heatmap.html"
    )


def calculate_iqr_outliers(
    dataframe: pd.DataFrame
) -> None:
    """
    Display IQR outlier count for numerical columns.
    """

    print("\n========================================")
    print("           IQR OUTLIER REPORT")
    print("========================================")

    numerical_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term"
    ]

    for column in numerical_columns:

        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            print(
                f"\n{column}: IQR is 0, "
                "outlier clipping is skipped."
            )
            continue

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outlier_count = (
            (
                values < lower_limit
            )
            |
            (
                values > upper_limit
            )
        ).sum()

        print(f"\nColumn: {column}")
        print(f"Q1: {q1:.2f}")
        print(f"Q3: {q3:.2f}")
        print(f"IQR: {iqr:.2f}")
        print(
            f"Lower limit: {lower_limit:.2f}"
        )
        print(
            f"Upper limit: {upper_limit:.2f}"
        )
        print(
            f"Outlier count: {outlier_count}"
        )


def run_eda() -> None:
    """
    Run complete exploratory data analysis.
    """

    dataframe = load_dataset()

    print_dataset_information(
        dataframe
    )

    calculate_iqr_outliers(
        dataframe
    )

    print("\n========================================")
    print("         CREATING PLOTLY CHARTS")
    print("========================================")

    create_loan_status_chart(
        dataframe
    )

    create_gender_chart(
        dataframe
    )

    create_married_chart(
        dataframe
    )

    create_dependents_chart(
        dataframe
    )

    create_education_chart(
        dataframe
    )

    create_self_employed_chart(
        dataframe
    )

    create_property_area_chart(
        dataframe
    )

    create_applicant_income_chart(
        dataframe
    )

    create_coapplicant_income_chart(
        dataframe
    )

    create_loan_amount_boxplot(
        dataframe
    )

    create_credit_history_chart(
        dataframe
    )

    create_total_income_chart(
        dataframe
    )

    create_correlation_heatmap(
        dataframe
    )

    print("\n========================================")
    print("EDA COMPLETED SUCCESSFULLY")
    print("========================================")

    print(
        f"\nCharts saved inside:\n{PLOTS_DIR}"
    )


if __name__ == "__main__":
    run_eda()