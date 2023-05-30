from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder


from ..models import CreditApplication


def load_classifier() -> VotingClassifier:
    """Load pre-trained classifier."""
    return joblib.load("VotingClassifier_model.pkl")


def load_label_encoders(dataset_path: Path) -> list[LabelEncoder]:
    """Load label encoders for current classification dataset."""
    dataset = pd.read_csv(dataset_path)
    return [
        LabelEncoder().fit(dataset[col])
        for col in dataset.columns[1:-1]
    ]


def credit_application_to_dataframe(application: CreditApplication) -> pd.DataFrame:
    """Transform credit application to list."""
    return pd.DataFrame([[
        application.applicant.id,
        application.limit_balance,
        application.gender,
        application.education_level,
        application.marriage,
        application.age,
        application.pay_1,
        application.pay_2,
        application.pay_3,
        application.pay_4,
        application.pay_5,
        application.pay_6,
        application.bill_amt_1,
        application.bill_amt_2,
        application.bill_amt_3,
        application.bill_amt_4,
        application.bill_amt_5,
        application.bill_amt_6,
        application.pay_amt_1,
        application.pay_amt_2,
        application.pay_amt_3,
        application.pay_amt_4,
        application.pay_amt_5,
        application.pay_amt_6,
    ]])