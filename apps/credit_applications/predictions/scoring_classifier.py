from sklearn.neighbors import KNeighborsClassifier
import joblib
from ..models import CreditApplication
from typing import Any
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_classifier() -> KNeighborsClassifier:
    """Load pre-trained classifier."""
    return joblib.load("KNeighborsClassifier_model.pkl")


def load_label_encoders(dataset_path: Path) -> list[LabelEncoder]:
    """Load label encoders for current classification dataset."""
    dataset = pd.read_csv(dataset_path)
    return [LabelEncoder().fit(dataset[col]) for col in dataset.columns[1:-1]]


def credit_application_to_dataframe(application: CreditApplication) -> pd.DataFrame:
    """Transform credit application to list."""
    return pd.DataFrame([[
        application.gender,
        application.own_car,
        application.own_realty,
        application.children_count,
        application.annual_income,
        application.income_type,
        application.education_level,
        application.family_status,
        application.housing_type,
        application.bday_shift,
        application.days_employed,
        application.has_mobile,
        application.has_work_phone,
        application.has_phone,
        application.has_email,
        application.occupation_type,
        application.family_members_count,
    ]])