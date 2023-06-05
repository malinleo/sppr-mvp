import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from libs.sklearn_extensions.label_encoder import NormalizingLabelEncoder

if __name__ == "__main__":

    df = pd.read_csv("dataset_2.csv", delimiter=";")
    df.rename(columns={'PAY_0':'PAY_1'}, inplace=True)
    fil = (df.EDUCATION == 5) | (df.EDUCATION == 6) | (df.EDUCATION == 0)
    df.loc[fil, 'EDUCATION'] = 4
    df.EDUCATION.value_counts()
    df.loc[df.MARRIAGE == 0, 'MARRIAGE'] = 3
    df.MARRIAGE.value_counts()

    fil = (df.PAY_1 == -2) | (df.PAY_1 == -1) | (df.PAY_1 == 0)
    df.loc[fil, 'PAY_1'] = 0
    fil = (df.PAY_2 == -2) | (df.PAY_2 == -1) | (df.PAY_2 == 0)
    df.loc[fil, 'PAY_2'] = 0
    fil = (df.PAY_3 == -2) | (df.PAY_3 == -1) | (df.PAY_3 == 0)
    df.loc[fil, 'PAY_3'] = 0
    fil = (df.PAY_4 == -2) | (df.PAY_4 == -1) | (df.PAY_4 == 0)
    df.loc[fil, 'PAY_4'] = 0
    fil = (df.PAY_5 == -2) | (df.PAY_5 == -1) | (df.PAY_5 == 0)
    df.loc[fil, 'PAY_5'] = 0
    fil = (df.PAY_6 == -2) | (df.PAY_6 == -1) | (df.PAY_6 == 0)
    df.loc[fil, 'PAY_6'] = 0

    X = df.drop('DEFAULT_PAY', axis=1)
    y = df['DEFAULT_PAY']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    # dec_tree_params = {'min_weight_fraction_leaf': 0.21707553685422343, 'criterion': 'gini', 'max_depth': 17, 'max_leaf_nodes': 11, 'min_samples_split': 2, 'min_samples_leaf': 8}
    ridge_params = {'alpha': 0.35567001456618497, 'tol': 0.005914268390502595, 'max_iter': 214}
    simple_models = [
        # GaussianNB(),
        # SGDClassifier(),
        # SVC(),
        # LogisticRegression(),
        # RidgeClassifier(class_weight="balanced"),
        # MLPClassifier(hidden_layer_sizes=(100, 64, 32)),
        # KNeighborsClassifier(),
        # DecisionTreeClassifier(),
    ]
    estimators = [
        ("gaussian", GaussianNB()),
        ("ridge", RidgeClassifier(class_weight="balanced")),
        # ("dtree", DecisionTreeClassifier()),
        # ("rf", RandomForestClassifier(n_estimators=155)),
        # ("tree_1", DecisionTreeClassifier(**dec_tree_params)),
        # ("tree_2", DecisionTreeClassifier(**dec_tree_params)),
        ("tree", DecisionTreeClassifier()),
    ]
    bagging_params = {'n_estimators': 224, 'max_samples': 895, 'max_features': 20, 'bootstrap_features': True}
    models = [
        # RandomForestClassifier(),
        # GradientBoostingClassifier(max_depth=5),
        # AdaBoostClassifier(),
        BaggingClassifier(
            estimator=RidgeClassifier(class_weight="balanced"),
            **bagging_params,
        ),
    #     StackingClassifier(
    #         estimators=estimators,
    #         final_estimator=RidgeClassifier(class_weight="balanced"),
    #     ),
    #     VotingClassifier(
    #         estimators=estimators,
    #     ),
    ]

    y_test = y_test.to_numpy()
    for model in models:
        scaled = make_pipeline(StandardScaler(), model)
        scaled.fit(X_train, y_train)
        joblib.dump(model, f"{type(model).__name__}_model.pkl", compress=9)
        predictions = scaled.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print(f"Model {type(model).__name__}")
        print(
            f"Accuracy score {acc}",
            f"F1 score: {f1}",
            sep="\n",
        )
