from dataclasses import dataclass

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
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
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Categorical, Continuous, Integer
from sklearn_genetic.space.base import BaseDimension


class ModelWithTuneParams:
    model: BaseEstimator
    params: dict[str, BaseDimension]


@dataclass
class RidgeWithTuneParams(ModelWithTuneParams):
    model = RidgeClassifier(class_weight="balanced")
    params = {
        "alpha": Continuous(.1, 10.0, distribution="log-uniform"),
        "tol": Continuous(1e-4, 1.0, distribution='log-uniform'),
        "max_iter": Integer(200, 300),
    }


@dataclass
class DecisionTreeWithTuneParams(ModelWithTuneParams):
    model = DecisionTreeClassifier()
    params = {
        'min_weight_fraction_leaf': Continuous(0, 0.5),
        'criterion': Categorical(['gini', 'entropy']),
        'max_depth': Integer(2, 20),
        'max_leaf_nodes': Integer(2, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
    }


@dataclass
class RandomForestWithTuneParams(ModelWithTuneParams):
    model = RandomForestClassifier()
    params = {
        "n_estimators": Integer(50, 200),
        "criterion": Categorical(['gini', 'entropy']),
        'max_depth': Integer(2, 20),
        'max_leaf_nodes': Integer(2, 20),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        "max_features": Categorical(["sqrt", "log2"]),
    }


@dataclass
class GradientBoostingWithTuneParams(ModelWithTuneParams):
    model = GradientBoostingClassifier()
    params = {
        "loss": Categorical(["log_loss", "exponential"]),
        "n_estimators": Integer(100, 300),
        "learning_rate": Continuous(0.1, 1.0, distribution="log-uniform"),
        'max_depth': Integer(2, 30),
        'max_leaf_nodes': Integer(2, 35),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 20),
        "max_features": Categorical(["sqrt", "log2"]),
    }


@dataclass
class VotingWithTuneParams(ModelWithTuneParams):
    model = VotingClassifier(
        estimators=[
            ("dtree", DecisionTreeClassifier()),
            ("rf", RandomForestClassifier()),
            ("grad_boost", GradientBoostingClassifier(max_depth=4)),
        ],
    )
    params = {
        "voting": Categorical(["soft", "hard"]),
        "weights": Categorical([[0.2, 0.35, 0.45]]),
    }


if __name__ == "__main__":
    df = pd.read_csv("dataset_2.csv", delimiter=";")
    df.drop("ID", axis=1)
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
        test_size=0.2,
        random_state=42,
    )

    simple_models: list[ModelWithTuneParams] = [
        RidgeWithTuneParams(),
        # DecisionTreeWithTuneParams(),
    ]
    estimators = [
        ("gaussian", GaussianNB()),
        ("ridge", RidgeClassifier()),
        ("dtree", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier()),
        ("grad_boost", GradientBoostingClassifier(max_depth=4)),
    ]
    models: list[ModelWithTuneParams] = [
        # RandomForestWithTuneParams(),
        # GradientBoostingWithTuneParams(),
        # AdaBoostClassifier(),
        # BaggingClassifier(),
        # StackingClassifier(estimators=estimators),
        # VotingWithTuneParams(),
    ]
    kfold_cross_validation = StratifiedKFold(n_splits=3, shuffle=True)

    for model in simple_models + models:
        evolved_estimator = GASearchCV(
            estimator=model.model,
            cv=kfold_cross_validation,
            scoring='f1',
            param_grid=model.params,
            n_jobs=-1,
            verbose=True,
            population_size=10,
            generations=20
        )
        evolved_estimator.fit(X_train, y_train)
        joblib.dump(evolved_estimator.estimator, f"{type(evolved_estimator.estimator).__name__}_model.pkl", compress=9)
        print(type(evolved_estimator.estimator).__name__)
        print(evolved_estimator.best_params_)
        y_predict_ga = evolved_estimator.predict(X_test)
        print(accuracy_score(y_test, y_predict_ga))
    plot_fitness_evolution(evolved_estimator)
    plt.show()