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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from libs.sklearn_extensions.label_encoder import NormalizingLabelEncoder

if __name__ == "__main__":
    # dataset = pd.read_csv("dataset_without_encoding.csv")

    # le = LabelEncoder()

    # for col in dataset.columns:
    #     if dataset[col].dtype == 'object':
    #         dataset[col] = le.fit_transform(dataset[col])
    
    # X = dataset.iloc[:, 1:-2]
    # y = dataset.iloc[:, -1]
    
    # dataset.drop_duplicates(inplace=True)
    # dataset.to_csv("dataset.csv")

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

    # Considering pay vars as binary?????
    # df.loc[df.PAY_1 > 0, 'PAY_1'] = 1
    # df.loc[df.PAY_2 > 0, 'PAY_2'] = 1
    # df.loc[df.PAY_3 > 0, 'PAY_3'] = 1
    # df.loc[df.PAY_4 > 0, 'PAY_4'] = 1
    # df.loc[df.PAY_5 > 0, 'PAY_5'] = 1
    # df.loc[df.PAY_6 > 0, 'PAY_6'] = 1

    X = df.drop('DEFAULT_PAY', axis=1)
    y = df['DEFAULT_PAY']

    # kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    # for train_index, test_index in kfold.split(X,y):
    #     # Split the data into train and test sets
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # model_names = [
    #     f"{model_class.__name__}_model.pkl"
    #     for model_class
    #     in (
    #         RandomForestClassifier,
    #         GradientBoostingClassifier,
    #         KNeighborsClassifier,
    #         GaussianNB,
    #     )
    # ]

    # models = [joblib.load(filename) for filename in model_names]
    # regression_parameters = {
    #     "alpha": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20],
    # }
    # lasso = Lasso()
    # ridge = Ridge()
    dec_tree_params = {'min_weight_fraction_leaf': 0.21707553685422343, 'criterion': 'gini', 'max_depth': 17, 'max_leaf_nodes': 11, 'min_samples_split': 2, 'min_samples_leaf': 8}
    ridge_params = {'alpha': 0.35567001456618497, 'tol': 0.005914268390502595, 'max_iter': 214}
    simple_models = [
        # GaussianNB(),
        # SGDClassifier(),
        # LogisticRegression(),
        RidgeClassifier(class_weight="balanced", **ridge_params),
        # MLPClassifier(),
        # KNeighborsClassifier(),
        DecisionTreeClassifier(**dec_tree_params),
    ]
    estimators = [
        ("gaussian", GaussianNB()),
        ("ridge", RidgeClassifier(class_weight="balanced", **ridge_params)),
        # ("dtree", DecisionTreeClassifier()),
        # ("rf", RandomForestClassifier(n_estimators=155)),
        # ("tree_1", DecisionTreeClassifier(**dec_tree_params)),
        # ("tree_2", DecisionTreeClassifier(**dec_tree_params)),
        ("tree", DecisionTreeClassifier(**dec_tree_params)),
    ]
    models = [
        # RandomForestClassifier(),
        # GradientBoostingClassifier(max_depth=5),
        # AdaBoostClassifier(),
        # BaggingClassifier(),
        # StackingClassifier(estimators=estimators),
        VotingClassifier(
            estimators=estimators,
        ),
    ]

    y_test = y_test.to_numpy()
    plt.figure(figsize=(10, 10))
    for model in simple_models + models:
        model.fit(X_train, y_train)
        # joblib.dump(model, f"{type(model).__name__}_model.pkl", compress=9)
        predictions = model.predict(X_test)
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
        if hasattr(model, "predict_proba"):
            pred_scr = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, pred_scr)
            roc_auc = roc_auc_score(y_test, pred_scr)
            md = str(model)
            md = md[:md.find('(')]
            pl.plot(fpr, tpr, label='ROC fold %s (auc = %0.2f)' % (md, roc_auc))

    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    pl.xlim([0, 1])
    pl.ylim([0, 1])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show(block=True)