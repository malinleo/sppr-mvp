import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # application_record = pd.read_csv("./application_record.csv")
    # credit_record = pd.read_csv("./credit_record.csv")

    # credit_record.replace(['X','C'], 0,inplace=True)
    # credit_record.STATUS = pd.to_numeric(credit_record.STATUS)
    # # Searching for customers who have at least one late month
    # drop_ls = []
    # for i in range(len(credit_record)):
    #     if credit_record.STATUS[i] != 0:
    #         credit_record.STATUS[i] = 1
    
    # credit_record.drop_duplicates(inplace=True)
    # dataset = application_record.merge(credit_record, on=['ID'], how='inner')
    # dataset.drop(['ID'],inplace=True,axis=1)
    # dataset.drop_duplicates(inplace=True)

    # dataset.OCCUPATION_TYPE.replace(np.nan, 'Other', inplace = True)

    # dataset.to_csv("dataset_without_encoding.csv")

    # le = LabelEncoder()

    # for col in dataset.columns:
    #     if dataset[col].dtype == 'object':
    #         dataset[col] = le.fit_transform(dataset[col])
    
    # dataset.drop_duplicates(inplace=True)
    # dataset.to_csv("dataset.csv")

    dataset = pd.read_csv("dataset.csv")

    X = dataset.iloc[:, 1:-2]
    y = dataset.iloc[:,-1]

    # kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
    models = []
    models.append(RandomForestClassifier(n_estimators=165, max_depth=4, criterion='entropy'))
    models.append(GradientBoostingClassifier(max_depth=4))
    models.append(KNeighborsClassifier(n_neighbors=20))
    models.append(GaussianNB())
    assert True

    y_test = y_test.to_numpy()
    plt.figure(figsize=(10, 10))
    for model in models:
        model.fit(X_train, y_train)
        joblib.dump(model, f"{type(model).__name__}_model.pkl", compress=9)
        predictions = model.predict(X_test)
        true_positives = 0
        true_negatives = 0
        for idx, num in enumerate(predictions):
            if num == 1 and y_test[idx] == 1:
                true_positives += 1
            elif num == 0 and y_test[idx] == 0:
                true_negatives += 1
        true_positive_percent = true_positives / len(y_test[y_test == 1])
        true_negative_percent = true_negatives / len(y_test[y_test == 0])
        print(f"Model: {type(model).__name__}, True Pos: {true_positive_percent}, True Neg: {true_negative_percent}")
        pred_scr = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, pred_scr)
        roc_auc = roc_auc_score(y_test, pred_scr)
        md = str(model)
        md = md[:md.find('(')]
        pl.plot(fpr, tpr, label='ROC fold %s (auc = %0.2f)' % (md, roc_auc))

    # pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    # pl.xlim([0, 1])
    # pl.ylim([0, 1])
    # pl.xlabel('False Positive Rate')
    # pl.ylabel('True Positive Rate')
    # pl.title('Receiver operating characteristic example')
    # pl.legend(loc="lower right")
    # pl.show(block=True)