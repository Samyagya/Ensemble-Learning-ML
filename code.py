#importing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier, VotingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier



#Loading and Preprocessing data

dataFrame = pd.read_csv('/content/drive/MyDrive/Obesity_Dataset.csv')
X = pd.get_dummies(dataFrame.drop('NObeyesdad', axis=1))
Y = dataFrame['NObeyesdad']

scaler = StandardScaler()
X = scaler.fit_transform(X)

XTrain, XTest, YTrainRaw, YTestRaw = train_test_split(X, Y, test_size=0.2, random_state=16)
encoder = LabelEncoder()
YTrain = encoder.fit_transform(YTrainRaw)
YTest = encoder.transform(YTestRaw)

results = []



#Evaluation Function

def evaluate(model_name, YTrue, YPredicted, results):
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(YTrue, YPredicted),
        "Precision": precision_score(YTrue, YPredicted, average='weighted'),
        "Recall": recall_score(YTrue, YPredicted, average='weighted'),
        "F1 Score": f1_score(YTrue, YPredicted, average='weighted')
    })



#Baseline Models

#Model 1 - KNN
K = int(np.sqrt(len(XTrain)))
if K % 2 == 0:
  K += 1

knn = KNeighborsClassifier(n_neighbors=K, algorithm='ball_tree')
knn.fit(XTrain, YTrain)
evaluate("KNN", YTest, knn.predict(XTest), results)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(XTrain, YTrain)
evaluate("Decision Tree", YTest, dt.predict(XTest), results)

# SVM
svm_rbf = SVC(kernel='rbf', probability=True)
svm_rbf.fit(XTrain, YTrain)
evaluate("SVM (RBF)", YTest, svm_rbf.predict(XTest), results)




#Ensemble Learning Models

#Bagging (Boostrap Agregation)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100, max_samples=0.8, max_features=0.8,
    bootstrap=True, random_state=42
)
bagging.fit(XTrain, YTrain)
evaluate("Bagging (DT)", YTest, bagging.predict(XTest), results)




#Boosting
#AdaBoost
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(), n_estimators=100, learning_rate=0.1, random_state=42)
adaboost.fit(XTrain, YTrain)
evaluate("AdaBoost", YTest, adaboost.predict(XTest), results)

#Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)
gb.fit(XTrain, YTrain)
evaluate("Gradient Boosting", YTest, gb.predict(XTest), results)

#XGBoost
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(XTrain, YTrain)
evaluate("XGBoost", YTest, xgb.predict(XTest), results)



#Voting Ensemble
# Hard Voting
hard_vote = VotingClassifier(
    estimators=[('dt', dt), ('knn', knn), ('svm', SVC(kernel='rbf'))],
    voting='hard'
)
hard_vote.fit(XTrain, YTrain)
evaluate("Hard Voting", YTest, hard_vote.predict(XTest), results)

# Soft Voting
soft_vote = VotingClassifier(
    estimators=[('dt', dt), ('knn', knn), ('svm', svm_rbf)],
    voting='soft'
)
soft_vote.fit(XTrain, YTrain)
evaluate("Soft Voting", YTest, soft_vote.predict(XTest), results)

# Weighted Soft Voting
scores = (
    cross_validate(dt, X, y=encoder.transform(Y), cv=5, scoring='accuracy')['test_score'].mean(),
    cross_validate(knn, X, y=encoder.transform(Y), cv=5, scoring='accuracy')['test_score'].mean(),
    cross_validate(svm_rbf, X, y=encoder.transform(Y), cv=5, scoring='accuracy')['test_score'].mean()
)
total_score = sum(scores)

weighted_vote = VotingClassifier(
    estimators=[('dt', dt), ('knn', knn), ('svm', svm_rbf)],
    voting='soft',
    weights=[score / total_score for score in scores]
)
weighted_vote.fit(XTrain, YTrain)



#Stacking
stack = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', CalibratedClassifierCV(SVC(kernel='rbf', random_state=42), cv=3)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    final_estimator=LogisticRegression(),
    cv=2,
    passthrough=False,
    n_jobs=-1
)
stack.fit(XTrain, YTrain)
evaluate("Stacking", YTest, stack.predict(XTest), results)




#Blending
blend_rf = RandomForestClassifier(n_estimators=100, random_state=42)
blend_svm = SVC(kernel='rbf', probability=True, random_state=42)
blend_knn = KNeighborsClassifier(n_neighbors=5)

blend_rf.fit(XTrain, YTrain)
blend_svm.fit(XTrain, YTrain)
blend_knn.fit(XTrain, YTrain)

pred_rf = blend_rf.predict_proba(XTest)
pred_svm = blend_svm.predict_proba(XTest)
pred_knn = blend_knn.predict_proba(XTest)

blended_pred = (pred_rf + pred_svm + pred_knn) / 3
blended_final = np.argmax(blended_pred, axis=1)

evaluate("Blending", YTest, blended_final, results)



#Results and comparation

resultsDataFrame = pd.DataFrame(results)
print("\nFinal Comparison Table: ->")
print(resultsDataFrame)

#plots
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for i, metric in enumerate(metrics):
    plt.subplot(1, 4, i + 1)
    sns.barplot(x='Model', y=metric, data=resultsDataFrame)
    plt.xticks(rotation=90)
    plt.title(metric)

plt.tight_layout()
plt.show()



#Baseline vs Ensemble
baseline_names = ["Decision Tree", "KNN", "SVM (RBF)"]
ensemble_names = [model for model in resultsDataFrame['Model'] if model not in baseline_names]
baseline_df = resultsDataFrame[resultsDataFrame["Model"].isin(baseline_names)].reset_index(drop=True)
ensemble_df = resultsDataFrame[resultsDataFrame["Model"].isin(ensemble_names)].reset_index(drop=True)
print("\nBaseline Models Performance: ->")
print(baseline_df)
print("\nEnsemble Models Performance: ->")
print(ensemble_df)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
# Baseline plot
baseline_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", ax=axes[0], colormap="Blues")
axes[0].set_title("Baseline Models")
axes[0].set_ylabel("Score")
axes[0].legend(loc='lower right')
axes[0].set_ylim(0, 1)

# Ensemble plot
ensemble_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", ax=axes[1], color=['red','purple','cyan','orange'])
axes[1].set_title("Ensemble Models")
axes[1].legend(loc='lower right')
axes[1].set_ylim(0, 1)

plt.suptitle("Baseline vs Ensemble Model Performance", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
