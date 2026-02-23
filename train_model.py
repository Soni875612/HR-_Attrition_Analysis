from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data.csv")

df = df.drop(["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"], axis=1)

df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("Logistic Accuracy:", lr_acc)
print("Random Forest Accuracy:", rf_acc)

# Save best model
best_model = rf if rf_acc > lr_acc else lr
cm = confusion_matrix(y_test, best_model.predict(X_test))
pickle.dump(cm, open("confusion_matrix.pkl", "wb"))

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("features.pkl", "wb"))
