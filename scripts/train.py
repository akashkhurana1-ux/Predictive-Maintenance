import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("data/train.csv")

X = df.drop("Engine_Condition", axis=1)
y = df["Engine_Condition"]

model = XGBClassifier()
model.fit(X, y)

joblib.dump(model, "model/predictive-maintenance_model.pkl")

