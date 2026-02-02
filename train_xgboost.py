import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from scipy.sparse import hstack
import pickle

# Load dataset
df = pd.read_csv("youtube_4792data_online_courses2.csv")


# Cleaning
df.fillna(0, inplace=True)

# Feature engineering
df["engagement_rate"] = (df["likes"] + df["comments"]) / (df["views"] + 1)

# Target label
df["success"] = ((df["views"] > df["views"].median()) &
                 (df["engagement_rate"] > df["engagement_rate"].median())).astype(int)

# NLP text column
df["text"] = df["course_title"].astype(str) + " " + df["instructor"].astype(str)

# TF-IDF
tfidf = TfidfVectorizer(max_features=500, stop_words="english")
X_text = tfidf.fit_transform(df["text"])

# Numeric features
X_num = df[["duration_minutes", "views", "likes", "comments", "engagement_rate"]].values

# Combine
X = hstack([X_text, X_num])
y = df["success"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save model
pickle.dump(model, open("xgboost_course_success_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
