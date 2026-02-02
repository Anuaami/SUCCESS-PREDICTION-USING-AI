import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Dataset
df = pd.read_csv("youtube_4792data_online_courses2.csv")

# 2. Text Feature (NLP)
df["text"] = df["course_title"].fillna("") + " " + df["instructor"].fillna("")

X_text = df["text"]

# 3. Numerical Features
num_features = ["duration_minutes", "views_per_day", "engagement_rate", "likes", "comments"]
X_num = df[num_features]

# 4. Target
y = df["success_label"]

# 5. Preprocessing Pipelines
text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=3000))
])

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("text", text_pipeline, "text"),
    ("num", num_pipeline, num_features)
])

# 6. Full ML Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train Model
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("\nModel Performance:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. Save Model
with open("course_success_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as: course_success_model.pkl")

