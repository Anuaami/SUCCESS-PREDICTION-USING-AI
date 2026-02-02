from flask import Flask, render_template, request
import pickle
import pandas as pd
from nlp_utils import course_advisor
from openai import OpenAI

# 🔐 Put your OpenAI API Key here
client = OpenAI(api_key="AIzaSyB88n7DovXKwPf6_cgqZ8xtsMuKgaA_92k")

app = Flask(__name__)
model = pickle.load(open("course_success_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "course_title": request.form["course_title"],
        "instructor": request.form["instructor"],
        "duration_minutes": float(request.form["duration_minutes"]),
        "platform": request.form["platform"],
        "views_per_day": float(request.form["views_per_day"]),
        "engagement_rate": float(request.form["engagement_rate"]),
        "likes": float(request.form["likes"]),
        "comments": float(request.form["comments"])
    }

    df = pd.DataFrame([data])
    df["text"] = df["course_title"] + " " + df["instructor"]

    pred = model.predict(df)[0]
    result = "High Success Probability" if pred == 1 else "Low Success Probability"

    advice = course_advisor(
        data["course_title"],
        data["duration_minutes"],
        data["platform"],
        data["views_per_day"],
        data["engagement_rate"],
        pred
    )

    return render_template("result.html", prediction=result, advice=advice)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form["message"]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert Online Course Success Advisor."},
            {"role": "user", "content": user_msg}
        ]
    )

    reply = response.choices[0].message.content
    return reply

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


