import io
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# 1. Dataset
csv_data = """study_hours,attendance_pct,assignments_done,past_score,result
1.5,60,no,42,fail
2.0,65,no,48,fail
2.5,70,yes,52,fail
3.0,72,no,55,fail
4.0,80,yes,64,pass
4.5,85,yes,70,pass
5.0,88,yes,75,pass
5.5,90,yes,82,pass
1.0,55,no,38,fail
3.5,78,yes,60,pass
4.2,82,yes,68,pass
2.2,68,yes,50,fail
6.0,92,yes,85,pass
1.8,62,no,45,fail
4.8,84,no,66,pass
5.2,87,yes,78,pass"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Encode features and target
le_assign = LabelEncoder()
df["assignments_encoded"] = le_assign.fit_transform(df["assignments_done"])  # no=0, yes=1

X = df[["study_hours", "attendance_pct", "assignments_encoded", "past_score"]]
y = df["result"].map({"fail": 0, "pass": 1})

# 3. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    max_depth=3,
    min_samples_split=2,
    max_leaf_nodes=5,
    random_state=3
)
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_01 = None
    prediction_label = None
    form_values = {}

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance_pct = float(request.form["attendance_pct"])
        assignments_done = request.form["assignments_done"]
        past_score = float(request.form["past_score"])

        form_values = {
            "study_hours": study_hours,
            "attendance_pct": attendance_pct,
            "assignments_done": assignments_done,
            "past_score": past_score,
        }

        input_df = pd.DataFrame({
            "study_hours": [study_hours],
            "attendance_pct": [attendance_pct],
            "assignments_encoded": le_assign.transform([assignments_done]),
            "past_score": [past_score],
        })

        pred = int(model.predict(input_df)[0])
        prediction_01 = pred
        prediction_label = "pass" if pred == 1 else "fail"

    return render_template(
        "index.html",
        prediction_01=prediction_01,
        prediction_label=prediction_label,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)