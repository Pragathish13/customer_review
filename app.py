from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
import os
import re
from collections import Counter, defaultdict

# ---------------- APP CONFIG ---------------- #

app = Flask(__name__)
app.secret_key = "secret123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "/"

# ---------------- LOAD ML MODEL ---------------- #

model = pickle.load(open(os.path.join(BASE_DIR, "sentiment_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def predict_sentiment(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

def extract_top_complaints(reviews, top_n=5):
    text = " ".join(reviews).lower()

    text = re.sub(r"[^a-z\s]", "", text)

    stopwords = [
        "the","is","and","to","a","of","for","in","on","this","that",
        "it","was","very","not","with","but","are","have","has"
    ]

    words = [w for w in text.split() if w not in stopwords and len(w) > 3]

    most_common = Counter(words).most_common(top_n)
    return most_common


# ---------------- DATABASE MODELS ---------------- #

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(10))  # admin / user

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product = db.Column(db.String(100))
    review_text = db.Column(db.Text)
    sentiment = db.Column(db.String(20))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- LOGIN ---------------- #

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            login_user(user)
            if user.role == "admin":
                return redirect("/admin")
            else:
                return redirect("/product")
    return render_template("auth/login.html")

# ---------------- ADMIN UPLOAD ---------------- #

@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin_upload():
    if request.method == "POST":
        file = request.files["file"]
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            sentiment = predict_sentiment(row["review"])

            review = Review(
                product=row["product"],
                review_text=row["review"],
                sentiment=sentiment
            )
            db.session.add(review)

        db.session.commit()
        return redirect("/admin/dashboard")

    return render_template("admin/upload.html")

# ---------------- ADMIN DASHBOARD ---------------- #

@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    reviews = Review.query.all()

    # Overall sentiment counts
    positive = sum(1 for r in reviews if r.sentiment == "positive")
    negative = sum(1 for r in reviews if r.sentiment == "negative")
    neutral  = sum(1 for r in reviews if r.sentiment == "neutral")

    # Complaint extraction (from negative reviews)
    negative_texts = [r.review_text for r in reviews if r.sentiment == "negative"]
    top_complaints = extract_top_complaints(negative_texts)

    # -------- STEP 6.1 PRODUCT-WISE LOGIC --------
    product_sentiment = defaultdict(lambda: {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    })

    for r in reviews:
        product_sentiment[r.product][r.sentiment] += 1

    product_labels = list(product_sentiment.keys())
    product_positive = [product_sentiment[p]["positive"] for p in product_labels]
    product_negative = [product_sentiment[p]["negative"] for p in product_labels]
    product_neutral  = [product_sentiment[p]["neutral"]  for p in product_labels]
    # --------------------------------------------

    return render_template(
        "admin/dashboard.html",
        reviews=reviews,
        positive=positive,
        negative=negative,
        neutral=neutral,
        complaints=top_complaints,
        product_labels=product_labels,
        product_positive=product_positive,
        product_negative=product_negative,
        product_neutral=product_neutral
    )
# ---------------- USER DASHBOARD ---------------- #

@app.route("/product",methods=["GET", "POST"])
@login_required
def user_dashboard():
    products = db.session.query(Review.product).distinct().all()
    selected_product = None
    reviews = []

    if request.method == "POST":
        selected_product = request.form["product"]
        reviews = Review.query.filter_by(product=selected_product).all()

    return render_template(
        "user/dashboard.html",
        products=products,
        reviews=reviews,
        selected_product=selected_product
    )
# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():
    logout_user()
    return redirect("/")

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
