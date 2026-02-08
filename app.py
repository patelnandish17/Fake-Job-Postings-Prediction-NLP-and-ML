import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import re

# ---------------- TESSERACT PATH ----------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="Fake Job Post Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Fake Job Post Detection Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "🔍 Search Jobs", "🧠 Predict Job Text", "🖼️ Detect from Poster"]
)

# ---------------- UPLOAD DATASET ----------------
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Job Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset Loaded Successfully!")

    # ---------------- DATA CLEANING ----------------
    df = df.dropna(subset=["description"])

    for col in ["title", "location", "company_profile"]:
        df[col] = df[col].fillna("Unknown")

    # ---------------- BALANCE DATA ----------------
    X = df["description"]
    y = df["fraudulent"]

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X.to_frame(), y)
    X_res = X_res["description"]

    # ---------------- TF-IDF ----------------
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    X_vec = vectorizer.fit_transform(X_res)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_res, test_size=0.2, random_state=42
    )

    # ---------------- MODELS ----------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        )
    }

    results = {}
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_curves[name] = (fpr, tpr, roc_auc)

    # =====================================================
    # ================= OVERVIEW PAGE =====================
    # =====================================================
    if page == "📊 Overview":
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Number of Fake Jobs", int(df["fraudulent"].sum()))

        # -------- MODEL ACCURACY --------
        st.subheader("📈 Model Accuracy Comparison")
        acc_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Model", y="Accuracy", data=acc_df, ax=ax)
        ax.set_ylim(0, 1)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height()*100:.2f}%",
                (p.get_x() + 0.25, p.get_height() + 0.01)
            )

        st.pyplot(fig)

        # -------- CONFUSION MATRIX --------
        st.subheader("🧩 Confusion Matrix (Random Forest)")
        best_model = models["Random Forest"]
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="coolwarm",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # -------- ROC CURVES --------
        st.subheader("📉 ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, (fpr, tpr, roc_auc) in roc_curves.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        st.pyplot(fig)

        # -------- PIE CHART --------
        st.subheader("Real vs Fake Job Distribution")
        fraud_counts = df["fraudulent"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(
            fraud_counts,
            labels=["Real", "Fake"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"]
        )
        ax.set_title("Real vs Fake Jobs")
        st.pyplot(fig)

        # -------- WORDCLOUD (FIXED) --------
        st.subheader("☁️ WordCloud of Fake Job Posts")

        fake_text = " ".join(
            df[df["fraudulent"] == 1]["description"]
            .astype(str)
            .tolist()
        )

        fake_text = re.sub(r"[^a-zA-Z ]", " ", fake_text).lower()

        if fake_text.strip():
            wordcloud = WordCloud(
                width=900,
                height=400,
                background_color="black",
                stopwords=STOPWORDS,
                max_words=200,
                collocations=False
            ).generate(fake_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No fake job text available for WordCloud.")

    # =====================================================
    # ================= SEARCH PAGE =======================
    # =====================================================
    elif page == "🔍 Search Jobs":
        st.subheader("Search and Filter Job Posts")

        title_search = st.text_input("Search by Job Title:")
        loc_search = st.text_input("Search by Location:")
        company_search = st.text_input("Search by Company:")

        filtered_df = df.copy()

        if title_search:
            filtered_df = filtered_df[
                filtered_df["title"].str.contains(title_search, case=False)
            ]
        if loc_search:
            filtered_df = filtered_df[
                filtered_df["location"].str.contains(loc_search, case=False)
            ]
        if company_search:
            filtered_df = filtered_df[
                filtered_df["company_profile"].str.contains(company_search, case=False)
            ]

        st.dataframe(
            filtered_df[
                ["title", "location", "company_profile", "fraudulent"]
            ].head(30)
        )

    # =====================================================
    # ================= PREDICT TEXT ======================
    # =====================================================
    elif page == "🧠 Predict Job Text":
        st.subheader("🔍 Test a Job Description or Poster Text")

        user_input = st.text_area("Paste job description or OCR text:",height=450)
        selected_model = st.selectbox("Select Model", list(models.keys()))

        if st.button("Predict"):
            if not user_input.strip():
                st.warning("Please enter a job description.")
            else:
                model = models[selected_model]
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                prob = model.predict_proba(input_vec)[0][1]

                if prediction == 1:
                    st.error(f"🚨 Fake Job Detected! ({prob*100:.1f}%)")
                else:
                    st.success(f"✅ Real Job Detected! ({(1-prob)*100:.1f}%)")

    # =====================================================
    # ================= POSTER OCR ========================
    # =====================================================
    elif page == "🖼️ Detect from Poster":
        st.subheader("Upload Job Poster Image")

        uploaded_image = st.file_uploader(
            "📷 Upload Poster Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Poster", use_column_width=True)

            extracted_text = pytesseract.image_to_string(image)

            st.write("### Extracted Text")
            st.write(extracted_text)

            if st.button("Analyze Poster"):
                model = models["Random Forest"]
                input_vec = vectorizer.transform([extracted_text])
                prediction = model.predict(input_vec)[0]
                prob = model.predict_proba(input_vec)[0][1]

                if prediction == 1:
                    st.error(f"🚨 Fake Job Poster Detected! ({prob*100:.1f}%)")
                else:
                    st.success(f"✅ Real Job Poster Detected! ({(1-prob)*100:.1f}%)")

else:
    st.warning("Please upload a dataset to start the app.")
