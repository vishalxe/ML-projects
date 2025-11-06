import streamlit as st
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from io import StringIO
from PyPDF2 import PdfReader
import base64

# Load your actual dataset
df = pd.read_csv("email_dataset.csv")  # Make sure the CSV is in the same directory or give full path

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["category"], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Alert keyword function
def detect_alert(text):
    alert_keywords = [
        "expire", "expiring", "ending", "last date", "final day",
        "today only", "due in", "offer ends", "deadline", "last chance"
    ]
    for keyword in alert_keywords:
        if keyword in text.lower():
            return True
    return False

# Streamlit setup
st.set_page_config(page_title="Email Auto-Categorizer", layout="wide")
st.title("📧 Email Auto-Categorizer")
st.markdown("### ▶️ Paste your email content or upload a file (.txt/.pdf)")

email_text = st.text_area("✉️ Email Content", height=150)
uploaded_file = st.file_uploader("📁 Upload .txt or .pdf", type=["txt", "pdf"])

def get_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        return file_contents
    return ""

if uploaded_file:
    email_text = get_text_from_file(uploaded_file)

# Prediction
if st.button("🚀 Categorize Email"):
    if email_text.strip() == "":
        st.warning("Please enter or upload email content.")
    else:
        predicted_category = model.predict(vectorizer.transform([email_text]))[0]
        
        # Override with 'alerts' if alert keyword found
        if detect_alert(email_text):
            predicted_category = "alerts"

        st.success(f"✅ Predicted Category: **{predicted_category}**")

        # Append to DataFrame
        df = pd.concat([df, pd.DataFrame({"text": [email_text], "category": [predicted_category]})], ignore_index=True)

# Display by category
st.markdown("## 🗂️ Categorized Emails")

categories = {
    "promotions": "📂 Promotions",
    "notifications": "🔔 Notifications",
    "important": "📌 Important",
    "jobs": "💼 Jobs",
    "spam": "🚫 Spam",
    "alerts": "⚠️ Alerts"
}
for key, label in categories.items():
    with st.expander(label):
        cat_df = df[df["category"] == key]
        if cat_df.empty:
            st.write("No emails in this category yet.")
        else:
            for email in cat_df["text"]:
                st.markdown(f"- {email}")
def get_text_from_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            pdf = PdfReader(uploaded_file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    return ""

# Download CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df_to_csv(df)
st.download_button(
    label="📥 Download Categorized Results as CSV",
    data=csv,
    file_name="categorized_emails.csv",
    mime="text/csv",
)

# Full Table
st.markdown("## 📋 Full Categorized Table")
st.dataframe(df, use_container_width=True)
