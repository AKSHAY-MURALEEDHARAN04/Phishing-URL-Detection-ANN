"""
Phishing URL Detection — Streamlit App
---------------------------------------
Loads the trained ANN (phishing_ann_model.keras), the fitted StandardScaler
(phishing_scaler.pkl), and the feature column list (phishing_feature_columns.pkl)
produced by the training notebook, and uses them to classify a URL as
legitimate or phishing.

The feature-extraction function below is an exact copy of `streamlit_extract`
in the training notebook — keep the two in sync if you change either one.

Run with:
    streamlit run app.py

Expected files in the same directory:
    phishing_ann_model.keras
    phishing_scaler.pkl
    phishing_feature_columns.pkl
"""

import ipaddress
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

MODEL_PATH = "phishing_ann_model.keras"
SCALER_PATH = "phishing_scaler.pkl"
COLUMNS_PATH = "phishing_feature_columns.pkl"


# --------------------------------------------------------------------------
# Feature extraction (identical to the training notebook)
# --------------------------------------------------------------------------
def streamlit_extract(url: str):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    domain = parsed.netloc
    domain_without_port = domain.split(":")[0]

    url_length = len(url)
    domain_length = len(domain_without_port)

    try:
        ipaddress.ip_address(domain_without_port)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    tld = domain_without_port.split(".")[-1] if "." in domain_without_port else ""
    tld_length = len(tld)

    domain_parts = domain_without_port.split(".")
    no_of_subdomain = len(domain_parts) - 2 if len(domain_parts) > 2 else 0

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    special_chars = sum(not c.isalnum() for c in url)

    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")
    other_special_chars = sum(url.count(c) for c in ["@", "#", "%", "$", "!", "*", ";"])

    letter_ratio = letters / url_length if url_length > 0 else 0
    digit_ratio = digits / url_length if url_length > 0 else 0
    special_ratio = special_chars / url_length if url_length > 0 else 0

    # Crucial Fix: Removing protocol before checking for obfuscation
    clean_url = url.replace("https://", "").replace("http://", "")
    obfuscated_chars = sum(clean_url.count(c) for c in ["%", "@", "\\", "//"])
    obfuscation_ratio = obfuscated_chars / url_length if url_length > 0 else 0
    has_obfuscation = 1 if obfuscated_chars > 0 else 0

    is_https = 1 if parsed.scheme == "https" else 0

    url_lower = url.lower()
    bank = int(any(word in url_lower for word in ["bank", "banking", "account"]))
    pay = int(any(word in url_lower for word in ["payment", "pay", "paypal", "checkout"]))
    crypto = int(any(word in url_lower for word in ["crypto", "bitcoin", "ethereum", "wallet"]))

    return [
        url_length, domain_length, is_domain_ip, tld_length, no_of_subdomain,
        has_obfuscation, obfuscated_chars, obfuscation_ratio, letters, letter_ratio,
        digits, digit_ratio, no_of_equals, no_of_qmark, no_of_ampersand,
        other_special_chars, special_ratio, is_https, bank, pay, crypto,
    ]


FEATURE_COLUMNS = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
    "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL",
    "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL", "IsHTTPS", "Bank", "Pay", "Crypto",
]


# --------------------------------------------------------------------------
# Cached artifact loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns


def predict_url(url: str, model, scaler, columns):
    features = streamlit_extract(url)
    X = pd.DataFrame([features], columns=columns)
    X_scaled = scaler.transform(X)
    prob_legit = float(model.predict(X_scaled, verbose=0)[0][0])
    prob_phish = 1 - prob_legit
    label = "Legitimate" if prob_legit >= 0.5 else "Phishing"
    return label, prob_legit, prob_phish, dict(zip(columns, features))


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🛡️ Phishing URL Detector")
st.caption(
    "ANN classifier trained on the PhiUSIIL Phishing URL Dataset "
    "(~235K URLs, 128→64→32→1 architecture, 97.5% accuracy)."
)

try:
    model, scaler, columns = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    st.error(
        "Couldn't load model artifacts. Make sure `phishing_ann_model.keras`, "
        "`phishing_scaler.pkl`, and `phishing_feature_columns.pkl` are in the "
        "same directory as this app.\n\n"
        f"Details: {e}"
    )

tab_single, tab_batch = st.tabs(["🔎 Check a URL", "📄 Batch (CSV)"])

# ---- Single URL tab -------------------------------------------------------
with tab_single:
    url_input = st.text_input(
        "Enter a URL to check",
        placeholder="e.g. https://www.example.com/login",
    )
    check_clicked = st.button("Analyze URL", type="primary", disabled=not artifacts_ok)

    if check_clicked:
        if not url_input.strip():
            st.warning("Please enter a URL first.")
        else:
            with st.spinner("Analyzing..."):
                label, prob_legit, prob_phish, feats = predict_url(
                    url_input.strip(), model, scaler, columns
                )

            if label == "Legitimate":
                st.success(f"✅ **{label}** — {prob_legit * 100:.2f}% confidence")
            else:
                st.error(f"🚨 **{label}** — {prob_phish * 100:.2f}% confidence")

            col1, col2 = st.columns(2)
            col1.metric("P(Legitimate)", f"{prob_legit * 100:.2f}%")
            col2.metric("P(Phishing)", f"{prob_phish * 100:.2f}%")
            st.progress(prob_legit)

            with st.expander("View extracted features"):
                feat_df = pd.DataFrame(feats.items(), columns=["Feature", "Value"])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

            with st.expander("Quick signal summary"):
                signals = []
                if not feats["IsHTTPS"]:
                    signals.append("⚠️ Not using HTTPS")
                if feats["IsDomainIP"]:
                    signals.append("⚠️ Domain is a raw IP address")
                if feats["HasObfuscation"]:
                    signals.append("⚠️ URL contains obfuscated characters")
                if feats["URLLength"] > 75:
                    signals.append("⚠️ Unusually long URL")
                if feats["NoOfSubDomain"] > 2:
                    signals.append("⚠️ Multiple subdomains")
                if feats["Bank"] or feats["Pay"] or feats["Crypto"]:
                    signals.append("⚠️ Contains banking/payment/crypto keywords")
                if not signals:
                    signals.append("✅ No obvious red flags detected")
                for s in signals:
                    st.write(s)

# ---- Batch CSV tab ---------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with a `URL` column to score multiple URLs at once.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], disabled=not artifacts_ok)

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read the CSV file: {e}")
            batch_df = None

        if batch_df is not None:
            if "URL" not in batch_df.columns:
                st.error("The CSV must contain a column named `URL`.")
            else:
                with st.spinner(f"Scoring {len(batch_df)} URLs..."):
                    feature_rows = batch_df["URL"].astype(str).apply(streamlit_extract).tolist()
                    X = pd.DataFrame(feature_rows, columns=columns)
                    X_scaled = scaler.transform(X)
                    probs_legit = model.predict(X_scaled, verbose=0).flatten()

                    results = batch_df.copy()
                    results["Prob_Legitimate"] = probs_legit
                    results["Prob_Phishing"] = 1 - probs_legit
                    results["Prediction"] = np.where(
                        probs_legit >= 0.5, "Legitimate", "Phishing"
                    )

                st.dataframe(results, use_container_width=True)

                n_phish = int((results["Prediction"] == "Phishing").sum())
                n_legit = int((results["Prediction"] == "Legitimate").sum())
                c1, c2 = st.columns(2)
                c1.metric("Flagged Phishing", n_phish)
                c2.metric("Flagged Legitimate", n_legit)

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="phishing_predictions.csv",
                    mime="text/csv",
                )

st.divider()
st.caption(
    "Model: Keras ANN (128→64→32→1) · Dataset: PhiUSIIL Phishing URL Dataset · "
    "For research/portfolio use — not a substitute for enterprise security tooling."
)
