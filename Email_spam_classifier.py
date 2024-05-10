import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample emails and corresponding labels
sample_emails = [
    "Get rich Quick! Click here to win a million dollars!",
    "Hello, could you please review this document for me",
    "Discounts on luxury watches and handbags!",
    "Meeting scheduled for tomorrow, please confirm your attendance.",
    "Congratulations, you've won a free gift card!",
    "SIX chances to win CASH!",
    "From 100 to 20,000 pounds txt CSH11 and send to 87575.",
    "07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mo...",
    "WHO ARE YOU SEEING?",
    "Yeah hopefully, if tyler can't do it I could maybe ask around a bit",
    "Yes..gauti and sehwag out of odi series."
]
sample_labels = [1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0]

# Convert sample emails and labels to array using NumPy
sample_emails_array = np.array(sample_emails)
sample_labels_array = np.array(sample_labels)

# Vectorize the sample emails using CountVectorizer
vectorizer = CountVectorizer()
x_sample = vectorizer.fit_transform(sample_emails_array)

# Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(x_sample, sample_labels_array)

# Create a Streamlit app
st.set_page_config(layout="wide")
st.title("Email Spam Classifier")
st.markdown("---")

# Prompt user to select a sample email or enter a custom email
option = st.selectbox("Select a sample email or enter your own:", ["Select Sample Email"] + sample_emails)
if option == "Select Sample Email":
    new_email = st.text_area("Enter the email message:", value="", height=100)
else:
    new_email = st.text_area("Enter the email message:", value=option, height=100)

# Predict the label for the new email
if st.button("Classify", key="classify") and new_email:
    new_email_vectorized = vectorizer.transform([new_email])
    predicted_label = model.predict(new_email_vectorized)
    st.markdown("---")
    if predicted_label[0] == 0:
        st.success("The email is classified as: Not SPAM")
    else:
        st.error("The email is classified as: SPAM")

# custom CSS for text color and layout
st.markdown(
    """
    <style>
    .stTextInput>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #ccc;
        padding: 8px 12px;
        height: 100px;
        width: 100%;
        resize: vertical;
    }
    .stButton>button, .stSelectbox>div>div>div[role="button"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
    }
    .stButton>button:hover, .stSelectbox>div>div>div[role="button"]:hover {
        background-color: #45a049;
    }
    .stAlert>div>div>div>span {
        color: white;
    }
    .stAlert.error {
        background-color: #ff6347;
    }
    .stAlert.success {
        background-color: #32CD32;
    }
    </style>
    """,
    unsafe_allow_html=True
)
