import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the saved model and tokenizer
model_path = 'fake_news_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to classify news text
def classify_text(text):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=-1).cpu().item()
    return ("Real" if prediction == 1 else "Fake", probabilities[prediction])

# Sidebar
st.sidebar.title("🛠️ Settings")
st.sidebar.write("Adjust the theme or explore helpful links:")
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"])
st.sidebar.markdown("[About Fake News](https://en.wikipedia.org/wiki/Fake_news)")
st.sidebar.markdown("[Contact Us](mailto:prashantchoudharys123@gmail.com)")

# Themed Background
if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Header
st.markdown(
    """
    <div style="background-color:#f7c948;padding:10px;border-radius:5px;">
        <h1 style="color:#333333;text-align:center;">📰 Fake News Classifier</h1>
        <p style="color:#333333;text-align:center;">Check if a piece of news is Real or Fake with confidence!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main interface
st.write("### 🤔 Enter the news text below to classify it:")

# User input
user_input = st.text_area("📜 News Text", "", height=200)

if st.button("🔍 Classify"):
    if user_input.strip():
        with st.spinner("Analyzing... Please wait!"):
            result, confidence = classify_text(user_input)
        st.success(f"**Prediction:** {result}")
        st.write(f"**Confidence Score:** {confidence:.2%}")

        # Display engaging icons or GIFs
        if result == "Fake":
            st.warning("⚠️ **Fake News Detected!** Please verify the source of this information.")
            st.image("https://media.giphy.com/media/3o7abldj0b3rxrZUxW/giphy.gif", caption="Stay Alert!", use_column_width=True)
        else:
            st.info("✅ **This news appears to be Real.** Good job staying informed!")
            st.image("https://media.giphy.com/media/xUPGcyiKfEl3kjTH5e/giphy.gif", caption="Keep Sharing Facts!", use_column_width=True)
    else:
        st.error("❌ Please enter some text for classification.")

# User Feedback
st.write("### 🌟 Help Us Improve!")
st.write("What did you think of the Fake News Classifier?")
feedback = st.radio(
    "Your feedback helps us improve:",
    ["It's amazing! ⭐", "It's good but needs improvement.", "It's not accurate. 😕"]
)
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center;">
        Made with ❤️ by [Prashant](https://www.linkedin.com/in/prashant-kumar-62b76024a/) | Powered by BERT and Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)

