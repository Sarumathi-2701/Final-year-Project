import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import re

# Optional: Import transformers for Hugging Face model support
try:
    from transformers import pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🔍",
    layout="wide"
)

# Define common toxic words
TOXIC_WORDS = {
    'profanity': ['fuck', 'shit', 'ass', 'damn', 'bitch'],
    'hate_speech': ['racist', 'nigger', 'nazi', 'faggot', 'retard'],
    'insults': ['idiot', 'stupid', 'dumb', 'moron', 'loser'],
    'threats': ['kill', 'die', 'murder', 'hurt', 'attack'],
    'harassment': ['stalker', 'creep', 'pervert', 'harassment']
}

# Load the saved model and vectorizer
@st.cache_resource
def load_saved_model():
    model = load_model('toxic_comment_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load Hugging Face model
@st.cache_resource
def load_huggingface_model():
    if HUGGINGFACE_AVAILABLE:
        return pipeline("text-classification", model="unitary/toxic-bert")
    else:
        return None

def predict_toxicity_custom(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text])
    raw_prediction = model.predict(text_vectorized.toarray())[0][0]
    # Convert to binary classification (toxic or not)
    is_toxic = raw_prediction > 0.5
    return is_toxic

def predict_toxicity_huggingface(text, classifier):
    result = classifier(text)[0]
    # The toxic-bert model returns LABEL_0 for non-toxic and LABEL_1 for toxic
    is_toxic = result['label'] == 'LABEL_1'
    return is_toxic

def identify_toxic_words(text):
    text_lower = text.lower()
    found_toxic_words = {}

    for category, words in TOXIC_WORDS.items():
        found_words = []
        for word in words:
            if word in text_lower:
                found_words.append(word)
        if found_words:
            found_toxic_words[category] = found_words

    return found_toxic_words

def create_result_display(is_toxic):
    """
    Creates a visual indicator for toxic/non-toxic classification
    """
    if is_toxic:
        return go.Figure(go.Indicator(
            mode="number+gauge+delta",
            gauge={'shape': "bullet", 'axis': {'range': [0, 1]}, 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}},
            value=1,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "TOXIC", 'font': {'color': 'red', 'size': 24}}
        ))
    else:
        return go.Figure(go.Indicator(
            mode="number+gauge+delta",
            gauge={'shape': "bullet", 'axis': {'range': [0, 1]}, 'threshold': {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': 0.5}},
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "NON-TOXIC", 'font': {'color': 'green', 'size': 24}}
        ))

def create_toxic_words_chart(toxic_words_dict):
    categories = []
    word_counts = []

    for category, words in toxic_words_dict.items():
        categories.append(category)
        word_counts.append(len(words))

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=word_counts,
            marker_color=['red', 'orange', 'yellow', 'purple', 'brown']
        )
    ])

    fig.update_layout(
        title="Toxic Words by Category",
        xaxis_title="Category",
        yaxis_title="Number of Words",
        showlegend=False
    )

    return fig

def create_prediction_history_chart(prediction_history):
    """
    Creates a bar chart to visualize number of toxic vs non-toxic comments
    """
    counts = prediction_history['is_toxic'].value_counts().reset_index()
    counts.columns = ['classification', 'count']
    counts['classification'] = counts['classification'].map({True: 'Toxic', False: 'Non-Toxic'})

    fig = px.bar(
        counts,
        x='classification',
        y='count',
        color='classification',
        color_discrete_map={'Toxic': 'red', 'Non-Toxic': 'green'},
        title="Classification History"
    )

    fig.update_layout(showlegend=False)
    return fig

def main():
    st.title("💬 Toxic Comment Classifier")
    st.write("Enter a comment (up to 200 words) to classify it as toxic or non-toxic")

    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=['text', 'is_toxic'])

    col1, col2 = st.columns([2, 1])

    with col1:
        # Model selection
        model_option = st.radio(
            "Select model:",
            ["Custom Model", "Hugging Face Model (toxic-bert)"] if HUGGINGFACE_AVAILABLE else ["Custom Model"]
        )

        comment = st.text_area("Enter your comment:", height=100, max_chars=1000)

        if st.button("Analyze Comment"):
            if comment:
                # Check word limit
                word_count = len(comment.split())
                if word_count > 200:
                    st.warning("Please limit your text to 200 words. Current word count: " + str(word_count))
                else:
                    # Get prediction based on selected model
                    try:
                        if model_option == "Custom Model":
                            model, vectorizer = load_saved_model()
                            is_toxic = predict_toxicity_custom(comment, model, vectorizer)
                        else:  # Hugging Face model
                            classifier = load_huggingface_model()
                            if classifier:
                                is_toxic = predict_toxicity_huggingface(comment, classifier)
                            else:
                                st.error("Failed to load Hugging Face model. Install with: pip install transformers")
                                return

                        # Identify toxic words
                        toxic_words = identify_toxic_words(comment)

                        # Add to prediction history
                        new_prediction = pd.DataFrame({
                            'text': [comment],
                            'is_toxic': [is_toxic]
                        })
                        st.session_state.prediction_history = pd.concat(
                            [st.session_state.prediction_history, new_prediction],
                            ignore_index=True
                        )

                        # Display result
                        st.plotly_chart(create_result_display(is_toxic), use_container_width=True)

                        # Display toxic words analysis
                        st.write("### Toxic Words Analysis:")
                        if toxic_words:
                            st.warning("Found toxic words in the following categories:")
                            for category, words in toxic_words.items():
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(words)}")

                            # Display toxic words chart
                            st.plotly_chart(create_toxic_words_chart(toxic_words), use_container_width=True)
                        else:
                            st.success("No common toxic words found in the text")

                        # Display overall result
                        st.write("### Overall Analysis Result:")
                        if is_toxic:
                            st.error("This comment is classified as TOXIC")
                        else:
                            st.success("This comment is classified as NON-TOXIC")
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")

            else:
                st.warning("Please enter a comment to analyze")

    with col2:
        st.write("### Statistics")
        if not st.session_state.prediction_history.empty:
            total_comments = len(st.session_state.prediction_history)
            toxic_comments = st.session_state.prediction_history['is_toxic'].sum()

            st.metric("Total Comments Analyzed", total_comments)
            st.metric("Toxic Comments Detected", int(toxic_comments))
            st.metric("Non-toxic Comments", total_comments - int(toxic_comments))

    if not st.session_state.prediction_history.empty:
        st.write("### Prediction History")
        history_chart = create_prediction_history_chart(st.session_state.prediction_history)
        st.plotly_chart(history_chart, use_container_width=True)

        st.write("### Recent Predictions")

        # Format the dataframe to show "Toxic" or "Non-Toxic" instead of True/False
        display_df = st.session_state.prediction_history.tail(5).copy()
        display_df['Classification'] = display_df['is_toxic'].map({True: 'Toxic', False: 'Non-Toxic'})
        st.dataframe(display_df[['text', 'Classification']])

        if st.button("Clear History"):
            st.session_state.prediction_history = pd.DataFrame(columns=['text', 'is_toxic'])
            st.experimental_rerun()

if __name__ == "__main__":
    main()