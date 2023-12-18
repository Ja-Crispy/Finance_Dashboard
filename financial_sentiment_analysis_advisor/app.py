import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import gradio as gr

# Initialize sentiment analysis model and tokenizer
sentiment_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("microsoft/phi-2")
sentiment_model.eval()

# Initialize stock identification model and tokenizer
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_model.eval()

def get_advice(sentiment_label, stocks_mentioned):
    # Add your own logic for providing advice based on sentiment and stocks mentioned
    if sentiment_label == "Positive":
        advice = "Positive sentiment. Consider taking advantage of positive market trends."
    elif sentiment_label == "Negative":
        if stocks_mentioned:
            advice = f"Negative sentiment. Consider re-evaluating your position on stocks: {', '.join(stocks_mentioned)}."
        else:
            advice = "Negative sentiment. Consider monitoring the market for potential impacts."
    else:
        advice = "Neutral sentiment. The market may not be strongly influenced. Monitor for changes."

    return advice

'''def predict_sentiment_and_stock_info(headline):
    # Sentiment Analysis
    sentiment_inputs = sentiment_tokenizer(headline, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        sentiment_outputs = sentiment_model(**sentiment_inputs)
        sentiment_prediction = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)

    pos, neg, neutr = sentiment_prediction[:, 0].item(), sentiment_prediction[:, 1].item(), sentiment_prediction[:, 2].item()
    sentiment_label = "Positive" if pos > neg and pos > neutr else "Negative" if neg > pos and neg > neutr else "Neutral"

    # Named Entity Recognition (NER)
    ner_inputs = ner_tokenizer(headline, return_tensors="pt")
    with torch.no_grad():
        ner_outputs = ner_model(**ner_inputs)

    # Identify stocks mentioned in the headline
    ner_predictions = torch.nn.functional.softmax(ner_outputs.logits, dim=-1).argmax(2)
    tokens = ner_tokenizer.convert_ids_to_tokens(ner_inputs['input_ids'][0].tolist())  # Use ner_inputs here
    entities = ner_tokenizer.convert_ids_to_tokens(ner_predictions[0].tolist())
    stocks_mentioned = [tokens[i] for i, entity in enumerate(entities) if entity.startswith("B")]

    # Advice based on sentiment and identified stocks
    advice = get_advice(sentiment_label, stocks_mentioned)

    return sentiment_label, advice'''

def predict_sentiment_and_stock_info(headline):
    # Sentiment Analysis
    sentiment_inputs = sentiment_tokenizer(headline, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        sentiment_outputs = sentiment_model(**sentiment_inputs)
        sentiment_prediction = torch.nn.functional.softmax(sentiment_outputs.logits, dim=-1)

    pos, neg, neutr = sentiment_prediction[:, 0].item(), sentiment_prediction[:, 1].item(), sentiment_prediction[:, 2].item()
    sentiment_label = "Positive" if pos > neg and pos > neutr else "Negative" if neg > pos and neg > neutr else "Neutral"

    # Named Entity Recognition (NER)
    ner_inputs = ner_tokenizer(headline, return_tensors="pt")
    with torch.no_grad():
        ner_outputs = ner_model(**ner_inputs)

    # Identify stocks mentioned in the headline
    ner_predictions = torch.nn.functional.softmax(ner_outputs.logits, dim=-1).argmax(2)
    tokens = ner_tokenizer.convert_ids_to_tokens(ner_inputs['input_ids'][0].tolist())  # Use ner_inputs here
    entities = ner_tokenizer.convert_ids_to_tokens(ner_predictions[0].tolist())
    stocks_mentioned = [tokens[i] for i, entity in enumerate(entities) if entity.startswith("B")]

    # Generate financial advice using phi2 model
    advice = sentiment_model.predict(sentiment_label, stocks_mentioned)

    return advice

# Gradio Interface
'''iface = gr.Interface(
    fn=predict_sentiment_and_stock_info,
    inputs="text",
    outputs=["text", "text"],
    live=True,
    title="Financial News Sentiment and Stock Analysis",
    description="Enter a financial news headline to analyze its sentiment, identify mentioned stocks, and get advice on how to proceed."
)

iface.launch()'''

# Gradio Interface
iface = gr.Interface(
    fn=predict_sentiment_and_stock_info,
    inputs=[gr.Textbox(lines=2, label="Headline")],
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Advice")
    ],
    live=True,
    title="Financial News Sentiment and Stock Analysis",
    description="Enter a financial news headline to analyze its sentiment, identify mentioned stocks, and get advice on how to proceed."
)

iface.launch()