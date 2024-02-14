from flask import Flask, request, jsonify, render_template
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='/home/praveen/Music/chatbot-main')

with open('data.txt', 'r', errors='ignore') as f:
    raw_doc = f.read()
raw_doc = raw_doc.lower()
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Greeting function
def greet(sentence):
    greet_inputs = ('hello', 'hi', 'whassup', 'how are you?')
    greet_responses = ('hi', 'hey', 'hey there!', 'There there!!')
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Response generation function
def response(user_response):
    robal_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robal_response = robal_response + "I am sorry. Unable to understand you!"
        return robal_response
    else:
        robal_response = robal_response + sentence_tokens[idx]
        return robal_response


# Flask route to handle incoming requests
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']  # Get user input from POST request
    if user_input != 'bye':
        if user_input == 'thank you' or user_input == 'thanks':
            response_text = 'You are welcome.'
        else:
            greeting = greet(user_input)
            if greeting:
                response_text = greeting
            else:
                sentence_tokens.append(user_input)
                word_tokens.extend(nltk.word_tokenize(user_input))
                final_words = list(set(word_tokens))
                response_text = response(user_input)
                sentence_tokens.remove(user_input)
    else:
        response_text = 'Goodbye!'
    return jsonify({'response': response_text})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
