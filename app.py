from flask import Flask, abort, jsonify, request, render_template
import joblib
from feature import *
import json
from find_real_news import get_real_news  # get real news from NewsAPI

pipeline = joblib.load('./pipeline_final.sav')

app = Flask(__name__)


@app.route('/')
def home():
    name = "TechnologIQ"
    return render_template('index.html', name=name)


@app.route('/api', methods=['POST'])
def get_delay():

    result = request.form
    query_news = result['news']
    # returns dictionary with language, original text and translated text
    translated_query = lang_translate(query_news)
    cleaned_query = remove_punctuation_stopwords_lemma(
        translated_query["final_text"])
    query = get_all_query(translated_query["final_text"])
    pred = pipeline.predict(query)
    print(pred)

    # if news is fake, display alternative real news using NewsAPI
    real_news = get_real_news(translated_query["final_text"])
    return render_template('resultpage.html', test="test2", pred=pred, translated_query=translated_query, cleaned_query=cleaned_query, real_news=real_news)


if __name__ == '__main__':
    app.run(port=8080, debug=True)
