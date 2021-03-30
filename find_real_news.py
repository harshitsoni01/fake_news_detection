from newsapi import NewsApiClient
import json
from feature import remove_punctuation_stopwords_lemma


def get_real_news(title):
    newsapi = NewsApiClient(api_key='29c56386f46a4be7a70b8a525efb0462')
    title = remove_punctuation_stopwords_lemma(title)
    # get first 3 words from text to search
    title = ' '.join(title.split()[:3])
    # convert title to lowercase and replace whitespace with '+' to use as query parameter
    query = title.lower().replace(" ", "+")
    all_articles = newsapi.get_everything(q=query, page=1)
    real_news = {"totalResults": 0}
    # create object of real article title, text, and link
    if all_articles["totalResults"] != 0:
        real_news = {
            "title": all_articles["articles"][0]["title"],
            "text": all_articles["articles"][0]["description"],
            "link": all_articles["articles"][0]["url"],
            "source": all_articles["articles"][0]["source"]["name"],
            "totalResults": 1
        }
    return real_news
