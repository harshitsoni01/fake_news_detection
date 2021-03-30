from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
import csv
import re

begin_date = dt.date(2020,7,1)
end_date = dt.date.today()

limits = input("Enter number of tweets you want to scrape:\n")
limits = int(limits)

langs = "english"
#user1 = PolitiFact
#user2 = GossipCop

basic_tweets = query_tweets("#news" , begindate=begin_date, enddate=end_date , limit=limits, lang=langs)
user1_tweets = query_tweets("@PolitiFact" , begindate= begin_date, enddate=end_date, limit=limits, lang=langs)
user2_tweets = query_tweets("@GossipCop" , begindate= begin_date, enddate=end_date, limit=limits, lang=langs)

df = pd.DataFrame(b.__dict__ for b in basic_tweets)
df.drop(df.columns.difference(['username','text']), 1, inplace=True)
df.to_csv("C:/fake_news_detection/csv_files/tweet.csv", index= False)

df1 = pd.DataFrame(u1.__dict__ for u1 in user1_tweets)
df1.drop(df1.columns.difference(['username','text']), 1, inplace=True)
df1.to_csv("C:/fake_news_detection/csv_files/politifact.csv" , index= False)

df2 = pd.DataFrame(u2.__dict__ for u2 in user2_tweets)
df2.drop(df2.columns.difference(['username','text']), 1, inplace=True)
df2.to_csv("C:/fake_news_detection/csv_files/gossipcop.csv" , index= False)