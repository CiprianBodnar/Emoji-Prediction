import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiments = {
    'compound': [],
    'negative': [],
    'positive': [],
    'neutral': []
}


pos = []
neg = []
neu = []
tweet_file = open("data/us/tweet_by_ID_11_3_2019__07_26_54.txt.text", "rb")
analyzer = SentimentIntensityAnalyzer()

max_tweets = 500

counter = 0
for tweet in tweet_file:
    counter += 1
    if counter > max_tweets:
        break
    tweet = tweet.decode()
    print(tweet, end='')
    scores = analyzer.polarity_scores(tweet)
    if scores['pos'] > scores['neg']:
        pos.append(tweet.encode())
        print('positive')
    elif scores['pos'] < scores['neg']:
        neg.append(tweet.encode())
        print('negative')
    else:
        neu.append(tweet.encode())
        print('neutral')
    print()

    with open("nltk_pos.txt", "wb") as file:
        file.writelines(pos)
    with open("nltk_neg.txt", "wb") as file:
        file.writelines(neg)
    with open("nltk_neu.txt", "wb") as file:
        file.writelines(neu)




