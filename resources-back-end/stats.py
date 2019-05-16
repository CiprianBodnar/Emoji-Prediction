from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import time
import sys

analyzer = SentimentIntensityAnalyzer()

tweets = []
entities = {}
tokens = []
punctuation = []
sents = {'pos': [], 'neg': [], 'neu': []}

is_punctuation = re.compile(r'[.!?\-,:;]+')

def get_wordnet_pos(pos: str):
    if pos.startswith("J"):
        return wordnet.ADJ
    elif pos.startswith("V"):
        return wordnet.VERB
    elif pos.startswith("N"):
        return wordnet.NOUN
    elif pos.startswith("R"):
        return wordnet.ADV
    else:
        return None


def normalize_tweet(tweet):
    global tweets
    global entities
    global tokens
    global punctuation
    tweets.append(tweet)
    tweet_tokens = word_tokenize(tweet)
    tokens_pos = pos_tag(tweet_tokens)
    chunks = ne_chunk(tokens_pos)
    for token in chunks:
        if type(token) == Tree:
            label = token.label()
            if label not in entities:
                entities[label] = []
            entities[label].append(token)
        if type(token) != Tree and is_punctuation.match(token[0]):
            punctuation.append(token)
        else:
            tokens.append(token)


if __name__ == "__main__":
    tweets_file = sys.argv[1]
    output_file = sys.argv[2]
    counter = 0
    max_counter = 100

    # tweet = "Hello, welcome to Colorado, Johny!"

    # start_time = time.time()
    # tweet_tokens = word_tokenize(tweet)
    # tokenisation_time = time.time() - start_time
    
    # start_time = time.time()
    # tokens_pos = pos_tag(tweet_tokens)
    # pos_time = time.time() - start_time
    
    # start_time = time.time()
    # chunks = ne_chunk(tokens_pos)
    # chunker_time = time.time() - start_time

    # print("tokensiation: {}, pos: {}, chunker: {}".format(tokenisation_time, pos_time, chunker_time))
    
    # exit(0)
    with open(tweets_file, "rb") as t_file:
        for tweet in t_file:
            tweet = tweet.decode()
            tweet = tweet.lower()
            #stats
            normalize_tweet(tweet)
            #sents
            scores = analyzer.polarity_scores(tweet)
            if scores['pos'] > scores['neg']:
                sents['pos'].append(tweet)
            elif scores['pos'] < scores['neg']:
                sents['neg'].append(tweet)
            else:
                sents['neu'].append(tweet.encode())            
            if counter % 10 == 0:
                print("{} tweets processed".format(counter))
            counter += 1
            if counter > max_counter:
                break

    with open(output_file, "w") as s_file:
        s_file.write("Number of processed: {}\n".format(len(tweets)))
        s_file.write("Number of tokens: {}\n".format(len(tokens)))
        s_file.write("Number of punctuation marks: {}\n".format(len(punctuation)))
        s_file.write("Entities:\n")
        for key in entities:
            s_file.write("    {} : {}\n".format(key, len(entities[key])))
        s_file.write("Sentiments:\n")
        for key in sents:
            s_file.write("    {} : {}\n".format(key, len(sents[key])))
        

    
        
