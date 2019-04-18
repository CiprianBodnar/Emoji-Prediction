import os
import sys
import math
import json
import time

from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tree import Tree
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

naive_bayes = NaiveBayesAnalyzer()

lemma_occurences = dict()
total_ocurrences = [0 for _ in range(0, 20)]
text_blob_pos = []
text_blob_neg = []


lemmatizer = WordNetLemmatizer()


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


# word_dict - dict where:
#   key: lemma
#   value: array of 20, each representing number of occurence for each emoji label
#
# totals - array of 20 representing total number of all lemma occurences for each emoji label
def compute_tfidf(word_dict: dict, totals: list) -> list:
    for key in word_dict:
        lemma_freq = word_dict[key]
        idf = math.log(
            20 / sum([1 if lemma_freq[i] > 0 else 0 for i in range(0, 20)]))

        lemma_freq = word_dict[key]
        tf = [lemma_freq[i] / totals[i] *
              idf for i in range(0, 20)] + [lemma_freq[20]]
        word_dict[key] = tf
    return word_dict


# word_dict - dict where:
#   key: lemma
#   value: array of 20, each value representing the tf_idf value for each emoji label
#
# emoji - int representing the emoji label for which to extract lemmas
# returns a list of tuples (lemma, tfidf) sorted by tfidf
def get_lemmas_for_emoji(word_dict: dict, emoji: list) -> list:
    lemmas = []
    for key in word_dict:
        emoji_freq = word_dict[key]
        if emoji_freq[emoji] > 0:
            lemmas.append((key, emoji_freq[emoji], emoji_freq[-1]))
    average_tfidf = sum([x[1] for x in lemmas]) / len(lemmas) * 1.3
    sorted_lemmas = sorted(lemmas, key=lambda x: x[1], reverse=True)

    return filter(lambda x: True if x[1] > average_tfidf else False, sorted_lemmas)


# tweet - a string representing content of a tweet
# returns lemmas of each word in tweet
def normalize_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens_pos = pos_tag(tokens)
    chunks = ne_chunk(tokens_pos)
    lemmas = []
    named_entities = []
    for token in chunks:
        if type(token) == Tree:
            named_entities.append(token)
            continue
        wn_pos = get_wordnet_pos(token[1])
        lemma = None
        if wn_pos:
            lemma = lemmatizer.lemmatize(token[0].lower(), pos=wn_pos)
        else:
            lemma = lemmatizer.lemmatize(token[0].lower())
        lemmas.append((lemma, wn_pos))
    return lemmas, named_entities


def get_lemma_syns(lemma, pos):
    synsets = None
    if pos:
        synsets = wordnet.synsets(lemma, pos)
    else:
        synsets = wordnet.synsets(lemma)

    syns = set()
    for synset in synsets:
        lemmas = synset.lemmas()
        if len(lemmas) > 0:
            syns.add(lemmas[0].name())

    return list(syns)


def get_lemma_hypers(lemma, pos):
    synsets = None
    if pos:
        synsets = wordnet.synsets(lemma, pos)
    else:
        synsets = wordnet.synsets(lemma)

    hypers = set()
    for synset in synsets:
        lemmas = synset.hypernyms()
        if len(lemmas) > 0:
            hypers.add(lemmas[0].name())

    return list(hypers)


# construct json object with lemmas for all emojis
def construct_emoji_obj(lemmas_tfidf):
    emojis = {}

    for label in range(0, 20):
        lemmas_for_emoji = get_lemmas_for_emoji(lemmas_tfidf, label)

        lemma_list = []
        for lemma in lemmas_for_emoji:
            lemma_obj = dict()
            lemma_pos = lemma[-1]

            lemma_obj['lemma'] = lemma[0]
            lemma_obj['tf_idf'] = lemma[1]
            lemma_obj['pos'] = lemma_pos
            lemma_obj['syns'] = get_lemma_syns(lemma[0], lemma_pos)
            lemma_obj['hypos'] = get_lemma_hypers(lemma[0], lemma_pos)

            lemma_list.append(lemma_obj)

        emojis[label] = lemma_list
    return emojis


if __name__ == "__main__":
    text_filename = sys.argv[1]
    labels_filename = sys.argv[2]
    output_filename = os.path.join(
        os.path.dirname(text_filename), "scoring.json")
    tweets = open(text_filename, 'rb')
    labels = open(labels_filename, 'rb')
    n_entities_file = open("named_entities.txt", "wb")
    entities = []
    counter = 0
    max_tweets = 500
    print("parsing file...")
    start_time = time.time()
    for tweet in tweets:
        # limit number of computed tweets
        counter += 1
        if counter > max_tweets:
            break

        tweet = tweet.decode()
        blob = TextBlob(tweet, analyzer=naive_bayes)
        if blob.sentiment.classification == 'pos':
            text_blob_pos.append(tweet.encode())
        else:
            text_blob_neg.append(tweet.encode())

        label = int(labels.readline().decode())
        lemmas, named_entities = normalize_tweet(tweet)

        for entity in named_entities:
            entities.append(entity)
        # prepare lemmas to compute tf_idf
        for lemma in lemmas:
            total_ocurrences[label] += 1
            emoji_freq = None
            if lemma[0] in lemma_occurences:
                emoji_freq = lemma_occurences[lemma[0]]
            else:
                emoji_freq = [0 for _ in range(0, 20)]
            emoji_freq[label] += 1
            lemma_occurences[lemma[0]] = emoji_freq + [lemma[1]]

    tweets.close()
    labels.close()
    for entity in entities:
        n_entities_file.write("{}\n".format(str(entity)).encode())
    n_entities_file.close()

    print("done in {}s".format(time.time() - start_time))
    print("computing TF-IDF on each lemma")
    start_time = time.time()
    lemmas_tfidf = compute_tfidf(lemma_occurences, total_ocurrences)
    print("done in {}s".format(time.time() - start_time))

    print("constructing json object...")
    start_time = time.time()
    emojis = construct_emoji_obj(lemmas_tfidf)
    print("done in {}s".format(time.time() - start_time))
    with open("textblob_neg.txt", "wb") as neg_file:
        neg_file.writelines(text_blob_neg)
    with open("textblob_pos.txt", "wb") as pos_file:
        pos_file.writelines(text_blob_pos)
    output = open(output_filename, "w")
    json.dump(emojis, output, indent=4)
    output.close()
