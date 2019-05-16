import json
import sys
import re

from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

word_re = re.compile(r"\w+")

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


json_file_name = sys.argv[1]
json_file = open(json_file_name)
json_object = json.load(json_file)
json_file.close()

# tweet - a string representing content of a tweet
# returns lemmas of each word in tweet
def normalize_tweet(tweet):
    tokens = word_tokenize(tweet)
    tokens_pos = pos_tag(tokens)
    lemmas = []
    for token in tokens_pos:
        wn_pos = get_wordnet_pos(token[1])
        lemma = None
        if wn_pos:
            lemma = lemmatizer.lemmatize(token[0].lower(), pos=wn_pos)
        else:
            lemma = lemmatizer.lemmatize(token[0].lower())
        if word_re.match(lemma):
            lemmas.append((lemma, wn_pos))
    return lemmas

def get_occurences_for_lemma(lemma, lemmas_list:list):
    for l in lemmas_list:
        if l['lemma'] == lemma:
            return l['occurences']
    return 0

def word_prob_for_label(word, label):
    global json_object
    n_k = get_occurences_for_lemma(word[0], json_object[str(label)])
    n_label = json_object['totals']['occurences'][label]
    voc_len = json_object['totals']['voc_len']
    prob = (n_k + 1) / (n_label + voc_len)
    print("Probability that {} is labeled {}: {}".format(word, label, prob))
    return (n_k + 1) / (n_label + voc_len)

def tweet_prob_for_label(tokens, label):
    product = 1
    for token in tokens:
        product *= word_prob_for_label(token, label)
    return 1/20 * product

def classify_tweet(tokens):
    probabilities = [tweet_prob_for_label(tokens, label) for label in range(0, 20)]
    prob_list = [i for i in range(0, 20)]
    return sorted(prob_list, key=lambda x: probabilities[x], reverse=True)

if __name__ == "__main__":
    tweet = input("Enter tweet: ")
    lemmas = normalize_tweet(tweet)
    
    print(classify_tweet(lemmas))