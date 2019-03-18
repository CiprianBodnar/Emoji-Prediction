import nltk
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector



puncttok = nltk.WordPunctTokenizer().tokenize
social_tokenizer = SocialTokenizer(lowercase=False).tokenize
seg_tw = Segmenter(corpus="english")
sp = SpellCorrector(corpus="english") 

def tweet_preprocessing(tweet):

    #tweet_segm = text_processor.pre_process_doc(tweet)
    new_tweet = []
    for word in puncttok(tweet):
        if word[0] !='<' and word[-1] != '>':
            word = seg_tw.segment(word)
            word = sp.correct(word)
        new_tweet.append(word)
      
    return new_tweet



