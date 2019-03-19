import nltk
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize


puncttok = nltk.WordPunctTokenizer().tokenize
social_tokenizer = SocialTokenizer(lowercase=False).tokenize
seg_tw = Segmenter(corpus="english")
sp = SpellCorrector(corpus="english") 
stop_words = set(stopwords.words('english'))

def tweet_preprocessing(tweet):
    
    #tweet_segm = text_processor.pre_process_doc(tweet)
    lemma = []
    for word in puncttok(tweet):
        if word[0] !='<' and word[-1] != '>' and word not in stop_words:
            word = seg_tw.segment(word)
            word = sp.correct(word)
            lemma.append(word)
      
    return lemma



print(tweet_preprocessing("CANT WAIT for the new season of #TwinPeaks ＼(^o^)／ yaaaay!!! #davidlynch #tvseries :)))"))