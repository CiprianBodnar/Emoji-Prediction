#Author: Nurendra Choudhary.
#Algorithm Reference: http://en.wikipedia.org/wiki/Lesk_algorithm

from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
import sys



def overlapcontext( synset, sentence ):
    gloss = set(word_tokenize(sentence))
    for i in synset.examples():
         gloss.union(i)
   
    if isinstance(sentence, str):
        sentence = set(sentence.split(" "))
    elif isinstance(sentence, list):
        sentence = set(sentence)
    elif isinstance(sentence, set):
        pass
    else:
        return
    return len( gloss.intersection(sentence) )

def lesk( word, sentence ):
    bestsense = None
    maxoverlap = 0
    for sense in wordnet.synsets(word):
        overlap = overlapcontext(sense,sentence)
        for h in sense.hyponyms():
            overlap += overlapcontext( h, sentence )
        if overlap > maxoverlap:
                maxoverlap = overlap
                bestsense = sense
    return bestsense


sentence = input("Enter the Sentence (or) Context :")
word = input("Enter the word :")

a = lesk(word,sentence)
print ("\n\nSynset:",a)
if a is not None:
    print ("Meaning:",a.definition())
    num=0
    print ("\nExamples:")
    for i in a.examples():
        num=num+1
        print (str(num)+'.'+')',i)
