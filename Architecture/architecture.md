### __Tweet2Emoji__ 
_Multilingual Emoji Predictor_

* ### State of the art

   We found two architecture model, and possible solutions for this problem, both implemented for _SemEval-2018_.  

   Both solution rely on some sort of __preprocessing__ of the tweets, which includes _tokenisation_, _spell-checking_, and _normalizing_ of the words. Also _noise reduction_ plays a big role.
   
   1. First approach relies on __deep learning__, using a _Long Short-Term Memory Network_.
   2. Second approach uses __Naive Bayes Classifier__ machinge learning algorithm, based on _Bayes Theorem_ with the assumption of independence among predictors.
 
* ### Our architecture
    Since the results on this papers favor the _Naive Bayes Classifier_, we are inclined to implement this algorithm as our go-to solution, but we may use the _Deep Learning_ approach as well.

   For preprocessing and noise reduction we will use the [ekphrasis](https://github.com/cbaziotis/ekphrasis) tool, which can also provide useful insights of some words via annotations (hashtag, allcaps, elongated, repeated, emphasis, censored). We may try to remove _uninteresting_ words (such as connection words) for better results.

   * ### Arhitectural diagram
    ![Diagram](https://github.com/CiprianBodnar/Multilingual-Emoji-Prediction/Architecture/architecture.png?raw=True "Architectural Diagram")

