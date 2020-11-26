# Arabic sentimenet analysis API
an public use api to help filter data and check for toxicity and/or offensiveness in arabic input


### aknowledgements
dataset used [Arabic Sentiment Twitter Corpus](https://www.kaggle.com/mksaad/arabic-sentiment-twitter-corpus)



### todo
- [X] create basic model with +60% accuracy
- [X] wrap around an api for ease of use
    - [ ] add CORS
    - [ ] jwt-auth
    - [ ] rate-limiter
- [ ] add a bad words filter




### routes
- /removeCurses --- dictionary
- /getPrecentage --- AI
- /cleanCorpus --- remove punctuation
