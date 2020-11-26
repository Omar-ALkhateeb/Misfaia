# Arabic sentimenet analysis API
An Open Source API to help filter arabic text and check for toxicity and/or offensiveness


### aknowledgements
 - dataset used [Arabic Sentiment Twitter Corpus](https://www.kaggle.com/mksaad/arabic-sentiment-twitter-corpus)
 - inspired by every other bad word filter and the lack of ararbic one's (as far as i know)



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


#### notice
this model was built using tensorflow '2.2.0' so if you have any problems with it i recommend changing to this specific version 