# Arabic sentimenet analysis API

An Open Source API to help filter arabic text and check for toxicity and/or offensiveness

### aknowledgements

- dataset used [Arabic-twitter-corpus-AJGT-](https://github.com/komari6/Arabic-twitter-corpus-AJGT#arabic-twitter-corpus-ajgt-)
- inspired by every other bad word filter and the lack of ararbic one's (as far as i know)

### todo

- [x] create basic model with +60% accuracy
- [x] wrap around an api for ease of use
  - [ ] add CORS
  - [ ] jwt-auth
  - [ ] rate-limiter
- [ ] add a bad words filter

### routes

- /removeCurses --- bad word filter
- /getPrecentage --- AI
- /cleanCorpus --- remove punctuation and other chars

#### notice

this model was built using tensorflow '2.2.0' so if you have any problems with it i recommend changing to this specific version
