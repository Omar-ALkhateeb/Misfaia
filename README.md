# Misfaia

## the Arabic toxicity filter API

![Misfaia](https://i.imgur.com/z5GuJHV.png)

An Open Source API to help filter arabic text and check for toxicity and/or offensiveness

### aknowledgements

- dataset used [Arabic-twitter-corpus-AJGT-](https://github.com/komari6/Arabic-twitter-corpus-AJGT#arabic-twitter-corpus-ajgt-)
- inspired by every other bad word filter and the lack of ararbic one's (as far as i know)

### todo

- [x] create basic model with +90% accuracy
- [x] wrap around an api for ease of use
  - [x] add CORS
  - [ ] jwt-auth
  - [x] rate-limiter
  - [x] security headers
- [ ] add a bad words filter
  - [ ] fill the dictionary

### routes

- /removeCurses --- bad word filter
- /getPrecentage --- AI
- /cleanCorpus --- remove punctuation and other chars

#### notice

this model was built using tensorflow '2.2.0' so if you have any problems with it i recommend changing to this specific version
