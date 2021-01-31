# Misfaia

## the Arabic toxicity filter API

![Misfaia](https://i.imgur.com/z5GuJHV.png)

An Open Source API to help filter arabic text and check for toxicity and/or offensiveness

### aknowledgements

- dataset used [Arabic-twitter-corpus-AJGT-](https://github.com/komari6/Arabic-twitter-corpus-AJGT#arabic-twitter-corpus-ajgt-)
- inspired by every other bad word filter and the lack of ararbic one's (as far as i know)

### todo

- [x] create basic model with +90% accuracy
  - [x] fixed model serialization issue
  - [ ] add pruning and quantization for faster model [ref1](http://digital-thinking.de/how-to-not-deploy-tensorflow-models-and-how-do-it-better/) [ref2](https://www.machinecurve.com/index.php/2020/09/23/tensorflow-model-optimization-an-introduction-to-pruning/)
- [x] wrap around an api for ease of use
  - [x] add CORS
  - [ ] jwt-auth
  - [x] rate-limiter
  - [x] security headers
- [ ] add a bad words filter
  - [ ] update curses list to a txt/csv file
  - [ ] fill the dictionary
- [ ] ingestion engine?

### routes

- /removeCurses --- bad word filter
- /getPrecentage --- AI
- /cleanCorpus --- remove punctuation and other chars
