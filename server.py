from secure import SecureHeaders
from flask_cors import CORS
from flask import Flask
import re
from tensorflow.keras.models import load_model

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

model = load_model("saved_model/full_model.tf")
app = Flask(__name__)

# rate limiter
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["15 per minute"]
)


CORS(app)  # This will enable CORS for all routes

secure_headers = SecureHeaders(csp=True, hsts=False, xfo="DENY")


@app.after_request
def set_secure_headers(response):
    secure_headers.flask(response)
    return response


# proto
curses = set(['كلب', 'حقير'])

# bad word filter
@app.route('/removeCurses/<words>')
def removeCurses(words=None):
    # lowercase our set to easily match bad words
    setWords = set(words.lower().split())
    words = words.split()
    badWords = setWords.intersection(curses)
    print(words)
    if len(badWords) > 0:
        res = [i if i not in badWords else '*'*len(i) for i in words]
        return ' '.join(res)

    return ' '.join(words)


# ml model prediction from 0 "bad" to 1 "good"
@app.route('/getPrecentage/<words>')
def getPrecentage(words=None):
    # workaround tensorflow needing array input
    print(words)
    return str(model.predict([words, words, words])[0][0])
    # print(model.predict([words, words, words]))


# sentence filter and cleaner
@app.route('/cleanCorpus/<words>')
def cleanCorpus(words=None):
    # print(words)
    # fix regex to keep punctuation and remove other chars
    filtered = re.sub('[^A-Za-z0-9ا-ي]+', ' ', words)
    return filtered


app.run(port=12000)
