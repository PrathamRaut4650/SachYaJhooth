import numpy as np
from flask import Flask,request,render_template
from flask_cors import CORS
import os
import pickle
import flask
import os
from newspaper import Article
import urllib
import nltk
nltk.download('punkt')

#loading the flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open('model.pkl','rb') as handle:
    model = pickle.load(handle)
    
    @app.route('/')
    def main():
        return render_template('index.html')
    
    
#receiving the input url from the user and web scraping to extract the news comntent
@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        url =request.get_data(as_text=True)[5:]
        url = urllib.parse.unquote(url)
        article = Article(str(url))
        article.download()
        article.parse()
        article.nlp()
        news = article.summary
        #passing the news article to model and returning it is fake or real
        pred = model.predict([news])
        return render_template('prediction.html', prediction_text='THE NEWS IS "{}"'.format(pred[0]))
    else:
        return render_template('prediction.html')

if __name__=="__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
