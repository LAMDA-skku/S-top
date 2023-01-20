import os 
import pymysql
import argparse
import json
import pandas as pd 
from attrdict import AttrDict
from flask import Flask, render_template, request

app = Flask(__name__, static_folder='./static/')

@app.route('/', methods=['GET', 'POST'])
def main_page():
    user_input = ''
    score = 9
    emotions = 'depressed'
    if request.method == 'POST':
        user_input = request.form['user_text']
        print(user_input)

    return render_template('main.html', user_input=user_input, score=score, emotions=emotions)

def main():
    # app.run(host="192.168.123.110", debug=True, port=9509)
    app.run(debug=True)

if __name__ == '__main__':
    main()