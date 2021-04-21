from app import app
from flask import render_template, url_for, jsonify, request
import os
from imageio import imread
from app.pythonscript import widthModifier, heightModifier

@app.route('/')
@app.route('/index', methods=["POST"])
def index():
    return render_template('index.html')

@app.route('/change_width/', methods=['POST', 'GET'])
def change_width():
    response = request.args.get('width')
    print(response)
    response = response[:-2]
    widthModifier(int(response))
    return jsonify(response)

@app.route('/change_height/', methods=['POST', 'GET'])
def change_height():
    response = request.args.get('height')
    print(response)
    response = response[:-2]
    heightModifier(int(response))
    return jsonify(response)