#!/usr/bin/env python
# -*- coding: utf-8 -*-
from base64 import b64decode

from flask_cors import CORS

import detectUtils
from flask import Flask, render_template, request, jsonify

# configuration
# DEBUG = False

# instantiate the app
app = Flask(__name__, template_folder='templates',
            static_folder='static')
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/detectAudio', methods=['GET', 'POST'])
def detect_audio():
    response_object = {'status': 'success'}
    if request.method == 'POST':
        post_data = request.get_json()
        # transform MIME string to bytes
        audio = b64decode(post_data.get('audio'))
        # call audio detection model here
        # then return its result

        label = detectUtils.detect_audio(audio)

        response_object['message'] = label

    return jsonify(response_object)


@app.route('/showResult')
def show_result():
    label = request.args.get('label')
    return render_template('show_file_template.html', label=label, is_image=True, is_show_button=False)


if __name__ == '__main__':
    # detect_audio()
    app.config['JSON_AS_ASCII'] = False
    app.config['DEBUG'] = True
    app.run()
