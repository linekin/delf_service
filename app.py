import json
import os
import tempfile
import logging
logging.basicConfig(level=logging.INFO)

from flask import Flask, request, jsonify

from delf_features import extract, load_features, match_features, Feature
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

features_dir = Path(os.getenv('FEATURES_DIR'))
haystack = load_features(features_dir)

app = Flask(__name__)


@app.route("/match", methods=['POST'])
def match():
    if 'file' in request.files:
        file = request.files['file']
        fp = tempfile.NamedTemporaryFile()
        file.save(fp)

        locations, descriptors = list(extract([fp.name]))[0]

        res = match_features(Feature(locations, descriptors), haystack)

        return jsonify(res)
    else:
        # error status
        pass


if __name__ == '__main__':
    app.run()
