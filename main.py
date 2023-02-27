from tensorflow import keras
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = keras.models.load_model('model.h5')
word_index = keras.datasets.imdb.get_word_index()


def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return keras.preprocessing.sequence.pad_sequences([tokens], maxlen=200)


@app.route('/', methods=['POST'])
def predict():
    try:
        text = request.get_json()['text']
        sentiment = model.predict(encode_text(text))[0][0]
        return jsonify({
            'label': 'positive' if sentiment > 0.5 else 'negative',
            'confidence': str(sentiment if sentiment > 0.5 else 1 - sentiment),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
