import os
import torch
from flask import Flask, request, jsonify
from flask import send_from_directory
from utils import process_message_pipeline, check_empty_message
from model.pandy import ChatbotModel
app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['POST', 'GET'])
def home():
    return jsonify({'message':'Welcome to the pandemic Chatbot'})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', None) #getting the data from the request we sent

        check_empty_message(user_message)
        processed_message = process_message_pipeline(user_message) #processing the message



        response = { #returning the processed tokens
            "original_message": user_message,
            "processed_message": processed_message
#            "predicted_class": predicted_class
        }

        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/test', methods=['GET']) #test if the server is active
def test():
    return jsonify({"message": "API is working!"}), 200


if __name__ == '__main__':
    app.run(debug=True)

