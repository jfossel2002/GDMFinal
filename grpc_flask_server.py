from flask import Flask, request, jsonify
from flask_cors import CORS
import grpc
import UNKBOT_pb2
import UNKBOT_pb2_grpc

app = Flask(__name__)
CORS(app)  # Enable CORS

def run_prompt_bot_client(prompt):
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = UNKBOT_pb2_grpc.UNKServiceStub(channel)
        request = UNKBOT_pb2.UNKRequest(prompt=prompt)
        response = stub.Prompt_Bot(request)
        return response.response

def run_train_bot_client(prompt):
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = UNKBOT_pb2_grpc.UNKServiceStub(channel)
        request = UNKBOT_pb2.TrainRequest(prompt=prompt)
        response = stub.Train_Bot(request)
        return response.response

@app.route('/prompt-bot', methods=['POST'])
def prompt_bot():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        response = run_prompt_bot_client(prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-bot', methods=['POST'])
def train_bot():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    try:
        response = run_train_bot_client(prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
