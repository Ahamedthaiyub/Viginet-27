from flask import Flask, request, jsonify

app = Flask(__name__)


models = {
    'model1': {
        'name': 'Model1',
        'status': 'Idle',
    },
    'model2': {
        'name': 'Model2',
        'status': 'Idle',
    }
}

@app.route('/train/Model1', methods=['POST'])
def train_model(model_name):
    if model_name in models:
        
        models[model_name]['status'] = 'Training'
        return jsonify({'message': f'Training {models[model_name]["name"]}'}), 202
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route('/predict/Model2', methods=['POST'])
def predict(model_name):
    if model_name in models:
        
        models[model_name]['status'] = 'Predicting'
        return jsonify({'message': f'Predicting using {models[model_name]["name"]}...'}), 202
    else:
        return jsonify({'error': 'Model not found'}), 404

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': models}), 200

if __name__ == '__main__':
    app.run(debug=True)
