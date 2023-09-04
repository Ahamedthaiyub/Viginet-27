
from flask import Flask, request, jsonify
import train  
import processing
import unsw
import classify

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def start_training():
    if request.method == 'POST':
        try:
            params = request.json
            
            result = train.train_model(params)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    
    return jsonify({"message": "Metrics endpoint"}), 200

if __name__ == '__main__':
    app.run(debug=True)
