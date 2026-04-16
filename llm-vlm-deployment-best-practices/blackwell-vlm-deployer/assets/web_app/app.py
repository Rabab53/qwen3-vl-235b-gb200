import os
import base64
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace with your actual compute node IP if it changes
VLLM_SERVER_URL = "http://x24b200v4-nodes3-0:8000/v1/chat/completions"
MODEL_NAME = "./Qwen3-VL-235B"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_text = data.get('text', '')
        image_data = data.get('image', None) # Base64 string from frontend

        # Prepare the payload for vLLM
        messages_content = [{"type": "text", "text": user_text}]
        
        if image_data:
            # image_data is already a base64 string from the frontend
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": messages_content}],
            "max_tokens": 1000
        }

        # Proxy the request to the vLLM server on the compute node
        response = requests.post(VLLM_SERVER_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return jsonify({"status": "success", "answer": answer})
        else:
            return jsonify({"status": "error", "message": f"vLLM Error: {response.text}"}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 so it's accessible via the login node's IP
    app.run(host='0.0.0.0', port=5001, debug=False)
