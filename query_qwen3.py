import os
from openai import OpenAI
import base64

# Configuration
# Replace with your actual compute node IP if it changes
COMPUTE_NODE_IP = "x24b200v4-nodes3-0"
MODEL_NAME = "./Qwen3-VL-235B"

client = OpenAI(
    base_url=f"http://{COMPUTE_NODE_IP}:8000/v1",
    api_key="token-not-needed"
)

def query_text(prompt):
    """Sends a standard text-based chat completion request."""
    print(f"\n--- Text Query: {prompt} ---")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content

def query_vision(prompt, image_url):
    """Sends a vision-language request using an image URL."""
    print(f"\n--- Vision Query: {prompt} ---")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

def query_local_vision(prompt, image_path):
    """Encodes a local image to base64 and sends a vision-language request."""
    print(f"\n--- Local Vision Query: {prompt} (File: {image_path}) ---")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example 1: Standard Text Completion
    print(query_text("Explain the concept of Mixture-of-Experts in three sentences."))

    # Example 2: Vision with URL
    # Using the official Qwen-VL demo image
    demo_url = "https://raw.githubusercontent.com/QwenLM/Qwen-VL/master/assets/demo.jpeg"
    print(query_vision("What is happening in this image?", demo_url))

    # Example 3: Local Image (Uncomment and provide path if needed)
    # print(query_local_vision("What is in this photo?", "test_image.jpg"))
