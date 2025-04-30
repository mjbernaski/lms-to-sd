import requests
import json

def test_lmstudio_call(messages, description):
    print(f"\n=== Testing: {description} ===")
    url = "http://127.0.0.1:1234/v1/chat/completions"
    data = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 120
    }
    
    try:
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Raw Response:", json.dumps(result, indent=2))
            message = result["choices"][0]["message"]
            print("\nContent:", message.get("content"))
            if "reasoning_content" in message:
                print("\nReasoning Content:", message.get("reasoning_content"))
        else:
            print("Error Response:", response.text)
    except Exception as e:
        print(f"Error: {str(e)}")
    print("\n" + "="*50)

# Test different prompt formats
tests = [
    {
        "description": "Direct instruction",
        "messages": [
            {
                "role": "system",
                "content": "You are a prompt generator. Output exactly two lines:\nLine 1: [description]\nLine 2: Negative: [negative traits]"
            },
            {
                "role": "user",
                "content": "A detailed and realistic depiction of a tarantula, inspired by the intricate and dramatic style of Carlo Simi, with rich textures and deep shadows."
            }
        ]
    },
    {
        "description": "JSON format request",
        "messages": [
            {
                "role": "system",
                "content": "You are a prompt generator. Output in this format:\n{\"positive\":\"[description]\",\"negative\":\"[negative traits]\"}"
            },
            {
                "role": "user",
                "content": "A detailed and realistic depiction of a tarantula, inspired by the intricate and dramatic style of Carlo Simi, with rich textures and deep shadows."
            }
        ]
    },
    {
        "description": "Single line instruction",
        "messages": [
            {
                "role": "system",
                "content": "You are a prompt generator. Respond with a single line containing the image description."
            },
            {
                "role": "user",
                "content": "A detailed and realistic depiction of a tarantula, inspired by the intricate and dramatic style of Carlo Simi, with rich textures and deep shadows."
            }
        ]
    },
    {
        "description": "Simple completion",
        "messages": [
            {
                "role": "user",
                "content": "Complete this prompt for Stable Diffusion: A detailed and realistic depiction of a tarantula, inspired by the intricate and dramatic style of Carlo Simi, with rich textures and deep shadows."
            }
        ]
    }
]

for test in tests:
    test_lmstudio_call(test["messages"], test["description"])

def test_nothink():
    url = "http://127.0.0.1:1234/v1/chat/completions"
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a prompt generator. Output exactly two lines:\nLine 1: [description]\nLine 2: Negative: [negative traits]"
            },
            {
                "role": "user",
                "content": "/nothink A detailed and realistic depiction of a tarantula, inspired by the intricate and dramatic style of Carlo Simi, with rich textures and deep shadows."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 120
    }
    
    print("\n=== Testing with /nothink ===")
    try:
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Raw Response:", json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")

test_nothink()