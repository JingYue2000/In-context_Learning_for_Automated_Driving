from openai import OpenAI


def send_to_chatgpt(api_key, structured_message):
    client = OpenAI(api_key=api_key)

    # Replace the messages below with your structured_message
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are an assistant, skilled in analyzing driving scenarios."},
            {"role": "user", "content": structured_message}
        ]
    )

    return completion.choices[0].message

def extract_decision(response):
    try:
        start = response.find('"decision": {') + len('"decision": {')
        end = response.find('}', start)
        decision = response[start:end].strip('"')
        return decision
    except Exception as e:
        print(f"Error in extracting decision: {e}")
        return None


def main():
    # Replace with your actual API Key
    with open('MY_KEY.txt', 'r') as f:
        api_key = f.read().strip()

    # Example structured message (replace with actual message)
    structured_message = "Your analysis of the driving situation..."

    # Send message and get response
    chatgpt_response = send_to_chatgpt(api_key, structured_message)
    print("ChatGPT Response:", chatgpt_response)


if __name__ == "__main__":
    main()
