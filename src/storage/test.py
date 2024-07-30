import openai


openai.api_key = "sk-proj-RzleBYai8C0kEWAzfpYTT3BlbkFJqK6ANK6SfW8xFW6w2Tjd"


def chatbot(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"].strip()


if __name__ == "__main__":
    print("Chatbot: Hi there! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot(user_input)
        print(f"Chatbot: {response}")
