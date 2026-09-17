# Most modern LLM providers and local runtimes expose an OpenAI-compatible API interface,
# making the openai Python SDK the standard way to programmatically query almost any model.
#
# Python Implementation (pip install openai)


from openai import OpenAI

# 1. Initialize the client
client = OpenAI(
    base_url="https://api.openai.com/v1",  # Swap endpoint for other providers/local LLMs
    api_key="YOUR_API_KEY"                 # Use any non-empty string for local models
)

# 2. Send a chat request
response = client.chat.completions.create(
    model="gpt-4o-mini",                  # Target model name
    messages=[
        {"role": "system", "content": "You are a concise AI assistant."},
        {"role": "user", "content": "Explain APIs in one sentence."}
    ],
    temperature=0.7
)

# 3. Print the output
print(response.choices[0].message.content)



# Provider Adapter Configurations
#
# To point the script to different backends, simply swap the base_url, api_key, and model parameters:
#
# Local Ollama: base_url="http://localhost:11434/v1", api_key="ollama", model="mistral"
#
# DeepSeek: base_url="[https://api.deepseek.com](https://api.deepseek.com)", api_key="YOUR_KEY", model="deepseek-chat"
#
# Groq: base_url="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)", api_key="YOUR_KEY", model="llama-3.3-70b-versatile"
#
# vLLM / LM Studio: base_url="http://localhost:8000/v1", api_key="EMPTY", model="YOUR_LOADED_MODEL"