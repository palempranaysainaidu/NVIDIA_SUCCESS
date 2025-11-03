from openai import OpenAI


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)


response = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct", 
    messages=[
        {"role": "user", "content": "Provide me an article on Machine Learning."}
    ],
    temperature=0.7,
    max_tokens=1024,
    top_p=0.9
)


print(response.choices[0].message.content)
