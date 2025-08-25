from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

base_url = 'http://llama:8000/v1'
model = 'meta/llama-3.1-8b-instruct'
llm = ChatNVIDIA(base_url=base_url, model=model, temperature=0)

statements = [
    "I had a fantastic time hiking up the mountain yesterday.",
    "The new restaurant downtown serves delicious vegetarian dishes.",
    "I am feeling quite stressed about the upcoming project deadline.",
    "Watching the sunset at the beach was a calming experience.",
    "I recently started reading a fascinating book about space exploration."
]

# sentiments analysis
sentiment_template = ChatPromptTemplate.from_template("""Give Positive or Negative sentiment for all combined in one single word: {statements}""")
prompt = sentiment_template.invoke(statements)
print(llm.invoke(prompt).content)
# output = Positive

# Main Topic Identification
main_topic_template = ChatPromptTemplate.from_template('''Identify the main topic: {statements}''')
prompt = main_topic_template.invoke(statements)
print(llm.invoke(prompt).content)
# output = The main topic is: Personal experiences and activities.

# Followup Question Generation
followup_template = ChatPromptTemplate.from_template('''Be as concise as possible and Generate an appropriate follow up question: {statements}''')
prompt = followup_template.invoke(statements)
print(llm.invoke(prompt).content)
# output = What was the most enjoyable part of your day yesterday?

