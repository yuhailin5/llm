from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os


from langchain_openai import ChatOpenAI

# 这个包已经弃用
#from langchain_community.llms import tongyi
#from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage

# 文本嵌入
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def get_weather(city):
    """
    this is a function to get the weather of params city, you just call it
    """
    return f"The weather in {city} is sunny with a high of 25°C and a low of 15°C."

""" llm=ChatOpenAI(
    model='qwen3.7-plus',
    api_key=os.getenv('TONGYI_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) """

model=ChatOpenAI(
    model='qwen3.7-plus',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)




""" agent=create_agent(
    model=llm,
    system_prompt="you are a poet"
) """

messages = [
    HumanMessage(content="写一首唐诗")
]

res = model.stream(input=messages)

for chunk in res:
    print(chunk.content,end="",flush=True)