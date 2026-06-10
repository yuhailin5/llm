from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool,ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver # 记忆模块
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from pydantic import BaseModel,Field


# TavilySearch 联网搜索工具

class student(BaseModel):
    name: str = Field(description="学生姓名",default='xiaoming')


# @tool(args_schema=student) 传入参数的描述类

SYSTEM_PROMPT = """
    你是一个每日天气查询助手，你可以使用一下两种工具：
    1. get_weather: 传入查询城市，返回天气
    2. sent_email: 将天气信息发送至该邮箱
    3. get_user_email: 获取用户邮箱
如果用户查询某城市天气，查询完天气后，发送至用户邮箱！
"""

chat=init_chat_model(
    model='deepseek-chat'
)

@tool
def sent_email(email:str,weather:str) -> str:
    """
    发送天气至用户邮箱
    """
    return f"已将{weather} 发送至用户邮箱 {email}"
    

@tool
def get_weather(city:str) -> str:
    """
    get the weather of params
    """
    return f"the weather of {city} is sunny"

# 根据用户id获取邮箱

@dataclass
class Context:
    """
    自定义运行时上下文模式
    """
    user_email: str


@tool
def get_user_email(runtime: ToolRuntime[Context]) -> str:
    """
    根据用户id获取用户邮箱
    """
    user_email = runtime.context.user_email
    return user_email


# 使用pydantic 导入并继承BaseModel 结构化输出

@dataclass
class ResponseFormat:
    """
    自定义响应格式
    """
    # 幽默回应
    punny_response: str
    # 邮箱信息
    email: str
    # 天气信息 有默认的放最后
    weather_condition: str | None = None

checkpoint = InMemorySaver()


# thread_id为对话标识
config = {
    "configurable": {"thread_id": "user_123"}
}


tools = [get_weather, sent_email, get_user_email]

# context_schema=Context 注册你的上下文结构，response_format自定义响应格式
model=create_agent(model='deepseek-chat',
                   system_prompt=SYSTEM_PROMPT,
                   tools=tools,
                   context_schema=Context,
                   response_format=ResponseFormat,
                   checkpointer=checkpoint
                   )

# 调用加上上下文包装
res = model.invoke({
    "messages": [
        HumanMessage(content="你好，我是林哥，重庆的天气怎么样？")
    ]
},
    context=Context(user_email='yhl@tianya.com'),
    config=config
    )
for msg in res['messages']:
    msg.pretty_print()

res2 = model.invoke(
    {
        "messages": [
            HumanMessage(content="我叫啥名字呀")
        ]
    },
    context=Context(user_email='yhl@tianya.com'),
    config=config
)
for msg in res2['messages']:
    msg.pretty_print()
