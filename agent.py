from langchain.agents import AgentExecutor, create_react_agent, create_openai_functions_agent
from langchain import hub
from tools.vector import kg_qa
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from llm import llm
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain.agents import load_tools

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


# prompt = PromptTemplate(
#     template="""
#     You are a movie expert. You find movies from a genre or plot.

#     Chat History:{chat_history}
#     Question:{input}
#     """,
#     input_variables=["chat_history", "input"],
# )

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

# chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# @tool("vector-search", return_direct=True)
# def vector_search(query: str) -> str:
#     """Look up things online."""
#     return kg_qa.invoke({'query':query})

# @tool("Vector Search Index",return_direct=True)
# def vector_search_index(query: str) -> str:
#     """Provides information about movie plots using Vector Search."""
#     res = kg_qa.invoke({'query':query})
#     return res['result']

# class CustomSearchTool(BaseTool):
#     name = "vector_search_index_1"
#     description = "Provides information about movie plots using Vector Search."

#     def _run(
#         self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         res = kg_qa.invoke({'query':query})
#         return res['result']

def search_function(query: str):
    res = kg_qa.invoke({'query':query})
    return res['result']


# search = Tool.from_function(
#     func=search_function,
#     name="vector_search_index_2",
#     description = "Provides information about movie plots using Vector Search."
# )


tools = [
    # Tool.from_function(
    #     func=search_function,
    #     name="vector_search_index_2",
    #     description = "Provides information about movie plots using Vector Search."
    # ),
    # Tool.from_function(
    #     name="General Chat",
    #     description="For general chat not covered by other tools",
    #     func=chat_chain.run,
    #     return_direct=True
    #     ),
    Tool.from_function(
        name="Vector Search Index",  # (1)
        description="Provides information about movie plots using Vector Search", # (2)
        func = kg_qa.invoke, # (3)
        return_direct=True,
    )
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    # handle_parsing_errors=True
    )
    
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    # response = agent_executor.invoke({"input": prompt})
    # {'query': 'Toy Story', 'result': 'Yes, the answer is "Toy Story."'}

    # input/output : {'output': {'query': 'Toy Story', 'result': 'Yes, the answer is "Toy Story."'}}
    #  ??? ({'input': 'What movie name toy story about', 'chat_history': []}, {'output': {'query': 'Toy Story', 'result': 'Yes, the answer is "Toy Story."'}})
    # response = kg_qa.invoke({"query": prompt})
    # {'query': 'What movie name toy story about', 'result': "The movie Toy Story is about a cowboy doll named Woody who feels threatened and jealous when a new spaceman toy named Buzz Lightyear becomes the top toy in a boy's room."}
    response = tools[0].run({"input": prompt})
    # {'query': 'What movie name toy story about', 'result': 'The movie "Toy Story" is about a cowboy doll named Woody who feels threatened and jealous when a new spaceman figure named Buzz Lightyear becomes the top toy in a boy\'s room.'}
    # response = vector_search_index({"query": prompt})
    # print(response)
    return response['result']
    # response = tools[0].run(prompt)
    # print(response)
    # if 'output' not in response:
    #     return response
    # if not isinstance(response['output'],str):
    #     return response['output'].content
    # return response['output']