from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from tools.vector import kg_qa
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from llm import llm
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are a movie expert. You find movies from a genre or plot.

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=chat_chain.run,
        return_direct=True
        ),
    # Tool.from_function(
    #     name="Vector Search Index",  # (1)
    #     description="Provides information about movie plots using Vector Search", # (2)
    #     func = kg_qa, # (3)
    #     return_direct=True
    # )
]

agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
    )
    
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    
    response = agent_executor.invoke({"input": prompt})
    print(response)
    if not isinstance(response['output'],str):
        return response['output'].content
    return response['output']