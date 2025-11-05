from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents import create_csv_agent
from langchain import hub

load_dotenv()

def main():
    
    instructions = """You are an agent designed to write and execute Python code to answer questions.
                      You have access to a Python REPL tool to run code and obtain results.
                      If you get an error, debug your code and try again.
                      Only use the output of your code to answer the question.
                      You might know the answer without running code, but you should still run code to get the answer.
                      If it does not seem like you can write code to answer the question, just return "I don't know" as the final answer."""
    
    tools = [PythonREPLTool()]
    
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(
        instructions=instructions,
        tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools])
    )

    agent = create_react_agent(
        prompt=prompt,
        llm=ChatOllama(model="gemma2:2b", temperature=0),
        tools=tools,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    csv_agent: AgentExecutor = create_csv_agent(
        llm=ChatOllama(model="gemma2:2b", temperature=0),
        path="USA_Housing.csv",
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=10,
        prefix="""You are working with a pandas dataframe in Python. The dataframe is called 'df'.
        
IMPORTANT: The columns in this CSV are:
- 'Avg. Area Income'
- 'Avg. Area House Age' 
- 'Avg. Area Number of Rooms'
- 'Avg. Area Number of Bedrooms'
- 'Area Population'
- 'Price'
- 'Address'

ALWAYS use df.columns to check column names FIRST before using them.
When filtering by bedrooms, use the column 'Avg. Area Number of Bedrooms'.
"""
    )

    def python_agent_wrapper(question: str) -> str:
        result = agent_executor.invoke({"input": question})
        return result.get("output", str(result))
    
    def csv_agent_wrapper(question: str) -> str:
        result = csv_agent.invoke({"input": question})
        return result.get("output", str(result))

    tools = [
        Tool(
            name="Python_Agent",
            func=python_agent_wrapper,
            description="Useful for when you need to transform natural language to python and execute python code, returning the results of execution. Input should be a natural language question or task. DOES NOT ACCEPT CODE AS INPUT."
        ),
        Tool(
            name="CSV_Agent",
            func=csv_agent_wrapper,
            description="Useful for when you need to analyze and answer questions about data in a CSV file. Input should be a question about the data in the CSV."
        )
    ]

    prompt = base_prompt.partial(instructions="")
    router_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOllama(model="gemma2:2b", temperature=0),
        tools=tools,
    )

    router_agent_executor = AgentExecutor.from_agent_and_tools(agent=router_agent, tools=tools, verbose=True)

    print(
        router_agent_executor.invoke(
            {
                "input": "Using the USA_Housing.csv file, what is the average 'Price' of houses with more than 3 bedrooms?"
            }
        )
    )

    print(
        router_agent_executor.invoke(
            {
                "input": "Generate a list of the first 10 Fibonacci numbers and calculate their sum."
            }
        )
    )

if __name__ == "__main__":
    main()
