from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
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


    #agent_executor.invoke(
    #    input={
    #        "input": """Generate and save in current working directory 15 QRcodes
    #                    that point to www.udemy.com/course/langchain, you have qrcode package installed already.
    #        """
    #    }
    #)

    csv_agent = create_csv_agent(
        llm=ChatOllama(model="gemma2:2b", temperature=0),
        path="USA_Housing.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    csv_agent.invoke(
        input={"input": """
            which variables have the highest and lowest correlation in file USA_Housing.csv?
            if you recieve an string error (ValueError: could not convert string to float), you need .  
            """}
    )



if __name__ == "__main__":
    main()
