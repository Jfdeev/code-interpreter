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


    #agent_executor.invoke(
    #    input={
    #        "input": """Generate and save in current working directory 15 QRcodes
    #                    that point to www.udemy.com/course/langchain, you have qrcode package installed already.
    #        """
    #    }
    #)

    csv_agent: AgentExecutor = create_csv_agent(
        llm=ChatOllama(model="gemma2:2b", temperature=0),
        path="USA_Housing.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    #csv_agent.invoke(
    #    input={"input": """
    #        Analyze the USA_Housing.csv file and find which NUMERIC variables have the highest and lowest correlation.
    #        
    #        IMPORTANT INSTRUCTIONS:
    #        1. First, use df.select_dtypes(include=['number']) to get ONLY numeric columns
    #        2. Then use df.corr() on the numeric data only
    #        3. Find the highest and lowest correlation pairs (excluding diagonal values which are 1.0)
    #        4. DO NOT try to calculate correlation on string columns like Address
    #        
    #        Example code structure:
    #        import pandas as pd
    #        df = pd.read_csv('USA_Housing.csv')
    #        numeric_df = df.select_dtypes(include=['number'])
    #        correlation = numeric_df.corr()
    #        # Then find highest and lowest correlations
    #        """}
    #)

    tools = [
        Tool(
            name="Python Agent",
            func=agent_executor.invoke,
            description="Useful for when you need to transform normal language to python and execute python code, returning the results of execution. DOES NOT ACCEPT CODE AS INPUT."
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="Useful for when you need to analyze and answer questions about data in a CSV file. Input should be a question about the data in the CSV."
        )
    ]



if __name__ == "__main__":
    main()
