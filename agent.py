import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import TOOL_KIT

load_dotenv()


class Agent:
    def __init__(self, instructions:str, model:str="gpt-4o-mini"):

        # Initialize the LLM

        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            base_url="https://openai.vocareum.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.instructions = instructions

        # Create the Energy Advisor agent
        self.graph = create_react_agent(
            # name="energy_advisor",
            # prompt=SystemMessage(content=instructions),
            model=llm,
            tools=TOOL_KIT,
        )



    def invoke(self, question: str, context:str=None) -> str:
        """
        Ask the Energy Advisor a question about energy optimization.
        
        Args:
            question (str): The user's question about energy optimization
            location (str): Location for weather and pricing data
        
        Returns:
            str: The advisor's response with recommendations
        """
        
        messages = [SystemMessage(content=self.instructions)]
        if context:
            messages.append(SystemMessage(content=context))
        messages.append(HumanMessage(content=question))

        response = self.graph.invoke({"messages": messages})

        
        return response

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor"""
        return [t.name for t in TOOL_KIT]
