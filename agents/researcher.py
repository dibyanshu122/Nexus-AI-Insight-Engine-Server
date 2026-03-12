from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

load_dotenv()

class ResearcherAgent:
    def __init__(self):
        # 2026 ka sabse stable model use kar rahe hain ab hum
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.3-70b-versatile" # Yeh decommission nahi hua hai
        )
        self.search = TavilySearchResults(k=5)

    async def execute_research(self, query: str):
        try:
            print(f"Agent is searching internet for: {query}")
            # Internet search
            search_results = self.search.invoke({"query": query})
            
            # Summary generation
            prompt = f"Summarize the following research data for the topic '{query}':\n\n{search_results}"
            summary = self.llm.invoke(prompt)
            
            return summary.content
        except Exception as e:
            return f"Error occurred: {str(e)}"