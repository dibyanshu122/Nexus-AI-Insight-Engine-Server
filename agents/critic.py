from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

class CriticAgent:
    def __init__(self):
        # Iske liye hum ek sophisticated model use karenge jo logic mein mahir ho
        self.llm = ChatGroq(
            temperature=0.2, # Thodi creativity taaki critical thinking achhi ho
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

    async def evaluate_research(self, research_data: str, original_topic: str):
        print(f"Critic is evaluating research for: {original_topic}")
        
        prompt = f"""
        As an expert Research Critic, evaluate the following AI-generated research summary for the topic: '{original_topic}'.
        
        RESEARCH DATA TO EVALUATE:
        {research_data}
        
        Your task is to:
        1. Check for accuracy and depth.
        2. Identify if any major 2026 trends are missing.
        3. Rate the research out of 10.
        4. Suggest 2-3 specific improvements.

        Return your response in a clear, professional tone.
        """
        
        response = self.llm.invoke(prompt)
        return response.content