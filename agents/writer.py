from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

class WriterAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7, 
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        )

    async def write_final_report(self, topic, research, critique):
        print(f"Writer is drafting the final report for: {topic}")
        
        prompt = f"""
        You are a Professional Technical Writer. 
        Your goal is to write a polished, high-quality research report on the topic: '{topic}'.
        
        Use the following Research Data:
        {research}
        
        And make sure to address or fix these points mentioned by the Critic:
        {critique}
        
        The report should have:
        1. An eye-catching Title.
        2. Executive Summary.
        3. Detailed Key Findings (well-structured).
        4. Future Outlook (specifically for 2026).
        5. A concluding thought.

        Write it in Markdown format. Make it sound human and authoritative.
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    # --- NAYA FUNCTION: Follow-up Chat ke liye ---
    async def chat_with_context(self, question, context):
        print(f"Writer is answering follow-up: {question}")
        
        prompt = f"""
        You are an AI Research Assistant. You have just completed a detailed research report.
        Based ONLY on the context provided below, answer the user's follow-up question.

        CONTEXT (The Research):
        {context}

        USER QUESTION:
        {question}

        INSTRUCTIONS:
        1. If the answer is in the context, provide a detailed but concise response.
        2. If the answer is NOT in the context, politely say that the current research doesn't cover this specific detail.
        3. Keep the tone professional and helpful.
        """
        
        response = self.llm.invoke(prompt)
        return response.content