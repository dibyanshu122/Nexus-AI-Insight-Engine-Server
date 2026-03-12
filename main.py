from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.writer import WriterAgent
from agents.memory import MemoryAgent 
from core.database import save_research

app = FastAPI()

# --- CORS Configuration (RE-FIXED) ---
# allow_origins=["*"] aur allow_credentials=True aksar Axios mein error dete hain.
# Isliye origins ko explicitly allow karna behtar hai.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://192.168.1.18:3000" # Jo aapka IP log mein tha
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agents Initialization ---
researcher = ResearcherAgent()
critic = CriticAgent()
writer = WriterAgent()
memory_agent = MemoryAgent()

@app.get("/test-full-workflow")
async def test_workflow(topic: str):
    try:
        # 1. Research Process
        research_summary = await researcher.execute_research(topic)
        critic_review = await critic.evaluate_research(research_summary, topic)
        final_report = await writer.write_final_report(topic, research_summary, critic_review)
        
        # 2. Pinecone mein Store karo (Memory)
        await memory_agent.store_research_vector(topic, final_report)
        
        # 3. Supabase mein Save karo (History)
        try:
            save_research(topic, research_summary, critic_review, final_report)
        except Exception as e:
            print(f"⚠️ DB Error (History not saved): {e}")
        
        return {
            "topic": topic,
            "researcher_output": research_summary,
            "critic_review": critic_review,
            "final_report": final_report
        }
    except Exception as e:
        print(f"❌ Workflow Error: {e}")
        return {"error": str(e)}

@app.get("/ask-agent")
async def ask_agent(question: str):
    try:
        # 1. Pinecone se context dhoondo
        context = await memory_agent.search_context(question)
        
        if not context or context == "":
            return {"answer": "Mera memory abhi khali hai. Pehle kuch research kijiye taaki main jawab de sakun!"}

        # 2. RAG Prompt Construction
        prompt = f"""
        You are a helpful Research Assistant. 
        Use the following retrieved context to answer the user's question accurately.
        If the answer is not in the context, say that you don't have that specific information yet.
        
        Context: {context}
        
        Question: {question}
        Answer:"""
        
        # 3. LLM Call
        response = researcher.llm.invoke(prompt)
        
        return {"answer": response.content}
    except Exception as e:
        print(f"❌ Chat Error: {e}") 
        return {"answer": f"Maaf kijiye, system mein kuch gadbad hui: {str(e)}"}

# Root route taaki 404 na aaye
@app.get("/")
async def root():
    return {"status": "Nexus AI Backend is Running"}