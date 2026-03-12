import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Force load environment variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# Safety Check: Agar keys missing hain toh server crash na kare, warning de
if not url or not key:
    print("❌ ERROR: Supabase credentials missing in .env file!")
    # Temporary dummy values taaki import error na aaye
    url = "https://placeholder.supabase.co"
    key = "placeholder"

supabase: Client = create_client(url, key)

def save_research(topic, research, critique, final_report):
    try:
        data = {
            "topic": topic,
            "researcher_output": research, 
            "critic_review": critique,
            "final_report": final_report
        }
        result = supabase.table("research_history").insert(data).execute()
        return result
    except Exception as e:
        print(f"Supabase Save Error: {e}")
        return None