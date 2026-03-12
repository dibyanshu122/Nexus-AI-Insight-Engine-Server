import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class MemoryAgent:
    def __init__(self):
        # Pinecone Connection
        api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = "nexus-research"
        self.index = self.pc.Index(self.index_name)
        
        # Model: all-mpnet-base-v2 (768 Dimensions)
        print("Memory Agent: Loading Embedding Model...")
        self.model = SentenceTransformer('all-mpnet-base-v2') 
        print("Memory Agent: Model Loaded and Connected to Pinecone.")

    async def store_research_vector(self, topic, content):
        try:
            print(f"Memory Agent: Vectorizing research for {topic}...")
            vector = self.model.encode(content).tolist()
            
            # Upsert using strict keyword arguments
            self.index.upsert(vectors=[{
                "id": topic.replace(" ", "_").lower(), 
                "values": vector, 
                "metadata": {"topic": topic, "content": content} 
            }])
            print(f"Memory Agent: Successfully stored vectors for '{topic}'")
        except Exception as e:
            print(f"Memory Agent Error (Store): {e}")

    async def search_context(self, question):
        """Follow-up question ke liye relevant context dhoondna"""
        try:
            query_vector = self.model.encode(question).tolist()
            print(f"Memory Agent: Searching Pinecone for context...")
            
            # FIXED: Naye SDK mein positional arguments zero tolerance hain
            # Sab kuch explicitly define kar diya hai
            results = self.index.query(
                vector=query_vector,
                top_k=1, 
                include_metadata=True
            )
            
            # Naye SDK mein results ek object hota hai, matches list hoti hai
            if results and hasattr(results, 'matches') and len(results.matches) > 0:
                match = results.matches[0]
                
                # Check metadata exists (Dot notation is safer in newer SDKs)
                if hasattr(match, 'metadata') and match.metadata and 'content' in match.metadata:
                    print("Memory Agent: Context found successfully!")
                    return match.metadata['content']
            
            print("Memory Agent: No relevant context found in memory.")
            return ""
        except Exception as e:
            # Detailed error logging
            print(f"Memory Agent Error (Search): {str(e)}")
            return ""