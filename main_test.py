from src.ingestion import IngestionSystem
from src.retrieval import RetrievalSystem
from src.generation import GeminiTutor

# 1. Setup
print("Initializing Systems...")
ingest = IngestionSystem()
retriever = RetrievalSystem()
tutor = GeminiTutor()

# 2. Run Pipeline
# Make sure you have your PDF in the 'data' folder or root
text = ingest.load_pdf("data/Dsa.pdf") # <--- CHANGE THIS FILENAME TO YOUR PDF
chunks = ingest.create_chunks(text)
retriever.embed_documents(chunks)

# 3. Ask
query = "What is the sorting algorithm with time complexity O(nlogn)?"
context_chunks = retriever.retrieve(query)
best_context = context_chunks[0] # Take top result

answer = tutor.ask(query, best_context)

print("\n🤖 ANSWER:")
print(answer)