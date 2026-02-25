import google.generativeai as genai
import os
from dotenv import load_dotenv
from google.api_core import exceptions

load_dotenv()

class GeminiTutor:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API Key not found in .env file")
            
        genai.configure(api_key=api_key)
        
        # 1. Fetch a list of all models YOUR specific key is allowed to use
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 2. Hunt for the best "Flash" model automatically
        selected_model_name = None
        
        # We check the list and grab the first one that has "flash" in the name
        for name in available_models:
            if 'flash' in name.lower():
                selected_model_name = name
                break # Found one! Stop looking.
                
        # 3. Fallback just in case "flash" is totally gone
        if not selected_model_name:
            # Just grab the very first text-generation model available
            selected_model_name = available_models[0] 
            
        print(f"🔌 Auto-Connected to AI Model: {selected_model_name}")
        self.model = genai.GenerativeModel(selected_model_name)
    
    def ask(self, query, context, chat_history):
        # 1. Format History
        history_text = ""
        for msg in chat_history[-6:]: 
            role = "Student" if msg["role"] == "user" else "Tutor"
            history_text += f"{role}: {msg['content']}\n"

        # 2. Build Prompt
        prompt = f"""
        Act as a university tutor. Answer the student's question based strictly on the context provided.
        
        CONTEXT FROM TEXTBOOK:
        {context}
        
        CONVERSATION HISTORY:
        {history_text}
        
        CURRENT QUESTION: {query}
        
        INSTRUCTIONS:
        - Answer in 2-3 sentences.
        - If the answer isn't in the context, say "I don't know."
        """
        
        # 3. STREAMING GENERATION
        try:
            # We add stream=True here
            response = self.model.generate_content(prompt, stream=True)
            
            # Instead of returning all at once, we "yield" each chunk as it arrives
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except exceptions.ResourceExhausted:
            yield "❌ **Quota Exceeded.** You have hit the Google Free Tier limit. Please wait 1-2 minutes."
        except Exception as e:
            yield f"❌ **Error:** {str(e)}"
    

    def generate_study_guide(self, chat_history):
        # 1. Check if there's actually a conversation to summarize
        if len(chat_history) < 2:
            return "❌ You need to chat with the textbook first to generate a study guide!"

        # 2. Format the history
        history_text = ""
        for msg in chat_history: 
            role = "Student" if msg["role"] == "user" else "Tutor"
            history_text += f"{role}: {msg['content']}\n"

        # 3. The "Study Guide" Prompt
        # Notice how we specifically instruct it to look for DSA concepts
        prompt = f"""
        Review the following conversation between a student and a tutor.
        Compile the successfully explained concepts into a structured Study Guide.
        
        CONVERSATION HISTORY:
        {history_text}
        
        STRICT INSTRUCTIONS:
        - ONLY include information, data structures, and algorithms that the Tutor explicitly explained and provided answers for.
        - DO NOT bring in outside knowledge. If the Tutor said "I don't know" or failed to answer a question, completely ignore that topic.
        - Use clear Markdown headings (##).
        - Use bullet points, bold text, and code blocks where appropriate.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating guide: {str(e)}"