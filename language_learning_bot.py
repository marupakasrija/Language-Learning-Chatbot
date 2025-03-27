import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import uuid

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field  # Updated import
from langchain.memory import ConversationBufferMemory

# Define Pydantic models for structured outputs
class LanguageCorrection(BaseModel):
    original: str = Field(description="The original text with the error")
    correction: str = Field(description="The corrected text")
    explanation: str = Field(description="Explanation of why this is an error and how to fix it")

class ErrorAnalysis(BaseModel):
    has_errors: bool = Field(description="Whether the text contains language errors")
    corrections: List[LanguageCorrection] = Field(default_factory=list, description="List of corrections if errors exist")

class ProgressAnalysis(BaseModel):
    strengths: List[str] = Field(description="Language strengths demonstrated by the user")
    areas_to_improve: List[str] = Field(description="Areas where the user needs improvement")
    recommended_practice: str = Field(description="Recommended practice activity")

class LanguageLearningBot:
    def __init__(self, api_key: str = None):
        """Initialize the language learning chatbot."""
        # Set Google API key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("Google API key is required. Please provide it as an argument or set the GOOGLE_API_KEY environment variable.")
        
        # Configure Google Gemini
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        
        # Initialize language models
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
        self.analysis_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
        
        # Initialize database
        self.init_database()
        
        # Initialize conversation state
        self.chat_id = None
        self.target_language = None
        self.native_language = None
        self.proficiency_level = None
        self.scenario = None
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)  # Added memory_key
        
        # Initialize parsers
        self.error_parser = PydanticOutputParser(pydantic_object=ErrorAnalysis)
        self.progress_parser = PydanticOutputParser(pydantic_object=ProgressAnalysis)
    
    def init_database(self):
        """Initialize SQLite database with required tables."""
        self.conn = sqlite3.connect('language_learning.db')
        self.cursor = self.conn.cursor()
        
        # Create chats table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            target_language TEXT NOT NULL,
            native_language TEXT NOT NULL,
            proficiency_level TEXT NOT NULL,
            scenario TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create messages table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
        ''')
        
        # Create mistakes table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS mistakes (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            original TEXT NOT NULL,
            correction TEXT NOT NULL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
        ''')
        
        self.conn.commit()
    
    def create_chat(self, target_language: str, native_language: str, proficiency_level: str, scenario: str) -> str:
        """Create a new chat session."""
        self.chat_id = str(uuid.uuid4())
        self.target_language = target_language
        self.native_language = native_language
        self.proficiency_level = proficiency_level
        self.scenario = scenario
        
        # Store in database
        self.cursor.execute(
            "INSERT INTO chats (id, target_language, native_language, proficiency_level, scenario) VALUES (?, ?, ?, ?, ?)",
            (self.chat_id, target_language, native_language, proficiency_level, scenario)
        )
        self.conn.commit()
        
        # Clear memory for new chat
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)  # Added memory_key
        
        # Generate initial greeting
        greeting = self.generate_greeting()
        
        # Store assistant message
        message_id = str(uuid.uuid4())
        self.cursor.execute(
            "INSERT INTO messages (id, chat_id, role, content) VALUES (?, ?, ?, ?)",
            (message_id, self.chat_id, "assistant", greeting)
        )
        self.conn.commit()
        
        # Add to memory
        self.memory.chat_memory.add_ai_message(greeting)
        
        return greeting
    
    def generate_greeting(self) -> str:
        """Generate an initial greeting based on the chat settings."""
        template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful language learning assistant for {target_language}.
            The user's native language is {native_language} and their proficiency level is {proficiency_level}.
            The conversation scenario is: {scenario}
            
            Generate a friendly greeting to start the conversation. 
            Introduce yourself as a language tutor and explain that you'll help them practice {target_language}.
            Mention the scenario and invite them to start practicing.
            
            For beginners, use simple language and include some {native_language} translation.
            For intermediate and advanced learners, use primarily {target_language}.
            """),
            ("human", "Please generate an appropriate greeting to start our language learning session.")
        ])
        
        # Using the recommended pipe syntax instead of LLMChain
        chain = template | self.llm
        
        response = chain.invoke({
            "target_language": self.target_language,
            "native_language": self.native_language,
            "proficiency_level": self.proficiency_level,
            "scenario": self.scenario
        })
        
        # Extract the content from the response dictionary
        return response.content
    
    def analyze_message(self, message: str) -> ErrorAnalysis:
        """Analyze a user message for language errors."""
        template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a language teacher for {target_language}. 
            The user's native language is {native_language} and their proficiency level is {proficiency_level}.
            
            Analyze this message for any language mistakes:
            "{message}"
            
            If there are mistakes, provide corrections. If there are no mistakes, indicate that.
            Be very precise and only mark actual language errors (grammar, vocabulary, syntax, etc.).
            
            {format_instructions}
            """),
            ("human", "Analyze the language errors in this message.")
        ])
        
        # Using the recommended pipe syntax
        chain = template | self.analysis_llm
        
        response = chain.invoke({
            "target_language": self.target_language,
            "native_language": self.native_language,
            "proficiency_level": self.proficiency_level,
            "message": message,
            "format_instructions": self.error_parser.get_format_instructions()
        })
        
        try:
            return self.error_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing analysis: {e}")
            # Return default if parsing fails
            return ErrorAnalysis(has_errors=False, corrections=[])
    
    def process_message(self, message: str) -> Tuple[str, List[str]]:
        """Process a user message and return the assistant's response and any corrections."""
        if not self.chat_id:
            return "Please start a new chat session first.", []
        
        # Store user message
        message_id = str(uuid.uuid4())
        self.cursor.execute(
            "INSERT INTO messages (id, chat_id, role, content) VALUES (?, ?, ?, ?)",
            (message_id, self.chat_id, "user", message)
        )
        self.conn.commit()
        
        # Add to memory
        self.memory.chat_memory.add_user_message(message)
        
        # Analyze for errors
        analysis = self.analyze_message(message)
        
        # Store mistakes if any
        if analysis.has_errors and analysis.corrections:
            for correction in analysis.corrections:
                mistake_id = str(uuid.uuid4())
                self.cursor.execute(
                    "INSERT INTO mistakes (id, chat_id, message_id, original, correction, explanation) VALUES (?, ?, ?, ?, ?, ?)",
                    (mistake_id, self.chat_id, message_id, correction.original, correction.correction, correction.explanation)
                )
            self.conn.commit()
        
        # Generate response
        response = self.generate_response()
        
        # Store assistant message
        assistant_message_id = str(uuid.uuid4())
        self.cursor.execute(
            "INSERT INTO messages (id, chat_id, role, content) VALUES (?, ?, ?, ?)",
            (assistant_message_id, self.chat_id, "assistant", response)
        )
        self.conn.commit()
        
        # Add to memory
        self.memory.chat_memory.add_ai_message(response)
        
        # Format corrections for display
        corrections = []
        if analysis.has_errors and analysis.corrections:
            for c in analysis.corrections:
                corrections.append(f"{c.original} → {c.correction}: {c.explanation}")
        
        return response, corrections
    
    def generate_response(self) -> str:
        """Generate an assistant response based on the conversation history."""
        template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful language learning assistant for {target_language}.
            The user's native language is {native_language} and their proficiency level is {proficiency_level}.
            The conversation scenario is: {scenario}
            
            Guidelines:
            1. Primarily respond in {target_language} at an appropriate level for {proficiency_level}
            2. Keep responses conversational and natural
            3. Stay in character for the scenario: {scenario}
            4. Use simple language for beginners, more complex for advanced
            5. Occasionally use {native_language} only to explain complex concepts
            6. Be encouraging and supportive
            """),
            ("human", "{chat_history}\nPlease respond to the latest message.")
        ])
        
        # Using the recommended pipe syntax
        chain = template | self.llm
        
        chat_history = self.memory.load_memory_variables({})["history"]
        
        response = chain.invoke({
            "target_language": self.target_language,
            "native_language": self.native_language,
            "proficiency_level": self.proficiency_level,
            "scenario": self.scenario,
            "chat_history": chat_history
        })
        
        return response.content
    
    def get_mistakes(self) -> List[Dict[str, str]]:
        """Get all mistakes for the current chat."""
        if not self.chat_id:
            return []
        
        self.cursor.execute(
            "SELECT original, correction, explanation FROM mistakes WHERE chat_id = ? ORDER BY created_at DESC",
            (self.chat_id,)
        )
        
        mistakes = []
        for row in self.cursor.fetchall():
            mistakes.append({
                "original": row[0],
                "correction": row[1],
                "explanation": row[2]
            })
        
        return mistakes
    
    def get_progress(self) -> Optional[ProgressAnalysis]:
        """Generate a progress analysis for the current chat."""
        if not self.chat_id:
            return None
        
        # Get all user messages
        self.cursor.execute(
            "SELECT content FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY created_at ASC",
            (self.chat_id,)
        )
        messages = self.cursor.fetchall()
        
        # Get all mistakes
        self.cursor.execute(
            "SELECT original, correction, explanation FROM mistakes WHERE chat_id = ? ORDER BY created_at ASC",
            (self.chat_id,)
        )
        mistakes = self.cursor.fetchall()
        
        # If not enough data, return None
        if len(messages) < 3:
            return None
        
        template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a language teacher for {target_language}. 
            The user's native language is {native_language} and their proficiency level is {proficiency_level}.
            
            Analyze the user's language learning progress based on their messages and mistakes.
            
            User messages:
            {user_messages}
            
            Mistakes made:
            {mistakes}
            
            Provide a structured analysis of the user's progress.
            
            {format_instructions}
            """),
            ("human", "Analyze my language learning progress.")
        ])
        
        # Using the recommended pipe syntax
        chain = template | self.analysis_llm
        
        user_messages_formatted = "\n".join([f"{i+1}. {msg[0]}" for i, msg in enumerate(messages)])
        
        if mistakes:
            mistakes_formatted = "\n".join([f"{i+1}. \"{m[0]}\" → \"{m[1]}\": {m[2]}" for i, m in enumerate(mistakes)])
        else:
            mistakes_formatted = "No recorded mistakes."
        
        response = chain.invoke({
            "target_language": self.target_language,
            "native_language": self.native_language,
            "proficiency_level": self.proficiency_level,
            "user_messages": user_messages_formatted,
            "mistakes": mistakes_formatted,
            "format_instructions": self.progress_parser.get_format_instructions()
        })
        
        try:
            # Updated to use model_dump_schema instead of model_json_schema
            schema = dict(self.progress_parser.pydantic_object.model_dump_schema().items())
            return self.progress_parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing progress analysis: {e}")
            # Return default if parsing fails
            return ProgressAnalysis(
                strengths=["Unable to analyze strengths"],
                areas_to_improve=["Unable to analyze areas to improve"],
                recommended_practice="Please continue practicing to generate a progress analysis."
            )
    
    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()

# Command-line interface for the chatbot
def run_cli():
    print("=" * 50)
    print("Language Learning Assistant (Google Gemini)")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your Google API key (or press Enter if set as environment variable): ").strip()
    if not api_key and "GOOGLE_API_KEY" not in os.environ:
        print("Error: Google API key is required.")
        return
    
    # Initialize bot
    bot = LanguageLearningBot(api_key if api_key else None)
    
    # Available options
    languages = [
        "Spanish", "French", "German", "Italian", "Portuguese", 
        "Japanese", "Chinese", "Korean", "Russian", "Arabic", "English"
    ]
    
    proficiency_levels = [
        "Beginner (A1)", "Elementary (A2)", "Intermediate (B1)", 
        "Upper Intermediate (B2)", "Advanced (C1)", "Proficient (C2)"
    ]
    
    scenarios = [
        "At a restaurant", "Shopping", "Asking for directions", 
        "At a hotel", "Making friends", "Business meeting", 
        "Medical appointment", "Public transportation"
    ]
    
    # Setup new chat
    print("\nLet's set up your language learning session:")
    
    print("\nAvailable languages:")
    for i, lang in enumerate(languages):
        print(f"{i+1}. {lang}")
    
    target_idx = int(input("\nSelect the language you want to learn (number): ")) - 1
    target_language = languages[target_idx]
    
    native_idx = int(input(f"\nSelect your native language (number, different from {target_language}): ")) - 1
    native_language = languages[native_idx]
    
    print("\nProficiency levels:")
    for i, level in enumerate(proficiency_levels):
        print(f"{i+1}. {level}")
    
    level_idx = int(input("\nSelect your proficiency level (number): ")) - 1
    proficiency_level = proficiency_levels[level_idx]
    
    print("\nConversation scenarios:")
    for i, scene in enumerate(scenarios):
        print(f"{i+1}. {scene}")
    
    scenario_idx = int(input("\nSelect a conversation scenario (number): ")) - 1
    scenario = scenarios[scenario_idx]
    
    # Start chat
    print("\nStarting conversation...\n")
    greeting = bot.create_chat(target_language, native_language, proficiency_level, scenario)
    print(f"Assistant: {greeting}\n")
    
    # Main conversation loop
    try:
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            if user_input.lower() == 'mistakes':
                mistakes = bot.get_mistakes()
                print("\n--- Your Mistakes ---")
                if mistakes:
                    for i, m in enumerate(mistakes):
                        print(f"{i+1}. \"{m['original']}\" → \"{m['correction']}\"")
                        print(f"   Explanation: {m['explanation']}")
                        print()
                else:
                    print("No mistakes recorded yet. Keep practicing!")
                print("-------------------\n")
                continue
            
            if user_input.lower() == 'progress':
                progress = bot.get_progress()
                print("\n--- Your Progress ---")
                if progress:
                    print("Strengths:")
                    for s in progress.strengths:
                        print(f"- {s}")
                    
                    print("\nAreas to improve:")
                    for a in progress.areas_to_improve:
                        print(f"- {a}")
                    
                    print(f"\nRecommended practice: {progress.recommended_practice}")
                else:
                    print("Not enough data to analyze progress. Keep practicing!")
                print("-------------------\n")
                continue
            
            if user_input.lower() == 'help':
                print("\n--- Commands ---")
                print("mistakes - View your language mistakes")
                print("progress - View your learning progress")
                print("exit/quit/bye - End the conversation")
                print("help - Show this help message")
                print("----------------\n")
                continue
            
            # Process regular message
            response, corrections = bot.process_message(user_input)
            
            # Show corrections if any
            if corrections:
                print("\n--- Language Corrections ---")
                for i, correction in enumerate(corrections):
                    print(f"{i+1}. {correction}")
                print("---------------------------\n")
            
            print(f"Assistant: {response}\n")
    
    except KeyboardInterrupt:
        print("\nExiting conversation...")
    
    finally:
        # Close database connection
        bot.close()
        print("\nThank you for practicing! Goodbye.")

if __name__ == "__main__":
    run_cli()