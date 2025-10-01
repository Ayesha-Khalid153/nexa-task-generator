import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables")
    print("Please set your API key in the .env file")
    exit(1)

genai.configure(api_key=api_key)

try:
    # List all available models
    print("Available models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")
    
    # Test a simple generation
    print("\nTesting model...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content("Hello, this is a test.")
    print("Test successful!")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTry these alternative model names:")
    print("- gemini-1.5-flash")
    print("- models/gemini-1.5-flash")
    print("- gemini-1.5-pro")
    print("- models/gemini-1.5-pro")