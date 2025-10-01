from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-2.5-flash')

app = FastAPI()

class ProjectInput(BaseModel):
    description: str

@app.post("/generate-tasks")
async def generate_tasks(input: ProjectInput):
    prompt = f"""
You are an Agile project planner. Based on this project description:
"{input.description}"

Generate the following:
- Epics
- User stories (with acceptance criteria)
- Tasks under each user story (with priority: High, Medium, Low)

IMPORTANT: Return ONLY valid JSON in the exact format below, with no additional text or explanation:

{{
  "epics": [
    {{
      "title": "Epic Title",
      "userStories": [
        {{
          "story": "As a user, I want...",
          "acceptanceCriteria": ["Criteria 1", "Criteria 2"],
          "tasks": [
            {{ "task": "Task description", "priority": "High" }},
            {{ "task": "Another task", "priority": "Medium" }}
          ]
        }}
      ]
    }}
  ]
}}
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000,
            )
        )
        result = response.text
        
        # Debug: print the raw response
        print("Raw response from Gemini:")
        print(result)
        print("---")
        
        # Try to extract JSON from the response
        try:
            # First, try to parse as-is
            return json.loads(result)
        except json.JSONDecodeError:
            # If that fails, try to find JSON within the response
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, return the raw text
                return {
                    "error": "No valid JSON found in response",
                    "raw_response": result
                }
    except Exception as e:
        return {"error": str(e)}
