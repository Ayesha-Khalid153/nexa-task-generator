from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

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

Return the output in this JSON format:

{{
  "epics": [
    {{
      "title": "...",
      "userStories": [
        {{
          "story": "...",
          "acceptanceCriteria": ["...", "..."],
          "tasks": [
            {{ "task": "...", "priority": "High" }}
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
                max_output_tokens=1000,
            )
        )
        result = response.text
        return json.loads(result)
    except Exception as e:
        return {"error": str(e)}
