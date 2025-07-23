from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()

# New OpenAI client setup (v1.x)
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        result = response.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        return {"error": str(e)}
