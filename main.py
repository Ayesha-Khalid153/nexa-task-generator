from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import requests, json, os, re, random
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Use stable, supported Gemini model
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction="You are an AI Agile Project Planner that outputs structured JSON for software project task planning."
)

NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL")
app = FastAPI()


# --- Pydantic input ---
class ProjectInput(BaseModel):
    project_id: str
    description: str
    auth_token: str = None


# --- Fallback generator ---
def generate_fallback_tasks(description: str, team: list):
    print("⚙️ Using fallback task generator")
    if not team:
        team = [{"name": "Unassigned", "role": "Developer"}]
    
    priorities = ["High", "Medium", "Low"]
    basic_tasks = [
        "Set up development environment",
        "Create project documentation",
        "Design user interface mockups",
        "Implement core functionality",
        "Write unit tests",
        "Perform code review",
        "Deploy to staging environment",
        "Conduct user testing"
    ]
    
    epics = [
        {"title": "Project Setup", "userStories": [{"story": f"Set up project foundation for {description}", "acceptanceCriteria": ["Setup complete"], "tasks": []}]},
        {"title": "Core Development", "userStories": [{"story": f"Develop main functionality of {description}", "acceptanceCriteria": ["Core features complete"], "tasks": []}]}
    ]

    for i, member in enumerate(team):
        if i < len(basic_tasks):
            epics[i % 2]["userStories"][0]["tasks"].append({
                "task": basic_tasks[i],
                "assignedTo": member.get("name", "Unassigned"),
                "role": member.get("role", "Developer"),
                "priority": priorities[i % 3],
                "deadlineDays": random.randint(3, 10),
                "status": "In Progress" if i == 0 else "Backlog",
                "queueOrder": i + 1
            })
    return {"epics": epics}


# --- Core route ---
@app.post("/generate-tasks")
async def generate_tasks(input: ProjectInput):
    print(f"🚀 Generating tasks for project {input.project_id}")

    # --- Step 1: Fetch team ---
    team = []
    try:
        team_url = f"{NODE_BACKEND_URL}/api/projectMember/{input.project_id}"
        headers = {"Authorization": f"Bearer {input.auth_token}"} if input.auth_token else {}
        print(f"📡 Fetching team data from {team_url}")
        resp = requests.get(team_url, headers=headers, timeout=10)

        if resp.status_code == 200:
            team_data = resp.json()
            team = team_data.get("team", [])
            print(f"👥 Team members: {[m.get('name') for m in team]}")
        else:
            print(f"⚠️ Team API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"⚠️ Error fetching team: {e}")

    if not team:
        team = [{"name": "Unassigned", "role": "Developer"}]

    # --- Step 2: Generate tasks with Gemini ---
    team_json = json.dumps(team, indent=2)
    prompt = f"""
You are an **AI Agile Project Planner** helping plan the project described below.

### Project
{input.description}

### Team
{team_json}

Generate a **comprehensive backlog** with ~80–90 tasks distributed across 6–8 epics, 2–4 user stories each.

Output ONLY valid JSON in this format:

{{
  "epics": [
    {{
      "title": "Epic Title",
      "userStories": [
        {{
          "story": "As a user, I want ...",
          "acceptanceCriteria": ["Criterion 1", "Criterion 2"],
          "tasks": [
            {{
              "task": "Task title",
              "assignedTo": "Member Name",
              "role": "Member Role",
              "priority": "High | Medium | Low",
              "deadlineDays": <int>,
              "status": "Backlog | In Progress",
              "queueOrder": <int>
            }}
          ]
        }}
      ]
    }}
  ]
}}

### Important:
- Each team member can have only ONE task with `"status": "In Progress"`.
- All other tasks assigned to that member must be `"Backlog"`.
- Assign `"queueOrder"` sequentially per member.
- Base task assignments on team roles.
- Use realistic task names and durations (2–7 days).
- Return ONLY valid JSON — no markdown, no commentary.
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.4,
                max_output_tokens=12000,  # allow large output
            )
        )

        if not response.candidates or not response.candidates[0].content.parts:
            print("❌ Empty response from Gemini — using fallback")
            return generate_fallback_tasks(input.description, team)

        result_text = response.text.strip()
        print(f"🧠 Gemini response (first 300 chars):\n{result_text[:300]}...\n")

        # --- Clean JSON output ---
        if result_text.startswith("```"):
            result_text = re.sub(r"^```[a-zA-Z]*\n", "", result_text)
            result_text = re.sub(r"\n```$", "", result_text)

        try:
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed: {e}")
            match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass

            print("🔄 Falling back due to JSON parsing failure")
            return generate_fallback_tasks(input.description, team)

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return generate_fallback_tasks(input.description, team)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
