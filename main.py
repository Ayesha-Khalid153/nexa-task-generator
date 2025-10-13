from fastapi import FastAPI
from pydantic import BaseModel, Field
from google.generativeai import GenerationConfig 
import google.generativeai as genai
import requests, json, os, re, random
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- CRITICAL FIX 1: Define FinishReason gracefully for version compatibility ---
# If the direct import fails, define the enum values based on the API documentation.
try:
    # Attempt the standard import (for newer/standard SDKs)
    from google.generativeai.types import FinishReason
except ImportError:
    print("⚠️ Warning: google.generativeai.types.FinishReason not found. Defining custom FinishReason enum for compatibility.")
    # Define a custom class to mimic the enum for older/different SDKs
    class FinishReason:
        # 0: FINISH_REASON_UNSPECIFIED
        STOP = 1
        SAFETY = 2
        RECITATION = 3
        MAX_TOKENS = 4
        # ... other reasons aren't needed for this logic

# --- Pydantic Schema for Guaranteed Output ---
class TaskModel(BaseModel):
    task: str = Field(description="A detailed title for the specific task.")
    assignedTo: str = Field(description="The exact 'Member Name' from the Team list responsible for the task.")
    role: str = Field(description="The role of the assigned member, derived from the team list.")
    priority: str = Field(description="Priority: High, Medium, or Low.")
    deadlineDays: int = Field(description="Estimated days to complete the task (2-7 days).")
    status: str = Field(description="Initial status: Backlog or In Progress.")
    queueOrder: int = Field(description="Sequential order of this task in the member's personal queue.")

class UserStoryModel(BaseModel):
    story: str = Field(description="User story written in the format: 'As a user, I want...'")
    acceptanceCriteria: List[str] = Field(description="List of criteria that define when the story is complete.")
    tasks: List[TaskModel]

class EpicModel(BaseModel):
    title: str = Field(description="The title of the Epic, grouping related user stories.")
    userStories: List[UserStoryModel]

class TaskGeneratorResponse(BaseModel):
    epics: List[EpicModel]


# --- Load environment ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Use stable, supported Gemini model
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction="You are an AI Agile Project Planner that MUST adhere strictly to the provided JSON schema for all outputs. You are a component of the NEXA multi-agent suite, responsible for Task Generation, assignment, and workload balancing."
)

NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL")
app = FastAPI()


# --- Pydantic input ---
class ProjectInput(BaseModel):
    project_id: str
    description: str
    auth_token: str = None


# --- Fallback generator (Unchanged) ---
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


# --- Role-based assignment post-processing (Unchanged) ---
def assign_tasks_by_role(result: dict, team: list) -> dict:
    if not result or 'epics' not in result:
        return result

    member_map = {}
    for m in team:
        name = m.get('name', 'Unassigned')
        member_map[name] = {
            'name': name,
            'role': (m.get('role') or 'Developer').strip(),
            'assigned': 0,
            'tasks': [] 
        }
    
    priority_order = {'high': 3, 'medium': 2, 'low': 1}

    for epic in result.get('epics', []):
        for us in epic.get('userStories', []):
            for task in us.get('tasks', []):
                assigned_name = task.get('assignedTo')
                
                mem = member_map.get(assigned_name)
                
                if not mem:
                    unassigned_member = member_map.setdefault(
                        "Unassigned", 
                        {'name': 'Unassigned', 'role': 'Developer', 'assigned': 0, 'tasks': []}
                    )
                    mem = unassigned_member
                    task['assignedTo'] = 'Unassigned'
                else:
                    task['assignedTo'] = mem['name']

                task['role'] = mem['role']
                mem['tasks'].append(task)
                mem['assigned'] += 1
    
    for name, mem in member_map.items():
        if not mem['tasks']:
            continue
            
        mem['tasks'].sort(key=lambda x: (
            priority_order.get(x.get('priority', '').lower(), 0), 
            x.get('queueOrder', 999) 
        ), reverse=True) 

        for i, task in enumerate(mem['tasks']):
            task['queueOrder'] = i + 1
            if i == 0:
                task['status'] = 'In Progress' 
            else:
                task['status'] = 'Backlog'

    return result

# Helper: extract balanced JSON by scanning braces (Unchanged)
def extract_balanced_json(s: str):
    start = s.find('{')
    if start == -1:
        return None
    i = start
    depth = 0
    in_string = False
    esc = False
    quote_char = None
    while i < len(s):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == quote_char:
                in_string = False
        else:
            if ch == '"' or ch == "'":
                in_string = True
                quote_char = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        i += 1
    return None

# Helper: escape raw newlines AND unescaped quotes inside JSON string literals (Unchanged, robust)
def escape_newlines_in_strings(s: str):
    out = []
    in_str = False
    esc = False
    quote = None
    for ch in s:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == '\\':
                out.append(ch)
                esc = True
            elif ch == quote:
                out.append(ch)
                in_str = False
            elif ch == '\n':
                out.append('\\n')
            elif ch in ('\t', '\r', '\f'):
                out.append('\\' + ch) 
            elif ch == '"' and quote == '"':
                out.append('\\"')
            elif ch == "'" and quote == "'":
                out.append("\\'")
            else:
                out.append(ch)
        else:
            if ch == '"' or ch == "'":
                in_str = True
                quote = ch
                out.append(ch)
            else:
                out.append(ch)
    return ''.join(out)


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

    if not team or not any(m.get('name') == 'Unassigned' for m in team):
        team.append({"name": "Unassigned", "role": "Developer"})

    # --- Step 2: Generate tasks with Gemini ---
    team_json = json.dumps(team, indent=2)
    prompt = f"""
You are an **AI Agile Project Planner** working on a **safe, technical project planning task**.
This is purely a professional software project management context.

The project to plan is:
\"\"\"{input.description}\"\"\"

### Team
{team_json}

Generate a **comprehensive backlog** with ~20–25 tasks distributed across 3–4 epics, 2–4 user stories each.

### Important Instructions:
- This request does **not** involve any unsafe, unethical, or harmful content.
- Focus only on **technical tasks**, **software engineering**, and **project management**.
- Assign each task to a team member based on their role.
- Return **ONLY valid JSON** (no markdown, no commentary, no triple backticks).
"""
    try:
        config = GenerationConfig(
            temperature=0.7, 
            response_mime_type="application/json", 
            response_schema=TaskGeneratorResponse,
            max_output_tokens=12000,
        )

        response = model.generate_content(
            prompt,
            generation_config=config,
        )
        
        # --- CRITICAL FIX 3: Robust Safety Check before accessing .text ---
        # We rely on the FinishReason definition established at the top of the file.
        if not response.candidates or response.candidates[0].finish_reason != FinishReason.STOP:
            if response.candidates:
                reason = response.candidates[0].finish_reason
                print(f"❌ Gemini API error: Response blocked/incomplete. Finish Reason: {reason}")
                if reason == FinishReason.SAFETY:
                    print("Hint: Content was blocked due to safety settings. Try simplifying the project description.")
            else:
                print("❌ Gemini API error: No candidates were returned.")
            
            return generate_fallback_tasks(input.description, team)

        # Proceed only if the response is valid
        raw_result = response.text.strip()
        print(f"🧠 Gemini response (first 300 chars):\n{raw_result[:300]}")
        
        # Final safety cleanup for any leftover issues (like trailing commas from the LLM)
        cleaned_json = re.sub(r',\s*([}\]])', r'\1', raw_result, flags=re.MULTILINE)
        
        try:
            parsed_result = json.loads(cleaned_json)
            print("✅ Successfully parsed Gemini JSON (Schema Enforced)")
            return assign_tasks_by_role(parsed_result, team)
        except json.JSONDecodeError as e:
            # Fallback to the extremely robust manual cleanup using helper functions
            print(f"⚠️ JSON parse failed (Schema enforced, falling back to manual cleanup): {e}")
            
            fixed = escape_newlines_in_strings(raw_result)
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed, flags=re.MULTILINE)

            try:
                parsed_result = json.loads(fixed)
                print("✅ Successfully parsed JSON after manual cleanup")
                return assign_tasks_by_role(parsed_result, team)
            except json.JSONDecodeError as e2:
                print(f"⚠️ Failed to parse even after manual cleanup: {e2}")
                print("🔄 Falling back due to catastrophic JSON failure")
                return generate_fallback_tasks(input.description, team)

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        if hasattr(e, '__traceback__'):
             import traceback
             print(traceback.format_exc())
        return generate_fallback_tasks(input.description, team)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)