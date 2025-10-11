from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import requests, json, os, re, random
from dotenv import load_dotenv

# --- Load environment ---
# Make sure GOOGLE_API_KEY and NODE_BACKEND_URL are set in your .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Use stable, supported Gemini model
model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction="You are an AI Agile Project Planner that outputs structured JSON for software project task planning. You are a component of the NEXA multi-agent suite, responsible for Task Generation."
)

NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL")
app = FastAPI()


# --- Pydantic input ---
class ProjectInput(BaseModel):
    project_id: str
    description: str
    auth_token: str = None


# --- Fallback generator (Unchanged for reliability) ---
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


# --- Role-based assignment post-processing (SMARTENED) ---
def assign_tasks_by_role(result: dict, team: list) -> dict:
    """
    Validates and cleans up LLM assignments based on hard constraints:
    1. Ensures all tasks have an 'assignedTo' from the team list (or 'Unassigned' fallback).
    2. Enforces the rule: max ONE 'In Progress' task per team member.
    3. Re-calculates and enforces 'queueOrder' sequentially per member based on LLM priority.
    """
    if not result or 'epics' not in result:
        return result

    # 1. Map team members for easy lookup and tracking
    member_map = {}
    for m in team:
        name = m.get('name', 'Unassigned')
        member_map[name] = {
            'name': name,
            'role': (m.get('role') or 'Developer').strip(),
            'assigned': 0,
            'tasks': [] # To store all tasks assigned to this member
        }
    
    # Define priority order for consistent 'In Progress' selection
    priority_order = {'high': 3, 'medium': 2, 'low': 1}

    # 2. Iterate and sort tasks into member queues
    for epic in result.get('epics', []):
        for us in epic.get('userStories', []):
            for task in us.get('tasks', []):
                assigned_name = task.get('assignedTo')
                
                # Use LLM-assigned name, but default to 'Unassigned' if missing or not in the team
                mem = member_map.get(assigned_name)
                
                if not mem:
                    # Fallback assignment to 'Unassigned' member if the name from LLM doesn't exist
                    unassigned_member = member_map.setdefault(
                        "Unassigned", 
                        {'name': 'Unassigned', 'role': 'Developer', 'assigned': 0, 'tasks': []}
                    )
                    mem = unassigned_member
                    task['assignedTo'] = 'Unassigned'
                else:
                    task['assignedTo'] = mem['name']

                # Ensure role field is populated correctly
                task['role'] = mem['role']
                
                # Track the task
                mem['tasks'].append(task)
                mem['assigned'] += 1
    
    # 3. Enforce 'In Progress' limit and re-calculate queueOrder
    for name, mem in member_map.items():
        if not mem['tasks']:
            continue
            
        # Sort all tasks to determine the true sequential order:
        # Sort key: 1. Priority (High > Medium > Low) 2. LLM's initial queueOrder (as tie-breaker)
        mem['tasks'].sort(key=lambda x: (
            priority_order.get(x.get('priority', '').lower(), 0), 
            x.get('queueOrder', 999) 
        ), reverse=True) # Highest priority first

        # Apply final status and sequential queueOrder
        for i, task in enumerate(mem['tasks']):
            task['queueOrder'] = i + 1
            if i == 0:
                # The highest priority task is set to 'In Progress'
                task['status'] = 'In Progress' 
            else:
                # All other tasks must be 'Backlog' to enforce the single In-Progress rule
                task['status'] = 'Backlog'

    return result


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

    # Add 'Unassigned' if team is empty, or ensure it exists for fallback
    if not team or not any(m.get('name') == 'Unassigned' for m in team):
        team.append({"name": "Unassigned", "role": "Developer"})

    # --- Step 2: Generate tasks with Gemini (SMART PROMPT) ---
    team_json = json.dumps(team, indent=2)
    prompt = f"""
You are an **AI Agile Project Planner** helping plan the project described below.
The project is: {input.description} [cite: 374]

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

### Important Instructions for Role-Based Assignment and Balancing:
- **Critical Role-Based Assignment**: You must assign the `assignedTo` field to a specific `Member Name` from the `Team` list.
- **Match Tasks to Roles**: Tasks should be assigned based on the team member's role (e.g., Frontend tasks to Frontend Developers, Database tasks to Backend/DB Engineers).
- **Workload Balancing**: Distribute the total number of tasks as evenly as possible among all relevant members, avoiding skill mismatch.
- **Priority and Status**: Set initial `priority` and `status`. The post-processing agent will enforce the final constraint: Each member can have only ONE task with `"status": "In Progress"` (the highest priority task for that member).
- Assign a realistic, sequential `"queueOrder"` per member.
- Use realistic task names and durations (2–7 days).
- Return **ONLY valid JSON** — no markdown, no commentary, no triple backticks.
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

        # Get text and parse
        try:
            if response.candidates[0].content and getattr(response.candidates[0].content, 'parts', None):
                raw = "".join(response.candidates[0].content.parts)
            else:
                raw = response.text if hasattr(response, 'text') else ''
        except Exception:
            raw = response.text if hasattr(response, 'text') else ''

        result = raw.strip()
        print(f"🧠 Gemini response (first 300 chars):\n{result[:300]}")

        # Remove triple-backtick fences if present
        def strip_code_fences(s):
            s2 = re.sub(r'^```[a-zA-Z]*\n', '', s)
            s2 = re.sub(r'\n```$', '', s2)
            return s2

        cleaned = strip_code_fences(result)

        # Helper: extract balanced JSON by scanning braces (handles nested braces and ignores braces inside strings)
        # This function is retained for robust JSON parsing
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

        # Helper: escape raw newlines inside JSON string literals
        # This function is retained for robust JSON parsing
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

        json_text = cleaned

        # Try direct parse, then balanced extraction + fixes
        try:
            parsed_result = json.loads(json_text)
            print("✅ Successfully parsed Gemini JSON (Direct)")
            return assign_tasks_by_role(parsed_result, team)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed (Direct): {e}")

        extracted = extract_balanced_json(cleaned)
        if extracted:
            # Escape newlines inside strings and remove trailing commas
            fixed = escape_newlines_in_strings(extracted)
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            try:
                parsed_result = json.loads(fixed)
                print("✅ Successfully extracted and parsed JSON (Fixed)")
                return assign_tasks_by_role(parsed_result, team)
            except json.JSONDecodeError as e2:
                print(f"⚠️ Failed to parse extracted JSON: {e2}")

            # As a last attempt, try to find the largest {...} substring and parse it
            candidates = re.findall(r'\{.*?\}', cleaned, re.DOTALL)
            if candidates:
                candidates.sort(key=len, reverse=True)
                for cand in candidates:
                    try:
                        cand_fixed = escape_newlines_in_strings(cand)
                        cand_fixed = re.sub(r',\s*([}\]])', r'\1', cand_fixed)
                        parsed_result = json.loads(cand_fixed)
                        print("✅ Successfully parsed JSON from a candidate substring")
                        return assign_tasks_by_role(parsed_result, team)
                    except Exception:
                        continue

            # Save the raw cleaned response to a debug file for inspection
            try:
                import pathlib, datetime
                debug_dir = pathlib.Path('debug_responses')
                debug_dir.mkdir(exist_ok=True)
                ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                debug_path = debug_dir / f'gemini_response_{ts}.txt'
                with debug_path.open('w', encoding='utf-8') as f:
                    f.write(cleaned)
                print(f"🔍 Saved raw Gemini response to: {debug_path.resolve()}")
            except Exception as dbg_e:
                print(f"Failed to write debug file: {dbg_e}")

            print("🔄 Falling back due to JSON parsing failure")
            return generate_fallback_tasks(input.description, team)

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return generate_fallback_tasks(input.description, team)


if __name__ == "__main__":
    import uvicorn
    # Make sure you run this file with 'uvicorn <filename>:app --reload'
    uvicorn.run(app, host="0.0.0.0", port=8000)