import os
import json
import random
import re
import traceback
import asyncio
from typing import List, Dict, Any, Optional, Set
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime

# Try to import google.genai, fallback to warnings and local-only operation
try:
    import google.generativeai as genai
    from google.generativeai import GenerationConfig
    GENAI_OK = True
except Exception:
    GENAI_OK = False
    print("⚠️ google.generativeai not available; agent will use fallback/heuristics.")

load_dotenv()

# --- Configuration / Tuning ---
NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEFAULT_TASK_COUNT = int(os.getenv("DEFAULT_TASK_COUNT", "20"))
DEFAULT_MAX_OUTPUT_TASKS = 60
RANDOM_SEED = int(os.getenv("TASK_GEN_SEED", "42"))

# If genai available, configure
if GENAI_OK and GOOGLE_API_KEY:
    pass

app = FastAPI(title="NEXA Task Generator Agent (Upgraded)")

# -------------------------
# Pydantic Input / Output
# -------------------------
class TeamMemberInput(BaseModel):
    # Use 'id' as the internal field name. Use alias for external input fields.
    id: Optional[str] = Field(None, alias="projectMemberId") 
    
    name: str
    role: Optional[str] = "Developer"
    reliability: Optional[float] = 0.8  # 0.0 - 1.0
    hourlyCapacity: Optional[float] = 40.0
    availabilityPct: Optional[float] = 1.0

    @property
    def _id(self) -> Optional[str]:
        return self.id
    
    class Config:
        populate_by_name = True
        # Pydantic V2 renamed 'allow_population_by_field_name' -> 'populate_by_name'.
        # Keep only the modern key to avoid V2 warnings.

# Point 0: Canonical Project Configuration Input
class ProjectConfigInput(BaseModel):
    projectName: str
    projectType: str = "Software Project"
    mainModules: List[str]
    techStack: List[str]
    teamRoles: List[str]
    complexityLevel: str = "Medium"
    preferredTaskCount: Optional[int] = DEFAULT_TASK_COUNT
    includeBugTasks: Optional[bool] = True
    includeTestCases: Optional[bool] = True

class TaskGenOptions(BaseModel):
    effortEstimateMethod: Optional[str] = "heuristic"
    deterministicSeed: Optional[int] = RANDOM_SEED

class ProjectInput(BaseModel):
    project_id: str
    description: str
    auth_token: Optional[str] = None
    team: Optional[List[TeamMemberInput]] = None
    options: Optional[TaskGenOptions] = TaskGenOptions()
    config: ProjectConfigInput 


# -------------------------
# Helpers & Heuristics
# -------------------------

# Point 6: Timestamp Helper for Audit Logs
def timestamp_log(message: str) -> str:
    """Prepends a timestamp to a log message."""
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

# FIX: Simple JSON load for structured output is now sufficient
def simple_json_load(s: str) -> Optional[Dict[str, Any]]:
    """Tries to load JSON from a clean string output."""
    try:
        # Check for markdown fences that Structured Output might sometimes generate
        s = s.strip()
        if s.startswith("```json"):
            s = s.lstrip("```json").rstrip("```")
        return json.loads(s)
    except Exception:
        return None

# FIX: Removed complex and brittle heuristic parsing functions

def find_nested_key(obj: Any, key: str) -> Optional[Any]:
    """Recursively search for a key in nested dict/list structures and return the first match."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            res = find_nested_key(v, key)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_nested_key(item, key)
            if res is not None:
                return res
    return None

def deterministic_random(seed: int):
    rnd = random.Random(seed)
    return rnd

# Updated role keywords for better type/role inference in heuristics
ROLE_MAPPING = {
    "backend": ["backend developer", "full-stack", "api", "server", "database"],
    "frontend": ["frontend developer", "ui", "react", "ux", "design"],
    "ai engineer": ["ai engineer", "ml", "fastapi", "agent"],
    "devops": ["devops", "infrastructure", "ci/cd", "deployment", "docker", "devops engineer"],
    "qa engineer": ["qa engineer", "test", "validate", "testing", "e2e", "automation and testing engineer"],
    "documentation": ["documentation", "writer", "technical writer"],
    "developer": ["developer"]
}

def heuristic_estimate_hours(priority: str, complexity: str) -> float:
    """Simple heuristic to estimate hours given priority+complexity."""
    base = 4.0
    pr_map = {"low": 0.75, "medium": 1.0, "high": 1.5, "critical": 2.5, "p1": 2.5, "p2": 1.5, "p3": 1.0}
    cx_map = {"low": 0.5, "medium": 1.0, "high": 1.5}
    p = pr_map.get(priority.lower(), 1.0)
    c = cx_map.get(complexity.lower(), 1.0)
    hours = base * p * c
    return round(max(2.0, hours), 1)


def _get_role_category(role_name: str) -> str:
    """Standardizes a role name to its category (e.g., 'Backend Developer' -> 'backend')."""
    role_name = role_name.lower().strip()
    return next((k for k, v in ROLE_MAPPING.items() if role_name in v or k in role_name), "developer")

def _normalize_task_role(task_role: str, team_names: Set[str]) -> str:
    """
    FIX: Checks if the task's requested role is actually a member's name and corrects it
    to a generalized role type.
    """
    task_role = task_role.strip()
    
    if task_role in team_names:
        # Fallback to the member's known role, as provided in the input team list
        # Since we don't have direct member-to-role map here, we infer from name-based keywords
        if "document" in task_role.lower() or "documentation" in task_role.lower() or "writer" in task_role.lower() or "ayesha" in task_role.lower():
            return "Documentation"
        elif "qa" in task_role.lower() or "test" in task_role.lower() or "automation" in task_role.lower():
             return "QA Engineer"
        elif "devops" in task_role.lower() or "taha" in task_role.lower():
             return "Devops Engineer"
        # Since M.Tayyab is Backend, tasks assigned by name to him are assumed backend
        elif "tayyab" in task_role.lower():
             return "Backend Developer"
        return "Developer" # Fallback to generic Developer role
        
    return task_role


def pick_best_member_for_task(task_role: str, team: List[TeamMemberInput], team_names: Set[str], rnd: random.Random) -> Optional[TeamMemberInput]:
    """
    Prioritizes members by role fit (with extreme dominance) then ranks by reliability.
    """
    
    # FIX 1: Normalize the task role first
    task_role_norm = _normalize_task_role(task_role, team_names).lower().strip()
    
    required_category = _get_role_category(task_role_norm)
    
    candidates = []

    for m in team:
        member_role_norm = (m.role or "developer").lower().strip()
        
        # 1. Role Match Score (The most critical factor)
        member_category = _get_role_category(member_role_norm)

        if required_category == member_category:
            role_fit_score = 1.0 # Perfect Category Match
        elif "developer" in required_category and member_category == "developer":
            role_fit_score = 0.8 # General Developer Match
        else:
            # FIX 2: Use extreme penalty for mismatch. 
            role_fit_score = 0.001 

        # 2. Weighted Score: Reliability * Availability 
        skill_weight = (m.reliability or 0.0) * (m.availabilityPct or 1.0)
        
        # Combine: Role Fit has DOMINANT weight (100x the maximum reliability weight)
        score = (role_fit_score * 100) + skill_weight + (rnd.random() * 0.01) 
        candidates.append((score, m))
        
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Only return a candidate if they have a non-trivial role fit
    if candidates and candidates[0][0] > 1.0: 
        return candidates[0][1]
    
    return candidates[0][1] if candidates else None

def attach_dependencies(epic_tasks: List[Dict[str,Any]], rnd: random.Random, dependency_rate=0.20):
    """Randomly create dependencies within a list of tasks."""
    for i in range(len(epic_tasks)):
        epic_tasks[i]['dependencies'] = epic_tasks[i].get('dependencies', [])
        if i == 0: 
            continue
            
        for j in range(0, i):
            if rnd.random() < dependency_rate:
                if epic_tasks[j]['_id'] not in epic_tasks[i]['dependencies']:
                    epic_tasks[i]['dependencies'].append(epic_tasks[j]['_id'])
    return epic_tasks

def add_qa_tasks_to_story(tasks_for_story: List[Dict[str,Any]], story_title: str, rnd: random.Random) -> List[Dict[str,Any]]:
    """For each story, add a small QA/test task."""
    qa_id = f"qa-{rnd.randint(100000,999999)}"
    feature_hours = sum(t.get("estimatedHours") or 8.0 for t in tasks_for_story if t.get("type", "feature").lower() not in ["qa", "test"])
    qa_hours = round(max(2.0, feature_hours * 0.25), 1)

    qa_task = {
        "_id": qa_id,
        "title": f"QA: Test and validate '{story_title}' features",
        "priority": "High",
        "status": "Backlog",
        "assignedTo": None,
        "dueDate": None,
        "estimatedHours": qa_hours,
        "queueOrder": len(tasks_for_story) + 1,
        "businessValue": 1,
        "dependencies": [t["_id"] for t in tasks_for_story if t["_id"] != qa_id] if tasks_for_story else [],
        "type": "QA",
        "role": "QA Engineer",
        "riskLevel": "Low", 
        "definitionOfDone": "All test cases pass; no critical bugs found.",
        "tags": ["testing", "quality"],
        "potentialBlockers": []
    }
    tasks_for_story.append(qa_task)
    return tasks_for_story

# -------------------------
# LLM prompt builders
# -------------------------
def build_prompt(config: ProjectConfigInput, team: List[TeamMemberInput]) -> str:
    """Build a compact prompt for Gemini based on the canonical ProjectConfigInput."""
    team_brief = ", ".join([f"{t.name}({t.role}, R={round(t.reliability or 0, 2)})" for t in team])
    
    config_details = f"""
    Project: {config.projectName} ({config.projectType})
    Main Modules: {', '.join(config.mainModules)}
    Tech Stack: {', '.join(config.techStack)}
    Complexity: {config.complexityLevel}
    Target Tasks: {config.preferredTaskCount}
    Include QA/Tests: {config.includeTestCases}
    """
    
    prompt = f"""
You are an expert Agile planner. Produce a comprehensive product backlog structured by Phases -> Epics -> UserStories -> Tasks.

--- PROJECT DETAILS ---
{config_details}
Team members available: {team_brief}
---

Produce approximately {config.preferredTaskCount} actionable technical tasks.
The `role` field for each task MUST be a generalized role (e.g., 'Backend Developer', 'Frontend Developer', 'DevOps', 'QA Engineer', 'Documentation') and NOT a person's name.
"""
    return prompt

# -------------------------
# Structured Output Schema (Mandatory for reliable JSON generation)
# -------------------------

TASK_SCHEMA_PROPERTIES = {
    "title": {"type": "STRING"}, 
    "priority": {"type": "STRING", "enum": ["Low", "Medium", "High", "Critical", "P1", "P2", "P3"]}, 
    "role": {"type": "STRING"}, # Will be normalized in post-processing
    "complexity": {"type": "STRING", "enum": ["Low", "Medium", "High"]}, 
    "type": {"type": "STRING", "enum": ["Feature", "Bug", "DevOps", "Security", "Documentation", "Risk", "QA"]},
    "riskLevel": {"type": "STRING", "enum": ["Low", "Medium", "High"]},
    "definitionOfDone": {"type": "STRING", "description": "Brief sentence defining completion criteria."},
    "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
    "potentialBlockers": {"type": "ARRAY", "items": {"type": "STRING"}},
}
TASK_SCHEMA_REQUIRED = ["title", "priority", "role", "complexity", "type", "riskLevel", "definitionOfDone", "tags", "potentialBlockers"]


BACKLOG_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "phases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "epics": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "title": {"type": "STRING"},
                                "userStories": {
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "story": {"type": "STRING"},
                                            "acceptanceCriteria": {"type": "ARRAY", "items": {"type": "STRING"}},
                                            "tasks": {
                                                "type": "ARRAY",
                                                "items": {
                                                    "type": "OBJECT",
                                                    "properties": TASK_SCHEMA_PROPERTIES,
                                                    "required": TASK_SCHEMA_REQUIRED
                                                }
                                            }
                                        },
                                        "required": ["story", "acceptanceCriteria", "tasks"]
                                    }
                                }
                            },
                            "required": ["title", "userStories"]
                        }
                    }
                },
                "required": ["title", "epics"]
            }
        }
    },
    "required": ["phases"]
}

# -------------------------
# Main generation function
# -------------------------
async def generate_with_gemini(prompt: str) -> Optional[Dict[str,Any]]:
    """Call gemini safely (if available) and return parsed JSON or None."""
    if not GENAI_OK:
        print(timestamp_log("⚠️ google.generativeai library not available; Gemini disabled."))
        return None
    if not GOOGLE_API_KEY:
        print(timestamp_log("⚠️ GOOGLE_API_KEY is not set; Gemini calls will be skipped."))
        return None
        
    try:
        import requests 
        
        config_dict = {
            "temperature": 0.6, 
            "responseMimeType": "application/json", 
            "maxOutputTokens": 8000,
            "responseSchema": BACKLOG_RESPONSE_SCHEMA
        }
        
        model_name = "gemini-2.5-flash-preview-09-2025"
        # Correct API URL construction
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GOOGLE_API_KEY}"
        
        payload = {
            "contents": [{ "parts": [{ "text": prompt }] }],
            "generationConfig": config_dict
        }
        
        headers = {'Content-Type': 'application/json'}
        
        response = await asyncio.to_thread(
            lambda: requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=90)
        )
        response.raise_for_status()
        
        result = response.json()
        raw_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
        
        # FIX: Simple parsing now that structured output is enforced
        parsed = simple_json_load(raw_text) 
        
        return parsed

    except Exception as e:
        print(timestamp_log(f"❌ Gemini call failed: {str(e)}"))
        print(traceback.format_exc())
        return None

# -------------------------
# Fallback content generator (heuristic)
# -------------------------
def fallback_generate(config: ProjectConfigInput, team: List[TeamMemberInput], rnd: random.Random) -> Dict[str,Any]:
    """Heuristic generator that returns structured phases->epics->stories->tasks based on ProjectConfigInput."""
    
    result_phases = []
    task_id_counter = rnd.randint(100000, 999999)
    
    # Heuristic Data for new fields
    risk_levels = ["Low", "Medium", "High"]
    blockers = ["Requires review from lead", "Awaiting design finalization", "External API dependency", "No known blocker"]
    tags_list = [["auth", "security"], ["database", "backend"], ["ui", "frontend"], ["devops", "ci"], ["testing", "quality"]]
    
    phases_map = {
        "Setup & Foundations": ["Project Initialization", "Authentication"],
        "Core Feature Development": config.mainModules,
        "Infrastructure & Quality": ["DevOps Pipeline", "Testing Frameworks"],
    }
    
    for phase_title, modules in phases_map.items():
        epics = []
        module_list = list(modules)
        
        for epic_title in module_list:
            userStories = []
            stories_count = rnd.randint(1, 2)
            
            for s_i in range(stories_count):
                story = f"As a user, I need {epic_title.lower()} access."
                tasks = []
                tasks_per_story = rnd.randint(2, 4)
                
                default_role = "Developer"
                if "backend" in epic_title.lower() or "api" in epic_title.lower() or "authentication" in epic_title.lower(): default_role = "Backend Developer"
                elif "frontend" in epic_title.lower() or "ui" in epic_title.lower(): default_role = "Frontend Developer"
                elif "devops" in epic_title.lower() or "pipeline" in epic_title.lower(): default_role = "DevOps"

                for t_i in range(tasks_per_story):
                    task_id_counter += 1
                    priority = rnd.choice(["High","Medium","Low"])
                    complexity = rnd.choice(["Low","Medium","High"])
                    risk = rnd.choice(risk_levels)
                    
                    tasks.append({
                        "_id": str(task_id_counter),
                        "title": f"{epic_title}: Implement task {t_i+1} for {story[:30]}",
                        "priority": priority,
                        "status": "Backlog",
                        "assignedTo": None,
                        "dueDate": None,
                        "estimatedHours": heuristic_estimate_hours(priority, complexity),
                        "queueOrder": t_i+1,
                        "businessValue": 1,
                        "dependencies": [],
                        "type": "Feature",
                        "role": default_role,
                        "riskLevel": risk,
                        "definitionOfDone": f"Code merged, reviewed, and passes all {priority} tests.",
                        "tags": rnd.choice(tags_list),
                        "potentialBlockers": [rnd.choice(blockers)] if risk != "Low" else []
                    })
                    
                if config.includeTestCases:
                    tasks = add_qa_tasks_to_story(tasks, story, rnd)
                    
                userStories.append({"story": story, "acceptanceCriteria": ["Implemented", "Reviewed"], "tasks": tasks})
            epics.append({"title": epic_title, "userStories": userStories})
        if epics:
            result_phases.append({"title": phase_title, "epics": epics})

    return {"phases": result_phases}


# -------------------------
# Post-process & assign (Suggestion Only)
# -------------------------
def flatten_and_assign(phases: Dict[str,Any], team: List[TeamMemberInput], options: TaskGenOptions, rnd: random.Random, config: Optional[ProjectConfigInput] = None):
    """
    Walk through phases->epics->stories->tasks:
      - Estimate effort (Heuristic used as final source of truth)
      - Attach dependencies within same story
      - Suggest tasks by role + reliability (NO CAPACITY CHECKING)
      - Build meta info for traceability
    """
    decision_log = []
    assignment_reasons = []
    rolled_over_tasks = []
    member_capacity = {}
    reliability_history = {}
    workload_difficulty = {}
    all_tasks_count = 0
    total_assigned_tasks = 0
    total_reliability = sum(m.reliability or 0.0 for m in team)
    avg_reliability = total_reliability / len(team) if team else 0.0
    
    team_names = {m.name for m in team}

    for m in team:
        member_id_key = m._id or m.name 
        member_capacity[member_id_key] = {
            "total": m.hourlyCapacity or 40.0,
            "assignedTasks": 0,
            "maxTasks": max(1, int((m.hourlyCapacity or 40.0) // 8))
        }
        reliability_history[member_id_key] = {"before": m.reliability or 0.0, "after": m.reliability or 0.0}
        workload_difficulty[m.role or "other"] = 0.0

    # Walk structure, estimate, and assign suggestion
    for phase in phases.get("phases", []):
        for epic in phase.get("epics", []):
            for story in epic.get("userStories", []):
                tasks = story.get("tasks", [])
                
                # Ensure IDs are present and unique
                for t in tasks:
                    if "_id" not in t or not t["_id"]:
                        t["_id"] = f"tg-{random.randint(100000,999999)}"
                
                # Attach dependencies within story
                tasks = attach_dependencies(tasks, rnd, dependency_rate=0.20)
                
                for t in tasks:
                    all_tasks_count += 1
                    
                    # Finalize Estimated Hours and mandatory enterprise fields
                    complexity = t.get("complexity") or t.get("type") or "Medium"
                    priority = t.get("priority") or "Medium"
                    t["estimatedHours"] = heuristic_estimate_hours(priority, complexity)
                    
                    # Ensure mandatory fields are set even if LLM missed them
                    t["riskLevel"] = t.get("riskLevel") or rnd.choice(["Low", "Medium"])
                    t["definitionOfDone"] = t.get("definitionOfDone") or f"Reviewed, merged, and all tests pass."
                    t["tags"] = t.get("tags") or [t.get("role", "general").lower()]
                    t["potentialBlockers"] = t.get("potentialBlockers") or []
                    
                    # FIX 1: Normalize role here before attempting assignment
                    t["role"] = _normalize_task_role(t.get("role", "Developer"), team_names)
                    
                    chosen = pick_best_member_for_task(t["role"], team, team_names, rnd)
                    
                    t["suggestedOwner"] = t["role"]
                    t["assignedTo"] = None
                    
                    if chosen:
                        chosen_id_key = chosen._id or chosen.name
                        t["assignedTo_suggested"] = chosen_id_key
                        
                        member_capacity[chosen_id_key]["assignedTasks"] += 1
                        total_assigned_tasks += 1
                        
                        assignment_reasons.append({
                            "task_id": t["_id"],
                            "task_title": t["title"],
                            "member_id": chosen_id_key,
                            "member_name": chosen.name,
                            "reason": f"Suggested owner based on strong role match ({chosen.role}) and reliability ({round(chosen.reliability or 0.0, 2)})."
                        })
                        decision_log.append(timestamp_log(f"✨ Suggested '{t['title']}' → {chosen.name} ({chosen.role or 'N/A'})."))
                        
                        # Use the member's *actual* role for workload reporting, not the task's suggested role
                        workload_difficulty[chosen.role or "other"] = workload_difficulty.get(chosen.role or "other", 0.0) + t["estimatedHours"]
                    else:
                        t["assignedTo_suggested"] = None
                        decision_log.append(timestamp_log(f"🔴 No suitable team member suggested for '{t['title']}'. Needs manual assignment."))

    # Point 4: Better Sprint Health Score (Backlog Quality)
    completeness_score = min(1.0, all_tasks_count / (config.preferredTaskCount or DEFAULT_TASK_COUNT))
    # Simple Health = Completeness * Assignment Quality * Avg Reliability
    sprint_health_score = round(completeness_score * (total_assigned_tasks / all_tasks_count if all_tasks_count > 0 else 0) * avg_reliability * 10, 2)
    
    # Generate heuristic summary
    summary_msg = f"Generated {all_tasks_count} tasks across {len(phases.get('phases', []))} phases. Backlog completeness: {completeness_score*100:.1f}%. The focus is heavily weighted toward {max(workload_difficulty, key=workload_difficulty.get) if workload_difficulty else 'general development'}."

    return {
        "decisionLog": decision_log,
        "assignmentReasons": assignment_reasons,
        "rolledOverTasks": rolled_over_tasks,
        "memberCapacity": member_capacity,
        "workloadDifficulty": workload_difficulty,
        "reliabilityHistory": reliability_history,
        "sprintHealthScore": sprint_health_score,
        "backlogSummary": summary_msg 
    }

# -------------------------
# API Endpoint
# -------------------------
@app.post("/generate-tasks")
async def generate_tasks(input: ProjectInput):
    """
    Primary endpoint for Task Generation.
    Generates a structured backlog based on ProjectConfigInput and team availability.
    """
    seed = (input.options.deterministicSeed if input.options and input.options.deterministicSeed else RANDOM_SEED)
    rnd = deterministic_random(seed)
    
    # 1. Assemble and normalize team data
    team_data_list = input.team or []
    
    # Ensure at least one 'Unassigned' placeholder
    if not any((isinstance(m, TeamMemberInput) and m.name == "Unassigned") or (isinstance(m, dict) and m.get("name") == "Unassigned") for m in team_data_list) and not any((isinstance(m, TeamMemberInput) and getattr(m, 'id', None) == "unassigned") or (isinstance(m, dict) and (m.get('id') == 'unassigned' or m.get('projectMemberId') == 'unassigned')) for m in team_data_list):
        # Append as dict so model_validate can normalize aliases like projectMemberId/_id
        team_data_list.append({
            "projectMemberId": "unassigned",
            "name": "Unassigned",
            "role": "Developer",
            "reliability": 0.5,
            "hourlyCapacity": 40.0,
            "availabilityPct": 1.0
        })

    # Convert all team items into the Pydantic model for consistent attribute access
    team_members = [m if isinstance(m, TeamMemberInput) else TeamMemberInput.model_validate(m) for m in team_data_list]
            
    if not team_members:
         raise HTTPException(status_code=400, detail="No valid team members could be processed.")


    # 2. Build prompt & call LLM
    config = input.config 
    
    target_task_count = max(1, min(DEFAULT_MAX_OUTPUT_TASKS, config.preferredTaskCount or DEFAULT_TASK_COUNT))
    config.preferredTaskCount = target_task_count
    
    prompt = build_prompt(config, team_members)

    llm_result = await generate_with_gemini(prompt)
    phases_struct = None

    # Accept several output shapes from the LLM:
    if llm_result:
        # Top-level dict with 'phases'
        if isinstance(llm_result, dict) and isinstance(llm_result.get('phases'), list):
            phases_struct = llm_result
        else:
            # Try to find nested 'phases' anywhere in the parsed object
            nested = find_nested_key(llm_result, 'phases')
            if nested and isinstance(nested, list):
                phases_struct = {'phases': nested}
                print(timestamp_log("ℹ️ Found nested 'phases' in Gemini output and will use it."))
            elif isinstance(llm_result, list):
                # If the LLM returned a list, assume it's the phases list
                phases_struct = {'phases': llm_result}
                print(timestamp_log("ℹ️ Gemini returned a top-level list — treating it as phases."))
            else:
                # Parsed successfully but shape is unexpected. Try to coerce known shapes
                coerced = None
                try:
                    # If the model returned a single Task-like dict, wrap it into phases->epics->userStories->tasks
                    if isinstance(llm_result, dict):
                        # heuristic: look for task-like keys
                        task_like_keys = {"title", "priority", "role"}
                        if task_like_keys.intersection(set(llm_result.keys())):
                            task_obj = llm_result.copy()
                            if "_id" not in task_obj:
                                task_obj["_id"] = f"tg-{random.randint(100000,999999)}"
                            coerced = {
                                "phases": [
                                    {
                                        "title": "Auto-generated Phase",
                                        "epics": [
                                            {
                                                "title": "Auto-generated Epic",
                                                "userStories": [
                                                    {
                                                        "story": task_obj.get("story", "Auto-generated story"),
                                                        "acceptanceCriteria": [],
                                                        "tasks": [task_obj]
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                            print(timestamp_log("ℹ️ Gemini returned a single task object — wrapping into phases/epic/story."))
                    # If it is a list of tasks, wrap similarly
                    if coerced is None and isinstance(llm_result, list) and llm_result:
                        # check first element for task-like shape
                        first = llm_result[0]
                        if isinstance(first, dict) and {"title", "priority"}.intersection(set(first.keys())):
                            tasks = []
                            for t in llm_result:
                                tt = t.copy()
                                if "_id" not in tt:
                                    tt["_id"] = f"tg-{random.randint(100000,999999)}"
                                tasks.append(tt)
                            coerced = {
                                "phases": [
                                    {
                                        "title": "Auto-generated Phase",
                                        "epics": [
                                            {
                                                "title": "Auto-generated Epic",
                                                "userStories": [
                                                    {
                                                        "story": "Auto-generated story",
                                                        "acceptanceCriteria": [],
                                                        "tasks": tasks
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                            print(timestamp_log("ℹ️ Gemini returned a list of tasks — wrapping into phases/epic/story."))
                except Exception:
                    coerced = None

                if coerced is not None:
                    phases_struct = coerced
                else:
                    # Final fallback logic remains the same (using heuristic generator)
                    print(timestamp_log("⚠️ Parsed Gemini output lacked 'phases' and failed coercion; using fallback."))
                    phases_struct = fallback_generate(config, team_members, rnd)
    else:
        print(timestamp_log("⚠️ Gemini result missing or invalid – using fallback heuristic generator."))
        phases_struct = fallback_generate(config, team_members, rnd)

    # 3. Post process, flatten, assign SUGGESTIONS
    meta = flatten_and_assign(phases_struct, team_members, input.options, rnd, config)
    
    # 4. Final assembled response
    response = {
        "success": True,
        "projectId": input.project_id,
        "phases": phases_struct.get("phases", []),
        "meta": {
            "backlogSummary": meta["backlogSummary"],
            "sprintHealthScore": meta["sprintHealthScore"],
            "decisionLog": meta["decisionLog"],
            "assignmentReasons": meta["assignmentReasons"],
            "rolledOverTasks": meta["rolledOverTasks"],
            "memberCapacity": meta["memberCapacity"],
            "workloadDifficulty": meta["workloadDifficulty"],
            "reliabilityHistory": meta["reliabilityHistory"],
        }
    }

    return response

# If run standalone
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))