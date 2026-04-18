"""
MCP Server for NEXA Task Generator Agent.

Exposes task generation capability as an MCP tool via FastMCP (Streamable HTTP
transport). Mounted on the existing FastAPI app under /mcp — the original
POST /generate-tasks REST endpoint remains completely intact.

Circular-import note: main.py imports `mcp` from this module at the bottom
(after its own top-level definitions are complete). This module only imports
from main inside the tool body (lazy), so there is no circular-import issue.
"""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "nexa-task-generator",
    instructions=(
        "Generates structured project task backlogs organized as "
        "Phases → Epics → User Stories → Tasks using Gemini AI. "
        "Provide project config (name, modules, tech stack, complexity) and "
        "optionally the team roster. Returns a full backlog with effort estimates, "
        "suggested owners, sprint health score and decision log."
    ),
)


@mcp.tool()
async def generate_tasks(
    project_id: str,
    description: str,
    config: Dict[str, Any],
    team: Optional[List[Dict[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a complete AI-powered project backlog.

    project_id: Unique project identifier (MongoDB _id or any string).
    description: High-level project description used to guide the AI.
    config: Required fields —
        projectName (str), projectType (str), mainModules (list[str]),
        techStack (list[str]), teamRoles (list[str]),
        complexityLevel ('Low'|'Medium'|'High'),
        preferredTaskCount (int, default 20),
        includeBugTasks (bool), includeTestCases (bool).
    team: Optional list of member dicts with:
        projectMemberId (str), name (str), role (str),
        reliability (float 0-1), hourlyCapacity (float), availabilityPct (float).
    options: Optional dict with deterministicSeed (int) for reproducible output.

    Returns: success, projectId, phases (Phases→Epics→UserStories→Tasks tree), meta
             (backlogSummary, sprintHealthScore, assignmentReasons, decisionLog, ...).
    """
    # Lazy import to avoid circular dependency with main.py
    from main import (
        ProjectConfigInput,
        TeamMemberInput,
        TaskGenOptions,
        build_prompt,
        deterministic_random,
        fallback_generate,
        find_nested_key,
        flatten_and_assign,
        generate_with_gemini,
        DEFAULT_MAX_OUTPUT_TASKS,
        DEFAULT_TASK_COUNT,
        RANDOM_SEED,
    )

    seed = int((options or {}).get("deterministicSeed", RANDOM_SEED))
    rnd = deterministic_random(seed)

    # Ensure there is at least one placeholder member
    team_data: List[Dict[str, Any]] = list(team or [])
    has_unassigned = any(
        (isinstance(m, dict) and (m.get("name") == "Unassigned" or m.get("projectMemberId") == "unassigned"))
        for m in team_data
    )
    if not has_unassigned:
        team_data.append({
            "projectMemberId": "unassigned",
            "name": "Unassigned",
            "role": "Developer",
            "reliability": 0.5,
            "hourlyCapacity": 40.0,
            "availabilityPct": 1.0,
        })

    team_members = [TeamMemberInput.model_validate(m) for m in team_data]

    # Build config object
    preferred = min(
        DEFAULT_MAX_OUTPUT_TASKS,
        int(config.get("preferredTaskCount", DEFAULT_TASK_COUNT)),
    )
    cfg = ProjectConfigInput(
        projectName=config.get("projectName", "Project"),
        projectType=config.get("projectType", "Software Project"),
        mainModules=config.get("mainModules", []),
        techStack=config.get("techStack", []),
        teamRoles=config.get("teamRoles", []),
        complexityLevel=config.get("complexityLevel", "Medium"),
        preferredTaskCount=preferred,
        includeBugTasks=bool(config.get("includeBugTasks", True)),
        includeTestCases=bool(config.get("includeTestCases", True)),
    )

    # Try Gemini generation, fall back to heuristics
    prompt = build_prompt(cfg, team_members)
    llm_result = await generate_with_gemini(prompt)
    phases_struct: Optional[Dict[str, Any]] = None

    if llm_result:
        if isinstance(llm_result, dict) and isinstance(llm_result.get("phases"), list):
            phases_struct = llm_result
        else:
            nested = find_nested_key(llm_result, "phases")
            if nested and isinstance(nested, list):
                phases_struct = {"phases": nested}

    if not phases_struct:
        phases_struct = fallback_generate(cfg, team_members, rnd)

    # Ensure every phase has a description
    for phase in phases_struct.get("phases", []):
        if not phase.get("description"):
            preview = ", ".join(
                e.get("title", "") for e in (phase.get("epics") or [])[:3]
            )
            phase["description"] = (
                f"Work for {phase.get('title', 'Phase')}. Key areas: {preview}."
                if preview
                else f"Work for {phase.get('title', 'Phase')}."
            )

    # Post-process: assign suggestions, estimate hours, build meta
    opts = TaskGenOptions(deterministicSeed=seed)
    meta = flatten_and_assign(phases_struct, team_members, opts, rnd, cfg)

    return {
        "success": True,
        "projectId": project_id,
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
        },
    }
