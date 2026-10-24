import re
from .models import Task


def tier_for(text):
    if re.search(r'architect|deep|algorithm|refactor|distributed|trade.?off|complexity|prove|reasoning', text, re.I):
        return 'heavy'
    if re.search(r'\b(code|script|function|unit tests?|implement|debug)\b', text, re.I):
        return 'balanced'
    return 'eco'


def rule_plan(prompt):
    # Explicit separators avoid splitting data values or every sentence.
    parts = re.split(r'\n\s*---\s*\n|\n(?=\s*\d+[.)]\s)|\s+then\s+', prompt, flags=re.I)
    parts = [re.sub(r'^\s*\d+[.)]\s*', '', p).strip() for p in parts if p.strip()]
    if len(parts) > 8:
        parts = [prompt]
    return [Task(task_id=i, prompt_segment=p, target_tier=tier_for(p),
                 dependencies=[i - 1] if i > 1 else []) for i, p in enumerate(parts, 1)]


def validate_plan(raw):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
        raise ValueError('Planner must return 1–8 tasks.')
    tasks, seen = [], set()
    for item in raw:
        task = Task.model_validate(item)
        if task.task_id in seen or any(d not in seen for d in task.dependencies):
            raise ValueError('Task IDs must be unique and dependencies must precede consumers.')
        seen.add(task.task_id)
        tasks.append(task)
    return tasks
