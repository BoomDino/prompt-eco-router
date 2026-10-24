import ast
import json
import re
from jsonschema import Draft202012Validator, ValidationError


def unfence(text):
    match = re.fullmatch(r'\s*```[^\n]*\n(.*?)\n```\s*', text, re.S)
    return match.group(1) if match else text.strip()


def validate_output(content, prompt, response_format=None):
    if not isinstance(content, str) or not content.strip():
        return 'Output is empty.'
    fmt = (response_format or {}).get('type', 'text')
    wants_json = fmt in {'json_object', 'json_schema'} or bool(re.search(r'\bjson\b', prompt, re.I))
    try:
        if wants_json:
            def reject_constant(value):
                raise ValueError(f'Nonstandard JSON numeric constant: {value}')
            value = json.loads(content if fmt != 'text' else unfence(content), parse_constant=reject_constant)
            if fmt == 'json_object' and not isinstance(value, dict):
                return 'Expected a JSON object.'
            if fmt == 'json_schema':
                Draft202012Validator(response_format['json_schema']['schema']).validate(value)
        elif re.search(r'\bpython\b', prompt, re.I):
            blocks = re.findall(r'```python\s*\n(.*?)```', content, re.S)
            for block in blocks or [unfence(content)]:
                ast.parse(block)
    except (ValueError, SyntaxError, ValidationError) as exc:
        return str(exc)[:500]
    return None
