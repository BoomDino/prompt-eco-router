import json
import unittest
import httpx
from fastapi.testclient import TestClient
from main import create_app
from eco_router.config import Settings
from eco_router.planner import validate_plan
from eco_router.cache import SemanticCache
from eco_router.validation import validate_output


def payload(prompt='Format as bullets: Ada, Grace', **kwargs):
    return {'model': 'eco-router', 'messages': [{'role': 'user', 'content': prompt}], **kwargs}


class RouterTests(unittest.TestCase):
    def test_presets_and_cache(self):
        with TestClient(create_app(Settings())) as client:
            first = client.post('/v1/chat/completions', json=payload())
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json()['eco_telemetry']['tasks'][0]['actual_tier'], 'eco')
            second = client.post('/v1/chat/completions', json=payload()).json()
            self.assertTrue(second['eco_telemetry']['cache_hit'])
            self.assertEqual(second['usage']['total_tokens'], 0)
            self.assertEqual(second['eco_telemetry']['attempts'], [])
            self.assertNotEqual(first.json()['id'], second['id'])
            mixed = client.post('/v1/chat/completions', json=payload('Extract errors as JSON: ERROR timeout\n---\nAnalyze the architecture')).json()
            self.assertEqual([t['actual_tier'] for t in mixed['eco_telemetry']['tasks']], ['eco', 'heavy'])
            self.assertEqual(mixed['eco_telemetry']['tasks'][1]['dependencies'], [1])
            heavy = client.post('/v1/chat/completions', json=payload('Deep algorithm refactoring')).json()
            self.assertEqual(heavy['eco_telemetry']['tasks'][0]['actual_tier'], 'heavy')
            self.assertIn('x-eco-wh-saved', first.headers)

    def test_cache_isolation(self):
        with TestClient(create_app(Settings())) as client:
            for body in [payload('A before B'), payload('B before A'), payload('A before B', seed=9),
                         payload('A before B', temperature=1), payload('A before B', user='other')]:
                response = client.post('/v1/chat/completions', json=body)
                self.assertFalse(response.json()['eco_telemetry']['cache_hit'])
            body = payload('A before B')
            body['messages'].insert(0, {'role': 'system', 'content': 'Be concise'})
            self.assertFalse(client.post('/v1/chat/completions', json=body).json()['eco_telemetry']['cache_hit'])

    def test_escalation_and_accounting(self):
        with TestClient(create_app(Settings(demo_fail_eco=True))) as client:
            response = client.post('/v1/chat/completions', json=payload()).json()
            t = response['eco_telemetry']
            self.assertEqual(t['tasks'][0]['actual_tier'], 'balanced')
            self.assertEqual(len(t['attempts']), 2)
            self.assertIsNotNone(t['attempts'][0]['validation_error'])
            self.assertEqual(t['processed_tokens'], sum(a['prompt_tokens'] + a['completion_tokens'] for a in t['attempts']))

    def test_auth_and_request_errors(self):
        with TestClient(create_app(Settings(router_key='secret'))) as client:
            self.assertEqual(client.get('/v1/models').status_code, 401)
            self.assertEqual(client.get('/health').status_code, 200)
            headers = {'Authorization': 'Bearer secret'}
            self.assertEqual(client.get('/v1/models', headers=headers).status_code, 200)
            self.assertEqual(client.post('/v1/chat/completions', headers=headers, json=payload(model='wrong')).status_code, 422)
            self.assertEqual(client.post('/v1/chat/completions', headers=headers, json=payload(response_format={'type': 'json_schema'})).status_code, 422)

    def test_structured_and_failure_not_cached(self):
        with TestClient(create_app(Settings())) as client:
            good = client.post('/v1/chat/completions', json=payload(response_format={'type': 'json_object'}))
            self.assertIsInstance(json.loads(good.json()['choices'][0]['message']['content']), dict)
            bad = payload(response_format={'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'array'}}})
            self.assertEqual(client.post('/v1/chat/completions', json=bad).status_code, 502)
            self.assertEqual(client.post('/v1/chat/completions', json=bad).status_code, 502)

    def test_upstream_payload_retry_and_usage(self):
        calls = []
        def upstream(request):
            body = json.loads(request.content)
            calls.append(body)
            content = 'broken' if len(calls) == 1 else '{"ok":true}'
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
        with TestClient(create_app(Settings(api_key='test'), httpx.MockTransport(upstream))) as client:
            data = client.post('/v1/chat/completions', json=payload('Return JSON')).json()
            self.assertEqual(data['usage']['total_tokens'], 30)
            self.assertIn('failed validation', calls[1]['messages'][-1]['content'])
            self.assertEqual(calls[0]['model'], 'llama-3.2-3b-instruct')
            self.assertFalse(data['eco_telemetry']['mock_mode'])

    def test_agent_tool_passthrough_and_sse(self):
        captured = []
        tool_call = {'id': 'call_1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}
        def upstream(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [tool_call]}, 'finish_reason': 'tool_calls'}]})
        body = payload(tools=[{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object'}}}], stream=True, stream_options={'include_usage': True})
        with TestClient(create_app(Settings(api_key='test'), httpx.MockTransport(upstream))) as client:
            response = client.post('/v1/chat/completions', json=body)
            self.assertEqual(response.status_code, 200)
            self.assertIn('[DONE]', response.text)
            self.assertIn('"tool_calls"', response.text)
            self.assertIn('"index": 0', response.text)
            self.assertEqual(captured[0]['messages'], body['messages'])
            self.assertEqual(captured[0]['tools'], body['tools'])
            self.assertEqual(captured[0]['model'], 'llama3.3-70b-instruct')
            client.post('/v1/chat/completions', json=body)
            self.assertEqual(len(captured), 2)

    def test_upstream_errors_are_not_mocked(self):
        for status, expected in [(401, 502), (429, 429), (500, 502)]:
            with TestClient(create_app(Settings(api_key='test'), httpx.MockTransport(lambda req: httpx.Response(status)))) as client:
                response = client.post('/v1/chat/completions', json=payload())
                self.assertEqual(response.status_code, expected)
                self.assertIn('error', response.json())

    def test_planner_fallback_is_accounted(self):
        def upstream(request):
            body = json.loads(request.content)
            text = 'bad plan' if body['model'] == 'hermes-3-llama-3.1-8b' else 'good output'
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': text}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
        with TestClient(create_app(Settings(api_key='test', planner='model'), httpx.MockTransport(upstream))) as client:
            data = client.post('/v1/chat/completions', json=payload()).json()
            self.assertEqual(data['eco_telemetry']['planner'], 'rules_fallback')
            self.assertEqual(data['usage']['total_tokens'], 30)
        with self.assertRaises(ValueError):
            validate_plan([{'task_id': 1, 'prompt_segment': 'x', 'target_tier': 'eco', 'dependencies': [2]}])

    def test_cache_eviction_and_expiry(self):
        cache = SemanticCache(size=1, ttl=900)
        cache.put('a', {'answer': 1})
        cache.put('b', {'answer': 2})
        self.assertIsNone(cache.get('a'))
        returned = cache.get('b')
        returned['answer'] = 99
        self.assertEqual(cache.get('b')['answer'], 2)
        expired = SemanticCache(size=1, ttl=0)
        expired.put('a', {'answer': 1})
        self.assertIsNone(expired.get('a'))

    def test_python_validation(self):
        self.assertIsNone(validate_output('```python\ndef f():\n    return 1\n```', 'Write Python'))
        self.assertIsNotNone(validate_output('```python\ndef f(:\n```', 'Write Python'))

    def test_timeout_and_negative_savings(self):
        def timeout(request):
            raise httpx.ReadTimeout('timeout', request=request)
        with TestClient(create_app(Settings(api_key='test'), httpx.MockTransport(timeout))) as client:
            self.assertEqual(client.post('/v1/chat/completions', json=payload()).status_code, 504)
        settings = Settings(wh_per_1k={'eco': 50, 'balanced': .15, 'heavy': .5})
        with TestClient(create_app(settings)) as client:
            data = client.post('/v1/chat/completions', json=payload()).json()
            self.assertLess(data['eco_telemetry']['wh_saved'], 0)

    def test_model_planner_dependencies(self):
        calls = []
        def upstream(request):
            body = json.loads(request.content)
            calls.append(body)
            if len(calls) == 1:
                content = json.dumps([
                    {'task_id': 1, 'prompt_segment': 'Extract errors', 'target_tier': 'eco', 'dependencies': []},
                    {'task_id': 2, 'prompt_segment': 'Analyze architecture', 'target_tier': 'heavy', 'dependencies': [1]},
                ])
            else:
                content = 'error-result' if len(calls) == 2 else 'architecture-result'
            return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': content}}]})
        with TestClient(create_app(Settings(api_key='test', planner='model'), httpx.MockTransport(upstream))) as client:
            response = client.post('/v1/chat/completions', json=payload('Extract errors then analyze architecture'))
            self.assertEqual(response.status_code, 200)
            self.assertIn('error-result', calls[2]['messages'][-1]['content'])
            self.assertEqual(response.json()['eco_telemetry']['planner'], 'model')


if __name__ == '__main__':
    unittest.main()
