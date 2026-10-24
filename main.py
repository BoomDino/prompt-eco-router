"""Run with: uvicorn main:app --reload."""
import hmac
import json
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from eco_router.config import Settings
from eco_router.engine import Router
from eco_router.models import ChatRequest
from eco_router.provider import Provider, UpstreamError


def create_app(settings=None, transport=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=settings.timeout, transport=transport) as client:
            app.state.router = Router(settings, Provider(settings, client))
            yield

    app = FastAPI(title='Prompt Decomposer Eco-Router', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def authenticate(request: Request, call_next):
        if request.url.path.startswith('/v1/') and settings.router_key:
            expected = 'Bearer ' + settings.router_key
            if not hmac.compare_digest(request.headers.get('authorization', '').encode(), expected.encode()):
                return JSONResponse({'error': {'message': 'Invalid router API key.', 'type': 'authentication_error'}}, status_code=401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        errors = [{'location': list(e['loc']), 'message': e['msg']} for e in exc.errors()]
        return JSONResponse({'error': {'message': 'Invalid chat request.', 'type': 'invalid_request_error', 'details': errors}}, status_code=422)

    @app.exception_handler(UpstreamError)
    async def upstream_error(request, exc):
        return JSONResponse({'error': {'message': exc.message, 'type': 'upstream_error'}}, status_code=exc.status)

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'mock_mode': settings.mock, 'planner': settings.planner}

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': 'eco-router', 'object': 'model', 'created': 0, 'owned_by': 'eco-router'}]}

    @app.post('/v1/chat/completions')
    async def completions(body: ChatRequest):
        result = await app.state.router.run(body)
        telemetry = result['eco_telemetry']
        headers = {'X-Eco-Wh-Saved': f"{telemetry['wh_saved']:.9f}",
                   'X-Eco-Cost-Saved': f"{telemetry['cost_saved_usd']:.9f}",
                   'X-Eco-Cache-Hit': str(telemetry['cache_hit']).lower()}
        if not body.stream:
            return JSONResponse(result, headers=headers)

        async def events():
            base = {k: result[k] for k in ('id', 'created', 'model')}
            base['object'] = 'chat.completion.chunk'
            message = result['choices'][0]['message']
            delta = {'role': 'assistant', 'content': message.get('content')}
            if message.get('tool_calls'):
                delta['tool_calls'] = [dict(t, index=i) for i, t in enumerate(message['tool_calls'])]
            yield 'data: ' + json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}) + '\n\n'
            yield 'data: ' + json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': result['choices'][0]['finish_reason']}],
                                        'eco_telemetry': telemetry}) + '\n\n'
            if (body.stream_options or {}).get('include_usage'):
                yield 'data: ' + json.dumps({**base, 'choices': [], 'usage': result['usage']}) + '\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(events(), media_type='text/event-stream', headers=headers)

    return app


app = create_app()
