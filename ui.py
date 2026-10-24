"""Run with: streamlit run ui.py."""
import os
import time
import graphviz
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title='Prompt Decomposer Eco-Router', page_icon='🌿', layout='wide')
st.title('🌿 Prompt Decomposer Eco-Router')
st.caption('Small tasks. Right-sized models. Visible savings.')
PRESETS = {
    '1 · Simple formatting': 'Format these names as a bullet list: Ada, Grace, Linus',
    '2 · Logs + architecture': 'Extract ERROR entries from these logs as JSON:\nINFO service ready\nERROR database timeout\nERROR queue full\n---\nAnalyze the system architecture using the extracted errors. Propose backpressure and reliability improvements.',
    '3 · Deep refactoring': 'Deep algorithm refactoring: an event processor scans all previous events for each new event, resulting in O(n²) work. Design an indexed approach, analyze complexity, and reason about duplicate events and memory tradeoffs.',
}
with st.sidebar:
    st.header('Connection')
    api_url = st.text_input('Router URL', os.getenv('ECO_ROUTER_URL', 'http://127.0.0.1:8000')).rstrip('/')
    api_key = st.text_input('Router API key (optional)', os.getenv('ROUTER_API_KEY', ''), type='password')
    use_cache = st.toggle('Use duplicate cache', value=True)
    st.caption('Green = Eco · Amber = Balanced · Blue = Heavy')
    st.info('Energy and dollar values are configurable estimates, not power measurements or invoices.')

if 'prompt' not in st.session_state:
    st.session_state.prompt = next(iter(PRESETS.values()))
for column, (name, value) in zip(st.columns(3), PRESETS.items()):
    if column.button(name, use_container_width=True):
        st.session_state.prompt = value
st.text_area('Your prompt', key='prompt', height=180)
if st.button('Decompose & route', type='primary', use_container_width=True):
    if not st.session_state.prompt.strip():
        st.warning('Enter a prompt first.')
    else:
        payload = {'model': 'eco-router', 'messages': [{'role': 'user', 'content': st.session_state.prompt}],
                   'temperature': 0, 'cache': use_cache}
        start = time.perf_counter()
        with st.status('Planning, dispatching, and validating…', expanded=True) as status:
            try:
                response = httpx.post(api_url + '/v1/chat/completions', json=payload,
                                      headers={'Authorization': 'Bearer ' + api_key} if api_key else {}, timeout=900)
                response.raise_for_status()
                result = response.json()
                st.session_state.run = {'sent': payload, 'received': result,
                                        'elapsed_ms': (time.perf_counter() - start) * 1000,
                                        'headers': {k: v for k, v in response.headers.items() if k.startswith('x-eco-')}}
                status.update(label='Execution complete', state='complete', expanded=False)
            except (httpx.HTTPError, ValueError) as exc:
                st.session_state.pop('run', None)
                status.update(label='Request failed', state='error')
                if isinstance(exc, httpx.HTTPStatusError):
                    st.error(f'API returned {exc.response.status_code}: {exc.response.text[:1000]}')
                else:
                    st.error(f'Cannot complete request: {exc}. Check that the API is running.')

if 'run' in st.session_state:
    run = st.session_state.run
    data = run['received']
    eco = data['eco_telemetry']
    if eco['mock_mode']:
        st.warning('DEMO MODE — deterministic mock outputs; no model was called or billed.')
    if eco['cache_hit']:
        st.success('Cache hit — zero new inference tokens. Graph shows the original execution plan.')
    metrics = st.columns(4)
    metrics[0].metric('Latency', f"{eco['latency_ms']:,.1f} ms")
    metrics[1].metric('Total tokens processed', f"{eco['processed_tokens']:,}")
    metrics[2].metric('Estimated energy saved', f"{eco['wh_saved']:.6f} Wh")
    metrics[3].metric('Estimated cloud dollars saved', f"${eco['cost_saved_usd']:.6f}")
    st.caption(f"Browser round-trip: {run['elapsed_ms']:.1f} ms · Planner: {eco['planner']}")
    if eco['planner_warning']:
        st.warning(eco['planner_warning'])
    st.subheader('Execution graph')
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', bgcolor='transparent')
    graph.node('input', 'Prompt', shape='oval')
    graph.node('output', 'Final response', shape='oval')
    colors = {'eco': '#15803d', 'balanced': '#a16207', 'heavy': '#1d4ed8'}
    for task in eco['tasks']:
        node = str(task['task_id'])
        tier = task['actual_tier']
        label = f"Task {node} · {tier.upper()}"
        if task['escalated']:
            label += f"\nEscalated from {task['target_tier']}"
        graph.node(node, label, style='filled,rounded', shape='box', fillcolor=colors[tier], fontcolor='white')
        for dependency in task['dependencies'] or ['input']:
            graph.edge(str(dependency), node)
        graph.edge(node, 'output')
    st.graphviz_chart(graph, use_container_width=True)
    for task in eco['tasks']:
        with st.expander(f"Task {task['task_id']} · {task['target_tier']} → {task['actual_tier']}"):
            st.write(task['prompt_segment'])
            st.code(task['output'], language=None)
    st.subheader('Final response')
    st.markdown(data['choices'][0]['message'].get('content') or 'Agent tool calls returned.')
    if st.toggle('Show raw JSON sent / received'):
        left, right = st.columns(2)
        with left:
            st.caption('Request')
            st.json(run['sent'])
        with right:
            st.caption('Response and telemetry headers')
            st.json(run['received'])
            st.json(run['headers'])
