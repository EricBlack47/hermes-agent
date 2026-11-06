"""api_server admission trust is not machine origin.

The session authority stamps ``event.internal = True`` on every api_server admission because the
turn's authority comes from the private binding and preclaim, never client JSON. That flag used to
also decide the persisted user row's ``display_kind``, so a genuine ``/v1/chat/completions`` client
prompt persisted as ``internal_notification`` and transcript UIs rendered it as a machine notice.

Contract under test: an unmarked api admission (a real client prompt) persists as an ordinary user
row (``display_kind_for_event`` -> ``None``); the wake self-post — marked ``internal_wake`` by
``gateway/wake.py`` through either the HTTP body field or the in-process door — keeps
``internal_notification``. ``event.internal`` stays ``True`` in both cases, and every other
internal producer's event is untouched.
"""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND, display_kind_for_event
from gateway.session import SessionSource


def _drain(owner, ref, row):
    """Run one committed api admission through the real drain; return ``(seen_events, drain_coro)``."""
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result

    seen = []

    async def handle(event):
        seen.append(event)
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'

    owner.runner._handle_message = handle
    return seen, execute_admission(owner, ref, row)


@pytest.mark.asyncio
async def test_api_client_prompt_turn_is_an_ordinary_user_row(api, owner):
    from gateway.session_api_turn import admit_api_turn

    _, ref, row = admit_api_turn(
        api, session_id='client-chat', request_id='req-client',
        user_message='hello there', conversation_history=[])
    # No producer mark: this admission is a genuine client prompt.
    assert 'internal_wake' not in row['payload']['api_turn_v1']

    seen, drain = _drain(owner, ref, row)
    assert await drain == 'ok'
    [event] = seen
    # Admission trust is load-bearing and unchanged — only the display classification is exempted.
    assert event.internal is True
    assert display_kind_for_event(event) is None


@pytest.mark.asyncio
async def test_marked_wake_self_post_turn_stays_internal_notification(api, owner):
    from gateway.session_api import restore_api_session
    from gateway.session_api_turn import admit_api_turn
    from hermes_state_runtime import list_session_admissions

    _, ref, _ = admit_api_turn(
        api, session_id='wake-target', request_id='wake-1',
        user_message='wake: check the build', conversation_history=[], internal_wake=True)

    # The mark is committed in the durable payload: a restarted owner replays the same row and
    # classifies it identically (no live session state is consulted for the decision).
    owner.sessions.clear()
    restore_api_session(owner, ref.session_id)
    [stored] = [r for r in list_session_admissions(
        owner.db, session_id=ref.session_id, pending_only=True) if r['status'] == 'queued']
    assert stored['payload']['api_turn_v1']['internal_wake'] is True

    seen, drain = _drain(owner, ref, stored)
    assert await drain == 'ok'
    [event] = seen
    assert event.internal is True
    assert not getattr(event, '_api_client_prompt', False)
    assert display_kind_for_event(event) == INTERNAL_NOTIFICATION_DISPLAY_KIND


@pytest.mark.asyncio
async def test_push_capable_wake_delivery_stays_internal_notification():
    """``deliver_wake`` on a push-capable adapter (e.g. Telegram) still builds an internal event."""
    from gateway.wake import deliver_wake

    seen = []

    class Adapter:
        supports_async_delivery = True
        gateway_runner = None  # no session authority: the synthetic event goes straight to the handler

        async def handle_message(self, event):
            seen.append(event)
            event._gateway_accepted = True

    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c1', user_id='u1', chat_type='dm')
    await deliver_wake(Adapter(), text='wake text', source=source)
    [event] = seen
    assert event.internal is True
    assert display_kind_for_event(event) == INTERNAL_NOTIFICATION_DISPLAY_KIND


@pytest.mark.asyncio
async def test_committed_hidden_display_still_wins_over_internal():
    """A canonical admission's ``hidden`` kind is checked before the internal branch: an internal
    event carrying committed hidden presentation still persists hidden, never a user bubble."""
    from gateway.session_display import display_turn_scope

    event = SimpleNamespace(message_id='adm-1', internal=True, metadata={})
    with display_turn_scope('adm-1', {'kind': 'hidden'}):
        assert display_kind_for_event(event) == 'hidden'


@pytest.mark.asyncio
async def test_wake_self_post_marks_the_http_turn(api, owner, monkeypatch):
    """The HTTP self-post flags its turn so the chat-completions route can keep it a machinery
    notice; the diagnostic category field is unchanged."""
    import aiohttp
    from gateway.wake import _self_post_chat_completion

    calls = []

    class _Response:
        status = 200

        async def read(self):
            return b''

    class _Post:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            calls.append({'url': url, 'json': json, 'headers': headers})
            return _Post(_Response())

    monkeypatch.setattr(aiohttp, 'ClientSession', _Session)
    adapter = SimpleNamespace(_api_key='k' * 20, _host='127.0.0.1', _port=8642, _model_name='hermes')

    await _self_post_chat_completion(adapter, text='wake', session_id='sess-1')
    [call] = calls
    assert call['json']['hermes_internal_wake'] is True
    assert call['json']['stream'] is False
    assert call['headers']['X-Hermes-Session-Id'] == 'sess-1'
    assert call['headers']['Authorization'] == 'Bearer ' + 'k' * 20

    calls.clear()
    await _self_post_chat_completion(adapter, text='wake', session_id='sess-1',
                                     notification_category='diagnostic')
    [call] = calls
    assert call['json']['hermes_internal_wake'] is True
    assert call['json']['hermes_notification_category'] == 'diagnostic'


@pytest.mark.asyncio
async def test_chat_completions_route_forwards_the_wake_marker(api, owner, monkeypatch):
    """``hermes_internal_wake: true`` in the request body becomes the admission's ``internal_wake``
    kwarg; an ordinary client body carries no such kwarg."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    monkeypatch.setattr(api, '_api_key', 'test-local-key')
    seen = []

    async def fake_run_agent(**kwargs):
        seen.append(kwargs)
        return ({'final_response': 'ok', 'messages': []},
                {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0})

    monkeypatch.setattr(api, '_run_agent', fake_run_agent)
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    headers = {'Authorization': 'Bearer test-local-key', 'X-Hermes-Session-Id': 'sess-route'}
    async with TestClient(TestServer(app)) as client:
        marked = await client.post('/v1/chat/completions', headers=headers, json={
            'messages': [{'role': 'user', 'content': 'wake'}], 'stream': False,
            'hermes_internal_wake': True})
        assert marked.status == 200, await marked.text()
        plain = await client.post('/v1/chat/completions', headers=headers, json={
            'messages': [{'role': 'user', 'content': 'hello'}], 'stream': False})
        assert plain.status == 200, await plain.text()

    [wake_kwargs, client_kwargs] = seen
    assert wake_kwargs['internal_wake'] is True
    assert 'internal_wake' not in client_kwargs
