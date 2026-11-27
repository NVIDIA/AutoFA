# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detector endpoint readiness at startup and durable, bounded mid-run service blockers."""

import json
import logging
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from autofa.config import Config, RunConfig
from autofa.harness.claude import TurnResult
from autofa.llm.hub import HubProbeResult, humanize_hub_error
from autofa.session import detector_service as service
from autofa.session import runner
from autofa.session.llm_readiness import check_llm_readiness, llm_readiness_guidance
from autofa.util.atomic import atomic_write_json

from support.discovery import interactive as interactive
from support.discovery import interactive_runner as interactive_runner


class RecordingConsole(Console):
    def __init__(self):
        self.output = StringIO()
        super().__init__(file=self.output, width=160)


@pytest.fixture
def startup(tmp_path, monkeypatch):
    from autofa import diagnostics

    monkeypatch.setattr(diagnostics, "_DIAGNOSTICS", diagnostics.Diagnostics())
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    monkeypatch.delenv("INFERENCE_HUB_API_KEY", raising=False)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    # A deployment that provides a shared default endpoint, as a site preset would.
    runtime = SimpleNamespace(config=Config(
        feature_model="test-feature-model", hub_base_url="https://hub.example/v1",
        hub_api_key_env="INFERENCE_HUB_API_KEY"))  # pragma: allowlist secret
    console = RecordingConsole()
    logger = logging.getLogger("autofa")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    yield runtime, console
    # Integration sessions attach file sinks; don't leak them into later tests.
    for handler in list(logger.handlers):
        if handler not in original_handlers:
            logger.removeHandler(handler)
            handler.close()
    for handler in original_handlers:
        if handler not in logger.handlers:
            logger.addHandler(handler)
    logger.setLevel(original_level)


def configured(runtime, *, result=None, error=None):
    calls = []

    def probe():
        calls.append("probe")
        if error is not None:
            raise error
        return result or HubProbeResult(ok=True, model="test-feature-model", latency_s=0.1)

    client = SimpleNamespace(probe=probe, available=True)
    runtime.ensure_hub_llm = lambda: client
    return client, calls


@pytest.mark.parametrize("label_api", ["decisions", "responses"])
def test_verified_route_is_given_to_agent(startup, label_api):
    rt, console = startup
    configured(rt, result=HubProbeResult(ok=True, model="test", latency_s=.2,
                                         label_api=label_api, text_api="responses"))
    readiness = check_llm_readiness(rt, console=console)
    assert readiness["label_api"] == label_api and readiness["text_api"] == "responses"
    assert f"label_api={label_api}" in llm_readiness_guidance(readiness)


@pytest.mark.parametrize("missing", ["credentials", "client"])
def test_missing_hub_setup_has_prominent_warning_and_actionable_path(startup, missing):
    rt, console = startup

    def unavailable():
        if missing == "credentials":
            raise RuntimeError("INFERENCE_HUB_API_KEY not set")
        return None

    rt.ensure_hub_llm = unavailable
    readiness = check_llm_readiness(rt, console=console)
    output = console.output.getvalue()
    assert readiness["status"] == "not_configured"
    assert readiness["configured"] is False and readiness["available"] is False
    assert "INFERENCE_HUB_API_KEY" in output
    assert readiness["env_path"].endswith("profile/.env")
    assert readiness["env_path"] in output
    guidance = llm_readiness_guidance(readiness)
    assert "not_configured" in guidance and readiness["env_path"] in guidance


def test_unconfigured_endpoint_names_the_missing_setting(startup):
    from autofa.llm.endpoints import NO_ENDPOINT_HELP
    from autofa.mcp.runtime import Runtime

    _, console = startup
    # The real runtime path: endpoint resolution, then client creation, then the probe.
    rt = Runtime(dataset=None, library=None, run=None,
                 config=Config(feature_model="test-feature-model"))
    readiness = check_llm_readiness(rt, console=console)
    output = " ".join(console.output.getvalue().split())
    assert readiness["status"] == "not_configured" and readiness["base_url"] == ""
    assert readiness["credential_help"] == NO_ENDPOINT_HELP
    assert "No LLM endpoint is configured" in readiness["error"]
    assert "No LLM endpoint is configured" in output and "feature_base_url" in output


@pytest.mark.parametrize("error,status", [
    ("auth failed (HTTP 401) — check API key", "unavailable"),
    ("timed out after 20s", "unverified"),
])
def test_configured_probe_failure_does_not_claim_missing_credentials(startup, error, status):
    rt, console = startup
    client, calls = configured(rt, result=HubProbeResult(
        ok=False, model="test", error=error, label_api="decisions", text_api="responses"))
    readiness = check_llm_readiness(rt, console=console)
    assert calls == ["probe"] and client.available is True
    assert readiness["status"] == status and readiness["configured"] is True
    # A failed ping never claims the route it would have used.
    assert "label_api" not in readiness
    # Unverified (e.g. a timeout) keeps LLM detectors enabled; a rejection disables them.
    assert readiness["available"] is (False if status == "unavailable" else None)


@pytest.mark.parametrize("kind", ["auth", "timeout"])
def test_raised_probe_error_is_humanized_without_disabling_client(startup, kind):
    rt, console = startup
    if kind == "auth":
        error = RuntimeError("request rejected")
        error.status_code = 401
    else:
        error = TimeoutError("timed out")
    client, _ = configured(rt, error=error)
    readiness = check_llm_readiness(rt, console=console)
    assert readiness["status"] == ("unavailable" if kind == "auth" else "unverified")
    assert client.available is True
    if kind == "auth":
        assert readiness["error"] == "auth failed (HTTP 401) — check API key"


def test_reachable_hub_is_ready_without_echoing_its_key(startup, monkeypatch):
    rt, console = startup
    monkeypatch.setenv("INFERENCE_HUB_API_KEY", "private-test-hub-key")
    _, calls = configured(rt)
    readiness = check_llm_readiness(rt, console=console)
    assert calls == ["probe"]
    assert readiness["status"] == "ready"
    assert readiness["configured"] is True and readiness["available"] is True
    assert readiness["key_env"] == "INFERENCE_HUB_API_KEY"
    assert "private-test-hub-key" not in console.output.getvalue()
    assert "Semantic LLM detectors are available" in llm_readiness_guidance(readiness)


@pytest.mark.parametrize("truncated", [False, True])
def test_error_output_and_prompt_never_echo_known_hub_key(startup, monkeypatch, truncated):
    rt, console = startup
    secret = "private-hub-key-" + "sensitive-value-" * 20
    monkeypatch.setenv("INFERENCE_HUB_API_KEY", secret)
    error = RuntimeError(f"Request failed with key {secret}")
    if truncated:
        configured(rt, result=HubProbeResult(ok=False, model="test",
                                            error=humanize_hub_error(error)))
    else:
        configured(rt, error=error)
    readiness = check_llm_readiness(rt, console=console)
    rendered = console.output.getvalue() + llm_readiness_guidance(readiness) + json.dumps(readiness)
    assert "private-hub-key-" not in rendered
    assert "sensitive-value" not in rendered
    assert "[redacted]" in rendered


def test_client_key_is_redacted_even_without_environment_variable(startup):
    rt, console = startup
    secret = "client-only-private-key"
    client, _ = configured(rt, error=RuntimeError(f"Connection failed: {secret}"))
    client._client = SimpleNamespace(api_key=secret)
    readiness = check_llm_readiness(rt, console=console)
    assert secret not in console.output.getvalue() + json.dumps(readiness)


def test_custom_endpoint_warning_uses_its_key_and_redacts_it(startup, monkeypatch):
    rt, console = startup
    rt.config.feature_base_url = "https://provider.example/api/v1"
    rt.config.feature_api_key_env = " PROVIDER_KEY "
    monkeypatch.setenv("PROVIDER_KEY", "private-provider-key")
    configured(rt, error=RuntimeError("Connection failed: private-provider-key"))
    readiness = check_llm_readiness(rt, console=console)
    rendered = console.output.getvalue() + json.dumps(readiness)
    assert readiness["key_env"] == "PROVIDER_KEY"
    assert readiness["base_url"] == "https://provider.example/api/v1"
    assert "Save PROVIDER_KEY" in rendered
    assert "private-provider-key" not in rendered
    assert "INFERENCE_API_KEY" not in rendered


def test_local_server_warning_explains_connection_without_requesting_key(startup):
    rt, console = startup
    rt.config.feature_base_url = "http://localhost:8000/v1"
    rt.config.feature_api_key_env = "none"
    configured(rt, error=ConnectionError("Connection refused"))
    readiness = check_llm_readiness(rt, console=console)
    assert readiness["status"] == "unverified"
    assert readiness["key_env"] is None
    assert "No API key is required" in console.output.getvalue()
    assert "http://localhost:8000/v1" in console.output.getvalue()


@pytest.mark.parametrize("field,value", [
    ("feature_base_url", "https://provider.example/v1"), ("feature_api_key_env", ["KEY"]),
])
def test_invalid_endpoint_config_is_reported_before_discovery(startup, field, value):
    rt, console = startup
    setattr(rt.config, field, value)
    configured(rt)
    readiness = check_llm_readiness(rt, console=console)
    assert readiness["status"] == "not_configured"
    assert "feature_api_key_env" in readiness["error"]
    assert "Settings > LLM endpoints" in console.output.getvalue()


@pytest.mark.parametrize("resumed", [False, True])
def test_session_warns_and_informs_agent_before_new_or_resumed_discovery(
    startup, tmp_path, monkeypatch, resumed,
):
    from autofa.core.library import FunctionLibrary
    from autofa.data.parquet_store import Dataset
    from autofa.mcp.runtime import Runtime

    _, console = startup
    cfg = Config(library_root=str(tmp_path / "catalog"))
    run = RunConfig(run_id=f"session-{resumed}", parquet_paths=[], model_id="model")
    rt = Runtime(dataset=Dataset(parquet_paths=[], model_id="model"),
                 library=FunctionLibrary(model_id="model"), run=run, config=cfg)

    def no_key():
        raise RuntimeError("INFERENCE_HUB_API_KEY not set")

    monkeypatch.setattr(rt, "ensure_hub_llm", no_key)
    module = runner
    monkeypatch.setattr(module, "console", console)

    def load_runtime(*args):
        # The real loader also installs the runtime used by MCP tools. Keep that
        # contract so final scoring sees this run's empty library, not a prior test.
        monkeypatch.setattr("autofa.mcp.runtime._RUNTIME", rt)
        return rt

    monkeypatch.setattr(module, "load_runtime_from_run", load_runtime)
    monkeypatch.setattr(module, "build_mcp_server", lambda *args: object())
    monkeypatch.setattr(module, "write_session_workspace", lambda *args, **kwargs: run.run_dir())
    monkeypatch.setattr("autofa.util.memory.apply_memory_tuning",
                        lambda *args, **kwargs: SimpleNamespace(summary=lambda: "test memory"))
    monkeypatch.setattr("autofa.data.remote_library_publish.maybe_publish_run",
                        lambda *args, **kwargs: None)
    calls = []

    async def discover(*args, **kwargs):
        calls.append(kwargs)
        assert "INFERENCE_HUB_API_KEY" in console.output.getvalue()
        assert rt.llm_readiness["status"] == "not_configured"
        assert "LLM detector backend: not_configured" in kwargs["system_prompt"]
        return None

    monkeypatch.setattr(runner, "get_harness", lambda *args, **kwargs: SimpleNamespace(
        run_session=discover,
    ))
    runner.run_session(run, cfg=cfg, continue_mode=resumed)
    assert len(calls) == 1


# Durable detector-service blockers during a run.

@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOFA_HOME', str(tmp_path))
    run = RunConfig(run_id='service', parquet_paths=[], model_id='synthetic', benchmarks=[])
    run.save()
    return SimpleNamespace(run=run, config=Config())


def http_error(status, body=None, message='request rejected'):
    error = RuntimeError(f'HTTP {status}: {message}')
    error.status_code = status
    if body is not None:
        error.body = {'error': body}
    return error


def observe(rt, *, error=None, role='detector', phase='discovery', readiness_failure=False):
    service.observe_dispatch({'run_dir': str(rt.run.run_dir()), 'role': role, 'phase': phase,
                              'round_index': rt.run.round}, model='synthetic',
                             endpoint='https://mock.invalid/v1', api='responses', error=error,
                             readiness_failure=readiness_failure)


def test_failure_then_success_keeps_history_and_exposes_compact_blocker(rt):
    observe(rt)
    exc = RuntimeError('HTTP 500 https://example.invalid/path?token=secret')
    exc.status_code = 500
    observe(rt, error=exc)
    status = service.detector_service_status(rt.run.run_dir())
    assert status['infrastructure_blocked']
    assert status['active']['category'] == 'transport'
    assert status['active']['next_retry_at']
    assert 'secret' not in json.dumps(status)
    observe(rt)
    status = service.detector_service_status(rt.run.run_dir())
    assert not status['infrastructure_blocked']
    assert status['active']['failures'] == 1 and status['active']['successes'] == 2


def test_unrelated_workflows_and_alternative_probe_do_not_block_active_service(rt):
    observe(rt)
    for role, phase in [('detector', 'taxonomy'), ('highlighting', 'highlighting'),
                        ('judge', 'discovery'), ('classifier_probe', 'discovery')]:
        observe(rt, error=ConnectionError('Synthetic failure'), role=role, phase=phase)
    assert not service.detector_service_status(rt.run.run_dir())['infrastructure_blocked']


@pytest.mark.parametrize('error', [TypeError('bad local argument'),
                                   RuntimeError('Unknown result parser fault')])
def test_local_or_unclassified_fault_does_not_claim_global_service_outage(rt, error):
    observe(rt)
    observe(rt, error=error)
    assert not service.detector_service_status(rt.run.run_dir())['infrastructure_blocked']


# The image cases quote an HTTP 404/401 that belongs to the item, not to the service;
# each would otherwise read as a configuration or authentication failure.
@pytest.mark.parametrize('status,body', [
    (400, {'code': 'context_length_exceeded', 'message': 'This item exceeds the context limit'}),
    (400, {'code': 'invalid_image_url', 'message': 'Fetching item image returned HTTP 404'}),
    (400, {'code': 'invalid_image_url', 'message': 'Fetching item image returned HTTP 401'}),
    (422, {'message': 'Invalid item payload'}),
])
def test_item_rejections_remain_unresolved_without_blocking_service(rt, status, body):
    observe(rt)
    observe(rt, error=http_error(status, body, 'item rejected'))
    summary = service.detector_service_status(rt.run.run_dir())
    assert not summary['infrastructure_blocked']
    assert summary['active']['category'] == 'request_rejected'
    assert summary['active']['failures'] == 1
    assert summary['active']['successes'] == 1  # Rejection never becomes a successful prediction.


@pytest.mark.parametrize('body', [
    {'code': 'model_not_found', 'message': 'Invalid deployment'},
    {'param': 'model', 'message': 'Unknown deployment ID'},
    {'message': 'Unknown model selected'},
])
def test_confirmed_configuration_failures_still_block_immediately(rt, body):
    observe(rt, error=http_error(400, body))
    summary = service.detector_service_status(rt.run.run_dir())
    assert summary['infrastructure_blocked']
    assert summary['active']['category'] == 'configuration'


def test_provider_budget_and_previously_saved_budget_rejection_block_until_recovery(rt):
    observe(rt)
    observe(rt, error=http_error(422, {'type': 'budget_exceeded', 'code': '422'}))
    folder = rt.run.run_dir()
    summary = service.detector_service_status(folder)
    assert summary['infrastructure_blocked'] and summary['active']['category'] == 'provider_budget'
    path = folder / service.STATE_FILE
    state = json.loads(path.read_text())
    state['blocked'] = False
    state['routes'][state['active_route']]['category'] = 'request_rejected'
    path.write_text(json.dumps(state))
    assert service.infrastructure_block(rt)['active']['category'] == 'provider_budget'
    observe(rt)
    assert service.infrastructure_block(rt) is None


def test_readiness_rejections_still_block_immediately(rt):
    observe(rt, error=http_error(422), readiness_failure=True)
    assert service.detector_service_status(rt.run.run_dir())['infrastructure_blocked']


def test_item_rejection_or_alternative_probe_cannot_clear_existing_service_outage(rt):
    observe(rt, error=ConnectionError('Selected route failed'))
    observe(rt, error=http_error(422, message='invalid item payload'))
    assert service.detector_service_status(rt.run.run_dir())['infrastructure_blocked']
    observe(rt, role='classifier_probe')
    assert service.detector_service_status(rt.run.run_dir())['infrastructure_blocked']


def test_health_observer_does_not_modify_other_workflows(rt):
    rt.run.analysis_outputs = ['efficiency']
    rt.run.save()
    observe(rt, error=ConnectionError('Unrelated workflow failure'))
    assert not (rt.run.run_dir() / service.STATE_FILE).exists()


def test_bounded_recovery_turns_survive_resume_and_round_changes(rt):
    observe(rt, error=ConnectionError('Synthetic outage'))
    assert service.recovery_turn_guidance(rt)
    rt.run.round += 1
    assert service.recovery_turn_guidance(rt)
    assert service.recovery_turn_guidance(rt) is None


def test_unavailable_work_does_not_consume_stagnation_or_destroy_saved_history(rt):
    from autofa.session import round_stagnation

    observe(rt, error=ConnectionError('Synthetic outage'))
    path = rt.run.run_dir() / 'discovery_round_stagnation.json'
    saved = {'version': 1, 'run_id': rt.run.run_id,
             'progress_accounting_version': 3, 'policy': round_stagnation._policy(rt),
             'rounds_without_improve': 2, 'known_detectors': [{'name': 'retained'}]}
    atomic_write_json(path, saved)
    result = round_stagnation.complete_discovery_round(rt, {'ok': True, 'fraction': 1.0})
    assert result['status'] == 'infrastructure_blocked' and not result['counted']
    assert json.loads(path.read_text()) == saved


def test_item_rejection_does_not_skip_final_scoring(interactive_runner, monkeypatch):
    _, _, rt = interactive_runner
    rt.run.campaign_policy = {'round_limit': rt.run.round}
    rt.run.save()
    observe(rt, error=http_error(422, message='invalid image in one item'))
    finalizations = []

    async def finalize(console):
        finalizations.append(True)

    monkeypatch.setattr('autofa.session.finalization.finalize_discovery', finalize)
    runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=True, publish_on_exit=False)
    assert finalizations == [True]
    target = json.loads((rt.run.run_dir() / 'discovery_target.json').read_text())
    assert target['stop_reason'] == 'round_limit'


@pytest.mark.parametrize('interactive', ['claude', 'codex'], indirect=True)
def test_midrun_outage_has_finite_recovery_and_partial_report(interactive_runner, monkeypatch):
    harness, _, rt = interactive_runner
    turns, finalizations, reports = [], [], []
    original_round = rt.run.round

    async def turn(prompt, **kwargs):
        turns.append(prompt)
        assert len(turns) <= 3
        if len(turns) == 1:
            observe(rt, error=ConnectionError('Synthetic outage'))
        return TurnResult(session_id='same-session', text='Detector service unavailable.')

    async def finalize(console):
        finalizations.append(True)

    def report(run, **kwargs):
        reports.append(run.run_id)
        return run.run_dir() / 'failure_cards.html'

    monkeypatch.setattr(harness, 'run_turn', turn)
    monkeypatch.setattr(runner, 'write_run_report', report)
    monkeypatch.setattr('autofa.session.finalization.finalize_discovery', finalize)
    runner.run_session(rt.run, cfg=rt.config, exit_after_discovery=True, publish_on_exit=False)
    assert len(turns) == 3 and 'DETECTOR SERVICE RECOVERY' in turns[-1]
    assert not finalizations and reports == [rt.run.run_id]
    assert rt.run.round == original_round
    target = runner.require_batch_target(rt.run, cfg=rt.config)
    assert target['stop_reason'] == 'infrastructure_blocked' and not target['goal_met']
    outcome = json.loads((rt.run.run_dir() / 'finalization_outcome.json').read_text())
    assert outcome['state'] == 'blocked' and outcome['resumable']
    assert outcome['report_generated'] and not outcome['retained_results_complete']
