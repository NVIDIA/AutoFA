# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classifier profiles are set up explicitly, never invent labels, and adopt only on controls."""

import asyncio
import json
import tomllib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from autofa.config import Config, RunConfig
from autofa.core.library import FeatureFunction, FunctionLibrary
from autofa.core.sandbox import DetectorAbstained
from autofa.data.parquet_store import Dataset
from autofa.llm.hub import HubLLM, completion_text
from autofa.mcp import runtime, server, tools_classifier, tools_eval, tools_library
from autofa.preflight.classifiers import main, profile_readiness, propose_profile
from autofa.session import classifier_profiles as profiles
from autofa.session.review_evidence import review_context

from support.judge import client_for, response

# Setup preflight: readiness and proposals are read-only until explicitly saved.


def secret_options():
    return {"max_tokens": 1024, "reasoning_effort": "medium", "extra_body": {
        "vendor": {"Authorization": "Bearer literal-auth", "apiKey": "literal-key",
                   "extra_headers": {"x-private": "literal-header"},
                   "nested": [{"refresh_token": "literal-token"},
                              {"password": "literal-password", "sampling": "retained"}]}}}


def test_ladder_preview_never_authorizes_profiles_claims_capabilities_or_creates_config(
    tmp_path, capsys,
):
    path = tmp_path / 'absent' / 'config.toml'
    assert main(['--config', str(path), '--suggest-ladder', '--luna-model', 'custom/luna',
                 '--sol-model', 'custom/sol']) == 0
    result = json.loads(capsys.readouterr().out)
    # The configured default ladder is permitted; previewing it makes no calls.
    assert result['alternatives_permitted'] == 3
    assert result['inference_calls'] == 0
    assert result['profiles'][0]['capabilities']['vision'] == 'unverified'
    assert not result['proposed_ladder']['authorized']
    assert result['proposed_ladder']['proposed_profiles'][1]['feature_model'] == 'custom/sol'
    assert not path.parent.exists()


def test_profile_preview_and_explicit_save_preserve_other_settings(tmp_path, capsys):
    path = tmp_path / "config.toml"
    path.write_text('objective_weights = [50, 20, 30]\n# retained\n[private]\nkey="unchanged"\n')
    before = path.read_bytes()
    args = ["--config", str(path), "--add-profile", "reasoning", "--model", "approved-deployment",
            "--base-url", "https://example.invalid/v1", "--key-env", "APPROVED_PROVIDER_KEY",
            "--api", "chat_completions", "--request-options", '{"reasoning_effort":"medium"}']
    assert main(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert not preview["configuration_written"]
    assert path.read_bytes() == before
    assert main([*args, "--write-config"]) == 0
    saved = tomllib.loads(path.read_text())
    # A new profile follows the default ladder, which the saved file now lists explicitly.
    assert [p["name"] for p in saved["classifier_profiles"]] == [
        "luna_medium", "sol_medium", "sol_high", "reasoning"]
    assert saved["classifier_profiles"][-1] == preview["proposed_profile"]
    assert saved["objective_weights"] == [50, 20, 30]
    assert saved["private"] == {"key": "unchanged"}
    assert "# retained" in path.read_text()


def test_provider_change_requires_explicit_auth_and_does_not_enable_model():
    cfg = Config()
    configured = json.loads(json.dumps(cfg.classifier_profiles))
    with pytest.raises(ValueError, match="different provider"):
        propose_profile(cfg, name="reasoning", model="approved", base_url="https://other.test/v1")
    assert cfg.classifier_profiles == configured
    proposed = propose_profile(cfg, name="reasoning", model="approved",
                               base_url="http://localhost:1234/v1", key_env="none",
                               request_options={})
    assert proposed["feature_api_key_env"] == "none"
    assert proposed["feature_request_options"] == {}


def test_inherited_profile_endpoint_errors_wait_for_client_creation():
    from autofa.llm.endpoints import endpoint_for_config

    # The default ladder inherits the endpoint, so an incomplete one still loads.
    cfg = Config(feature_base_url="https://custom.invalid/v1")
    assert len(cfg.classifier_profiles) == 3
    with pytest.raises(ValueError, match="feature_api_key_env"):
        endpoint_for_config(cfg)
    with pytest.raises(ValueError, match="feature_api"):
        Config(classifier_profiles=[{"name": "bad", "feature_model": "m", "feature_api": "x"}])
    with pytest.raises(ValueError, match="feature_api_key_env"):
        Config(classifier_profiles=[{"name": "own", "feature_model": "m",
                                     "feature_base_url": "https://other.invalid/v1"}])


def test_secret_values_never_appear_in_readiness(monkeypatch):
    monkeypatch.setenv("EXAMPLE_KEY", "private-secret")
    cfg = Config(feature_base_url="https://example.invalid/v1", feature_api_key_env="EXAMPLE_KEY")
    result = profile_readiness(cfg)
    assert result["profiles"][0]["credential_present"]
    assert "private-secret" not in json.dumps(result)


def test_default_profile_inspection_does_not_create_home(tmp_path, monkeypatch, capsys):
    from autofa.config import config_path

    home = tmp_path / "absent-profile"
    monkeypatch.setenv("AUTOFA_HOME", str(home))
    monkeypatch.delenv("AUTOFA_CONFIG", raising=False)
    assert main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["config_path"] == str(home / "config.toml")
    assert not home.exists()
    # The read-only inspection leaves ordinary default creation intact.
    assert config_path() == home / "config.toml"
    assert home.is_dir()


def test_nested_sensitive_options_redacted_without_changing_config():
    options = secret_options()
    before = deepcopy(options)
    cfg = Config(feature_request_options=options)
    report = profile_readiness(cfg)
    serialized = json.dumps(report)
    assert "literal-" not in serialized
    visible = report["profiles"][0]["request_options"]
    assert visible["max_tokens"] == 1024 and visible["reasoning_effort"] == "medium"
    vendor = visible["extra_body"]["vendor"]
    assert vendor["Authorization"] == vendor["apiKey"] == vendor["extra_headers"] == "[redacted]"
    assert vendor["nested"][1]["sampling"] == "retained"
    assert cfg.feature_request_options == before


def test_sensitive_preview_redacted_but_explicit_fresh_save_preserves_values(
    tmp_path, monkeypatch, capsys,
):
    from autofa.harness import settings

    monkeypatch.setattr(settings, "detect_default_harness", lambda **kwargs: "claude")
    path = tmp_path / "config.toml"
    options = secret_options()
    args = ["--config", str(path), "--add-profile", "approved", "--model", "approved",
            "--request-options", json.dumps(options)]
    assert main(args) == 0
    assert "literal-" not in capsys.readouterr().out
    assert not path.exists()
    assert main([*args, "--write-config"]) == 0
    assert "literal-" not in capsys.readouterr().out
    saved = tomllib.loads(path.read_text())
    assert saved["classifier_profiles"][-1]["feature_request_options"] == options
    # A fresh profile config uses the normal harness detection.
    assert saved["harness"] == "claude"


def test_ladder_preview_redacts_inherited_sensitive_options(tmp_path, capsys):
    import tomlkit

    path = tmp_path / 'config.toml'
    path.write_text(tomlkit.dumps({'feature_request_options': secret_options()}))
    before = path.read_bytes()
    assert main(['--config', str(path), '--suggest-ladder', '--luna-model', 'luna',
                 '--sol-model', 'sol']) == 0
    assert 'literal-' not in capsys.readouterr().out
    assert path.read_bytes() == before


def test_probe_error_redacts_nested_credential_options(monkeypatch):
    from autofa.preflight.classifiers import probe_profile

    class FakeSDK:
        api_key = "literal-sdk-key"

        def with_options(self, **kwargs):
            return self

        def close(self):
            pass

    monkeypatch.setattr("autofa.llm.hub.make_hub_client", lambda **kwargs: FakeSDK())
    result = SimpleNamespace(ok=False, label_api=None, text_api=None,
                             error="literal-sdk-key literal-token literal-header literal-password")
    monkeypatch.setattr("autofa.preflight.classifiers.HubLLM",
                        lambda *args, **kwargs: SimpleNamespace(probe=lambda **kwargs: result))
    report = probe_profile(Config(feature_request_options=secret_options()), "baseline")
    assert not report["ok"]
    assert "literal-" not in json.dumps(report)


@pytest.mark.parametrize("explicit,failed", [(False, False), (True, True)])
def test_standalone_probe_records_usage_without_modifying_profile(
    tmp_path, monkeypatch, capsys, explicit, failed,
):
    import httpx
    from openai import OpenAI

    path = tmp_path / "config.toml"
    path.write_text('feature_api = "chat_completions"\nfeature_model = "model"\n')
    before = path.read_bytes()
    usage_dir = tmp_path / "selected.usage" if explicit else path.with_suffix(".probe.usage")

    def respond(request):
        if failed:
            return httpx.Response(503, json={"error": {"message": "Temporary failure"}})
        return httpx.Response(200, json={
            "id": "probe", "created": 0, "model": "model", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 7},
        })

    def make_client(**kwargs):
        return OpenAI(api_key="private-api-key", max_retries=0, http_client=httpx.Client(
            transport=httpx.MockTransport(respond),
        ))

    monkeypatch.setattr("autofa.llm.hub.make_hub_client", make_client)
    args = ["--config", str(path), "--probe", "baseline"]
    if explicit:
        args += ["--usage-dir", str(usage_dir)]
    assert main(args) == (1 if failed else 0)
    output = capsys.readouterr().out
    assert "private-api-key" not in output
    result = json.loads(output)
    assert result["usage_dir"] == str(usage_dir)
    assert path.read_bytes() == before
    totals = result["autofa_usage"]["totals"]
    assert (totals["calls"], totals["missing_usage_calls"]) == (1, 1 if failed else 0)
    assert (usage_dir / "usage.sqlite3").is_file()


def test_usage_destination_requires_explicit_inference(tmp_path):
    path = tmp_path / "new-profile" / "config.toml"
    with pytest.raises(SystemExit):
        main(["--config", str(path), "--usage-dir", str(tmp_path / "usage")])
    assert not path.parent.exists() and not (tmp_path / "usage").exists()


def test_first_profile_write_migrates_legacy_config(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from autofa import config

    monkeypatch.delenv('AUTOFA_HOME', raising=False)
    monkeypatch.delenv('AUTOFA_CONFIG', raising=False)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path / 'home'))
    project = tmp_path / 'project'
    project.mkdir()
    monkeypatch.setattr(config, 'project_root', lambda: project)
    legacy = Path.home() / '.autofa/config.toml'
    legacy.parent.mkdir(parents=True)
    original = ('harness="claude"\nfeature_base_url="https://custom.invalid/v1"\n'
                'feature_api_key_env="none"\n')
    legacy.write_text(original)
    assert main(['--add-profile', 'approved', '--model', 'approved']) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview['config_source_path'] == str(legacy)
    assert preview['profiles'][0]['base_url'] == 'https://custom.invalid/v1'
    assert preview['proposed_profile']['feature_base_url'] == 'https://custom.invalid/v1'
    assert not (project / 'autofa.toml').exists()
    assert legacy.read_text() == original
    main(['--add-profile', 'approved', '--model', 'approved', '--write-config'])
    saved = tomllib.loads((project / 'autofa.toml').read_text())
    assert saved['harness'] == 'claude'
    assert saved['feature_base_url'] == 'https://custom.invalid/v1'
    assert [p['name'] for p in saved['classifier_profiles']] == [
        'luna_medium', 'sol_medium', 'sol_high', 'approved']
    assert saved['classifier_profiles'][-1]['feature_base_url'] == 'https://custom.invalid/v1'
    assert saved['classifier_profiles'][-1] == preview['proposed_profile']
    assert legacy.read_text() == original


@pytest.mark.parametrize('kwargs', [
    {'classifier_comparison_limit': -1},
    {'classifier_profiles': [{'name': 'x', 'feature_model': 'test', 'api_key': 'secret'}]},
    {'classifier_profiles': [{'name': 'baseline', 'feature_model': 'test'}]},
    {'classifier_profiles': [{'name': 'x', 'feature_model': ''}]},
    {'classifier_profiles': [{'name': 'x', 'feature_model': 'test',
                             'feature_request_options': {'extra_headers': {}}}]},
], ids=['negative-limit', 'inline-secret', 'reserved-name', 'empty-model', 'header-options'])
def test_profile_config_rejects_unsafe_or_unbounded_settings(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


# Bounded transport recovery never invents a detector label or changes explicit limits.


@pytest.mark.parametrize("reply", [
    response("yes", "content_filter"), response("yes", refusal="not available"),
    # A refusal outranks truncation: no length-retry call and no "incomplete" error.
    response("yes", "length", refusal="not available"),
    response([{"type": "refusal", "refusal": "not available"}]),
], ids=["content-filter", "refusal-field", "truncated-refusal", "refusal-block"])
def test_refusal_is_typed_unknown_without_retry(reply):
    with pytest.raises(DetectorAbstained, match="no label"):
        completion_text(reply)
    client, calls = client_for(reply)
    with pytest.raises(DetectorAbstained):
        HubLLM(client, api="chat_completions").classify_options("evidence", "Question")
    assert len(calls) == 1


def test_decision_refusal_is_typed_unknown_without_other_api_fallback(monkeypatch):
    client, calls = client_for()
    hub = HubLLM(client, api="decisions")
    monkeypatch.setattr(hub, "decide", lambda *a, **kw: [{"type": "refusal"}])
    with pytest.raises(DetectorAbstained):
        hub.classify_options("evidence", "Question")
    assert calls == []


def test_last_completion_rescue_does_not_raise_shared_floor():
    client, calls = client_for(response("", "length"), response("", "length"),
                               response("no"), response("yes"))
    hub = HubLLM(client, api="chat_completions")
    assert hub.classify_options("hard evidence", "Question") == "no"
    assert hub.classify_options("other evidence", "Question") == "yes"
    assert [c["max_tokens"] for c in calls] == [64, 4096, 8192, 4096]
    assert calls[0]["messages"] == calls[1]["messages"] == calls[2]["messages"]


def test_short_invalid_label_is_reassessed_once_and_never_fuzzy_mapped():
    client, calls = client_for(response("yess"), response("no"))
    hub = HubLLM(client, api="chat_completions")
    assert hub.classify_options("evidence", "Question", max_tokens=17) == "no"
    assert [c["max_tokens"] for c in calls] == [17, 17]
    assert calls[0]["messages"][0] == calls[1]["messages"][0]
    assert calls[1]["messages"][1] == {"role": "assistant", "content": "yess"}
    client, calls = client_for(response("yess"), response("yess"))
    with pytest.raises(RuntimeError, match="exactly one"):
        HubLLM(client, api="chat_completions").classify_options("evidence", "Question")
    assert len(calls) == 2
    # An explanation is not a short formatting slip, so it is never reassessed.
    client, calls = client_for(response("The answer might be yes"))
    with pytest.raises(RuntimeError, match="exactly one"):
        HubLLM(client, api="chat_completions").classify_options("evidence", "Question")
    assert len(calls) == 1


# Native paired diagnostics are bounded and cannot turn author controls into grades.

SOURCE = '''
def detect(item, llm_client=None):
    return llm_client.classify_options(item['response_text'],
        'Does the observed trace contradict the required condition?',
        options=['yes', 'no']) == 'yes'
'''


class Controls(Dataset):
    def __init__(self):
        super().__init__(parquet_paths=[])
        self.items = [
            {'_item_id': 'p', 'response_text': 'Observed required condition violated.',
             'problem_id': 'p', 'is_correct': False},
            {'_item_id': 'n', 'response_text': 'Observed condition satisfied after correction.',
             'problem_id': 'n', 'is_correct': True},
        ]

    def iter_batches(self, *, batch_size=512, max_items=None, columns=None):
        yield self.items[:max_items]


@pytest.fixture
def rt(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTOFA_HOME', str(tmp_path))
    cfg = Config(feature_model='base', feature_api='chat_completions',
                 feature_base_url='https://mock.invalid/v1', feature_api_key_env='none',
                 feature_request_options={}, llm_eval_workers=1,
                 classifier_profiles=[{'name': 'strong', 'feature_model': 'strong'}])
    run = RunConfig(run_id='paired', parquet_paths=[], model_id='tested', benchmarks=['bench'])
    live = runtime.Runtime(dataset=Controls(), library=FunctionLibrary(), run=run, config=cfg)
    live.calls = []
    live.outputs = {'base': ['no', 'no'], 'strong': ['yes', 'no']}
    live.dispatch_hook = None
    live.account = 'test-account'

    def client(config):
        def create(**kwargs):
            live.calls.append(kwargs['model'])
            if live.dispatch_hook:
                live.dispatch_hook(kwargs)
            body = str(kwargs['messages'])
            index = 0 if 'violated' in body else 1
            result = live.outputs[kwargs['model']][index]
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=result), finish_reason='stop')])
        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                              base_url='https://mock.invalid/v1', api_key=live.account,
                              close=lambda: None)
        return HubLLM(sdk, model=config.feature_model, api='chat_completions', request_options={})

    monkeypatch.setattr(profiles, '_client', client)
    live.hub_llm = client(cfg)
    monkeypatch.setattr(tools_classifier, 'get_runtime', lambda: live)
    monkeypatch.setattr(runtime, 'get_runtime', lambda: live)
    monkeypatch.setattr(tools_eval, 'get_runtime', lambda: live)
    # Context identity is tested independently; this fixture has no parquet-backed corpus.
    monkeypatch.setattr(tools_eval, '_evaluation_context', lambda rt, **kw: rt.config.feature_model)
    live.persist_library = lambda: live.library.save(run.run_dir() / 'library.json')
    return live


def compare(**overrides):
    args = dict(action='compare', profile='strong', source=SOURCE,
                item_ids=['p', 'n'], expected_outputs={'p': True, 'n': False},
                diagnosis='Original evidence checked; the default misses an explicit condition.')
    args.update(overrides)
    return tools_classifier.classifier_profiles(**args)


def adopt(comparison):
    return tools_classifier.classifier_profiles('adopt', comparison_id=comparison['comparison_id'])


def _block_service(rt):
    from autofa.session.detector_service import record_readiness

    rt.run.save()
    record_readiness(rt, {'available': False, 'error': 'HTTP 500: service unavailable'})


def _recover_through_tool(monkeypatch):
    monkeypatch.setattr(server, '_try_record_tool_exposure', lambda *args: None)
    monkeypatch.setattr(server, '_with_discovery_recovery', lambda name, result: result)
    result = asyncio.run(asyncio.wait_for(server.invoke_tool('classifier_profiles', {
        'action': 'recover', 'source': SOURCE, 'item_ids': ['p', 'n'],
        'expected_outputs': {'p': True, 'n': False}, 'diagnosis': 'Service outage',
    }), timeout=5))
    return json.loads(result['content'][0]['text'])


def test_native_paired_comparison_is_cached_frozen_and_score_free(rt):
    before = rt.library.to_dict()
    first = compare()
    assert first['ok'], first
    assert first['status'] == 'complete' and first['adoption_eligible']
    assert first['outcome'] == 'capability_improved'
    assert first['run_id'] == rt.run.run_id
    assert first['baseline']['checks_passed'] is False
    assert first['fallback']['checks_passed'] is True
    assert len(rt.calls) == 4
    for side in ['baseline', 'fallback']:
        assert first[side]['diagnostic_receipt_id']
    again = compare()
    assert again['reused'] and again['comparison_id'] == first['comparison_id']
    assert len(rt.calls) == 4
    assert rt.library.to_dict() == before
    assert not rt.run.classifier_profile_settings and rt.config.feature_model == 'base'
    changed = compare(expected_outputs={'p': False, 'n': True})
    assert not changed['ok'] and 'frozen' in changed['error']
    assert len(rt.calls) == 4


def test_resumed_passing_baseline_does_not_dispatch_unused_fallback(rt, monkeypatch):
    rt.outputs['base'] = ['yes', 'no']
    save = profiles._save

    def interrupt_after_baseline(runtime, state):
        save(runtime, state)
        if any(record.get('baseline', {}).get('checks_passed')
               and record['status'] == 'incomplete'
               for record in state['comparisons'].values()):
            raise KeyboardInterrupt('Interrupted after the baseline panel was saved')

    monkeypatch.setattr(profiles, '_save', interrupt_after_baseline)
    with pytest.raises(KeyboardInterrupt, match='baseline panel was saved'):
        compare()
    assert rt.calls == ['base', 'base']

    monkeypatch.setattr(profiles, '_save', save)
    result = compare()
    assert result['ok'] and result['status'] == 'complete'
    assert result['outcome'] == 'baseline_controls_pass'
    assert not result['adoption_eligible'] and 'fallback' not in result
    assert rt.calls == ['base', 'base']


def test_availability_recovery_skips_unavailable_tiers_without_quality_claim(rt):
    from autofa.session.detector_service import detector_service_status

    rt.config.classifier_profiles += [{'name': 'stronger', 'feature_model': 'stronger'}]
    rt.outputs['stronger'] = ['yes', 'no']
    _block_service(rt)
    before = deepcopy(rt.library.to_dict())

    def unavailable(kwargs):
        if kwargs['model'] == 'base':
            exc = RuntimeError('HTTP 500: synthetic service outage')
            exc.status_code = 500
            raise exc
        if kwargs['model'] == 'strong':
            raise ConnectionError('Synthetic service unavailable')
    rt.dispatch_hook = unavailable
    result = compare(action='recover', profile=None)
    assert result['ok'] and result['recovered'], result
    assert result['status'] == 'availability_replacement'
    assert result['quality_improvement'] is False
    assert rt.config.feature_model == 'stronger'
    assert set(rt.calls) <= {'base', 'strong', 'stronger'}  # Only approved profiles dispatch.
    assert rt.run.classifier_profile_settings['selection_kind'] == 'availability_replacement'
    assert rt.library.to_dict() == before
    assert not detector_service_status(rt.run.run_dir())['infrastructure_blocked']
    assert not profiles.profile_status(rt)['comparisons']


def test_availability_controls_cannot_regress_or_assign_grades(rt):
    _block_service(rt)
    rt.outputs['strong'] = ['yes', 'yes']
    result = compare(action='recover')
    assert result['ok'] and not result['recovered'], result
    assert not rt.run.classifier_profile_settings
    assert rt.config.feature_model == 'base'
    count = len(rt.calls)
    second = compare(action='recover')
    assert not second['recovered']
    assert len(rt.calls) == count  # Saved backoff survives another request.
    rt.run.round += 1
    assert not compare(action='recover')['recovered']
    assert len(rt.calls) == count  # A round change grants no new attempt.


def test_recovery_preserves_user_stop_and_existing_round_budget(rt):
    from autofa.session.supervision import SupervisorInterrupted, lifecycle_request

    _block_service(rt)
    rounds = rt.run.max_rounds
    lifecycle_request(rt.run.run_dir(), 'stop')
    with pytest.raises(SupervisorInterrupted):
        compare(action='recover')
    assert not rt.calls and rt.run.max_rounds == rounds


def test_startup_recovery_checks_explicit_vision_and_records_capability_only(rt, monkeypatch):
    from autofa.session.classifier_recovery import recover_classifier

    _block_service(rt)
    seen = []

    def client(cfg):
        def classify(text, prompt, *, options, images=None):
            if cfg.feature_model == 'base':
                raise ConnectionError('Baseline unavailable')
            seen.append((text, images))
            return ('red' if images else 'yes' if text == 'The lamp is on.' else 'no')
        return SimpleNamespace(classify_options=classify)
    monkeypatch.setattr(profiles, '_client', client)
    monkeypatch.setattr('autofa.setup_contract.required_capabilities', lambda *args: ['vision'])
    result = recover_classifier(rt, startup=True)
    assert result['recovered'] and rt.config.feature_model == 'strong'
    assert result['controls']['scope'] == 'startup_capability_only_not_detector_validation'
    assert len(seen) == 3 and seen[-1][1][0]['status'] == 'available'
    assert not rt.library.functions


@pytest.mark.parametrize('intent', ['stop', 'finalize'])
def test_supervisor_intent_during_recovery_returns_tool_result(rt, monkeypatch, intent):
    from autofa.session.supervision import lifecycle_request
    from autofa.util.cancellation import current_cancellation

    _block_service(rt)
    requested = []

    def dispatch(kwargs):
        if kwargs['model'] == 'base':
            raise ConnectionError('Synthetic baseline outage')
        if not requested:
            requested.append(True)
            lifecycle_request(rt.run.run_dir(), intent)
            current_cancellation()._last_poll = 0
    rt.dispatch_hook = dispatch
    payload = _recover_through_tool(monkeypatch)
    assert payload['status'] == ('campaign_boundary' if intent == 'finalize'
                                 else 'stopped_by_supervisor'), payload
    assert not rt.run.classifier_profile_settings
    if intent == 'finalize':
        assert payload['stop_reason'] == 'user_finalize'


def test_recovery_selection_committed_before_stop_keeps_completed_tool_result(rt, monkeypatch):
    from autofa.session.classifier_recovery import recovery_status
    from autofa.session.supervision import lifecycle_request
    from autofa.util.cancellation import current_cancellation

    _block_service(rt)

    def dispatch(kwargs):
        if kwargs['model'] == 'base':
            raise ConnectionError('Synthetic baseline outage')
    rt.dispatch_hook = dispatch
    original = profiles.apply_profile_selection

    def select(*args, **kwargs):
        original(*args, **kwargs)
        lifecycle_request(rt.run.run_dir(), 'stop')
        current_cancellation()._last_poll = 0
    monkeypatch.setattr(profiles, 'apply_profile_selection', select)
    payload = _recover_through_tool(monkeypatch)
    assert payload['ok'] and payload['recovered'], payload
    assert rt.run.classifier_profile_settings['name'] == 'strong'
    assert recovery_status(rt)['attempts'][-1]['outcome'] == 'availability_replacement'


def test_fallback_negative_regression_never_authorizes_adoption(rt):
    rt.outputs['strong'] = ['yes', 'yes']
    result = compare()
    assert result['ok'] and not result['adoption_eligible']
    denied = adopt(result)
    assert not denied['ok'] and not rt.run.classifier_profile_settings


def test_adoption_is_run_scoped_durable_and_preserves_saved_scores(rt, monkeypatch):
    from autofa.llm.endpoints import endpoint_for_config

    judge = endpoint_for_config(rt.config, 'judge')
    rt.library.upsert(FeatureFunction(
        name='existing', description='Existing condition', source=SOURCE, scores={
            'evaluated': True, 'evaluation_context': 'base', 'recall': .42,
            'behavior_verdicts': [{'verdict': 'no'}],
        }))
    before_scores = deepcopy(rt.library.get('existing').scores)
    original = rt.config
    result = compare()
    monkeypatch.setattr(rt, 'persist_library',
                        lambda: pytest.fail('Profile adoption must not rewrite saved scores'))
    adopted = adopt(result)
    assert adopted['ok'] and adopted['adopted'], adopted
    assert rt.config.feature_model == 'strong' and original.feature_model == 'base'
    assert rt.hub_llm is rt.llm_readiness is None
    assert rt.library.get('existing').scores == before_scores
    saved = json.loads((rt.run.run_dir() / 'config.json').read_text())
    assert saved['classifier_profile_settings']['name'] == 'strong'
    # Adoption snapshots the original judge route so a resume cannot drift with defaults.
    assert saved['classifier_profile_settings']['inherited_judge_settings'] == {
        'judge_base_url': judge.base_url, 'judge_api_key_env': judge.api_key_env,
        'judge_api': judge.api,
    }
    assert endpoint_for_config(rt.config, 'judge') == judge
    original.feature_base_url = 'https://new-default.invalid/v1'
    original.feature_api = 'responses'
    restored = profiles.apply_run_classifier(original, RunConfig(**saved))
    assert restored.feature_model == 'strong'
    assert endpoint_for_config(restored, 'judge') == judge
    assert original.feature_model == 'base'
    again = adopt(result)
    assert again['ok'] and again['already_adopted']


def test_failed_run_save_does_not_make_retry_skip_adoption(rt, monkeypatch):
    result = compare()
    save = rt.run.save
    monkeypatch.setattr(rt.run, 'save', lambda: (_ for _ in ()).throw(OSError('disk full')))
    failed = adopt(result)
    assert not failed['ok'] and not rt.run.classifier_profile_settings
    assert rt.config.feature_model == 'base'
    monkeypatch.setattr(rt.run, 'save', save)
    retried = adopt(result)
    assert retried['ok']
    assert rt.config.feature_model == 'strong'


@pytest.mark.parametrize('change', ['source', 'input', 'account', 'policy'])
def test_adoption_rejects_stale_comparison(rt, change):
    rt.library.upsert(FeatureFunction(
        name='candidate', description='Candidate condition', source=SOURCE))
    result = compare(source=None, name='candidate')
    assert result['ok'], result
    if change == 'source':
        rt.library.get('candidate').source += '\n# changed source\n'
    elif change == 'input':
        rt.dataset.items[0]['response_text'] = 'different evidence'
    elif change == 'account':
        rt.account = 'changed-account'
    else:
        rt.config.classifier_profiles[0]['feature_model'] = 'another'
    denied = adopt(result)
    assert not denied['ok'] and rt.config.feature_model == 'base'


def test_changed_evidence_between_profiles_is_not_adoptable(rt):
    def change(kwargs):
        if kwargs['model'] == 'strong':
            rt.dataset.items[0]['response_text'] = 'changed corpus evidence'
    rt.dispatch_hook = change
    result = compare()
    assert not result['ok'] and 'evidence changed' in result['error']
    assert not rt.run.classifier_profile_settings


def test_persistent_budget_cannot_be_reset_by_source_rename_or_round(rt):
    rt.config.classifier_comparison_limit = 1
    first = compare()
    assert first['ok']
    rt.run.round += 1
    denied = compare(source=SOURCE.replace('def detect', 'def renamed'))
    assert not denied['ok'] and 'budget exhausted' in denied['error']
    assert len(rt.calls) == 4
    assert compare()['reused']


def test_dispatch_budget_survives_interruption_and_resume(rt):
    from autofa.session.recovery import RunInterruptedError

    rt.dispatch_hook = lambda _: (_ for _ in ()).throw(ConnectionError('connection reset'))
    with pytest.raises(RunInterruptedError):
        compare()
    path = rt.run.run_dir() / profiles.STATE_FILE
    state = json.loads(path.read_text())
    assert len(state['comparisons']) == 1 and state['model_calls'] == 1
    record = next(iter(state['comparisons'].values()))
    record['model_calls'] = 120
    path.write_text(json.dumps(state))
    rt.dispatch_hook = None
    result = compare()
    assert result['ok'] and result['status'] == 'incomplete'
    assert result['baseline']['status'] == 'diagnostic_budget_exhausted'
    assert len(rt.calls) == 1


@pytest.mark.parametrize('overrides', [
    {'expected_outputs': {'p': True, 'n': True}},
    {'item_ids': ['p', 'p']},
    {'diagnosis': ''},
    {'profile': 'unconfigured'},
    {'item_ids': ['p', 'missing'], 'expected_outputs': {'p': True, 'missing': False}},
], ids=['no-negative-control', 'duplicate-item', 'no-diagnosis', 'unknown-profile',
        'missing-item'])
def test_invalid_comparison_fails_before_dispatch(rt, overrides):
    result = compare(**overrides)
    assert not result['ok'] and not rt.calls


def test_status_is_read_only_and_comparison_is_allowed_between_turns(rt):
    status = tools_classifier.classifier_profiles()
    assert status['ok'] and status['enabled']
    assert not (rt.run.run_dir() / profiles.STATE_FILE).exists()
    (rt.run.run_dir() / 'turn.json').write_text(json.dumps({'state': 'finished'}))
    result = compare()
    assert result['ok'] and rt.calls
    assert tools_classifier.classifier_profiles()['ok']


def test_comparison_is_closed_after_campaign_completion(rt):
    (rt.run.run_dir() / 'discovery_target.json').write_text(
        json.dumps({'campaign_complete': True}))
    assert not compare()['ok'] and not rt.calls


def test_queued_comparison_and_adoption_apply_at_finished_turn_boundary(rt):
    from autofa.session.control import consume_control_requests, submit_contribution
    from autofa.util.tool_result import result_payload

    rt.run.save()
    directory = rt.run.run_dir()
    (directory / 'turn.json').write_text(json.dumps({'state': 'finished'}))
    submit_contribution(directory, 'classifier_profiles', {
        'action': 'compare', 'profile': 'strong', 'source': SOURCE,
        'item_ids': ['p', 'n'], 'expected_outputs': {'p': True, 'n': False},
        'diagnosis': 'Source shows a missed required condition',
    }, request_id='compare')
    [comparison] = asyncio.run(consume_control_requests(rt))
    assert comparison['state'] == 'applied'
    payload = result_payload(comparison['result'])
    assert payload['adoption_eligible'] and len(rt.calls) == 4
    submit_contribution(directory, 'classifier_profiles', {
        'action': 'adopt', 'comparison_id': payload['comparison_id'],
    }, request_id='adopt')
    [adoption] = asyncio.run(consume_control_requests(rt))
    assert adoption['state'] == 'applied' and rt.config.feature_model == 'strong'


def test_expected_abstentions_remain_unknown_in_both_profiles(rt):
    rt.dataset.items.append({'_item_id': 'u', 'response_text': 'UNKNOWN',
                             'problem_id': 'u', 'is_correct': False})
    source = SOURCE.replace("    return llm_client", "    if item['response_text'] == 'UNKNOWN':\n"
                            "        library.abstain('Evidence unavailable')\n"
                            "    return llm_client")
    result = compare(source=source, item_ids=['p', 'n', 'u'], expected_abstentions=['u'])
    assert result['ok'] and result['adoption_eligible'], result
    assert result['baseline']['abstentions'] == result['fallback']['abstentions'] == 1
    assert result['fallback']['results'][2]['status'] == 'abstained'


@pytest.mark.usefixtures('rt')
def test_artifact_staleness_in_panel_blocks_adoption(monkeypatch):
    from autofa.session.diagnostic_receipts import read_diagnostic_receipt

    result = compare()

    def changed(runtime, receipt):
        if receipt == result['baseline']['diagnostic_receipt_id']:
            raise ValueError('Diagnostic artifact evidence changed or is unavailable')
        return read_diagnostic_receipt(runtime, receipt)

    monkeypatch.setattr('autofa.session.diagnostic_receipts.read_diagnostic_receipt', changed)
    refused = adopt(result)
    assert not refused['ok'] and 'artifact evidence changed' in refused['error']


def test_comparison_receipt_requires_original_control_expectations(rt):
    result = compare()
    path = rt.run.run_dir() / profiles.STATE_FILE
    state = json.loads(path.read_text())
    state['comparisons'][result['comparison_id']]['expected_outputs'] = {'p': False, 'n': True}
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='controls or results'):
        profiles.read_classifier_comparison(rt, result['comparison_id'])


def test_profile_order_and_run_scope_survive_multiple_adoptions(rt):
    rt.config.classifier_profiles.append({'name': 'stronger', 'feature_model': 'stronger'})
    skipped = compare(profile='stronger')
    assert not skipped['ok'] and not rt.calls
    first = compare()
    assert first['ok'] and first['adoption_eligible']
    assert adopt(first)['ok']
    assert profiles.profile_status(rt)['active_profile'] == 'strong'
    # A new source presents a new implementation challenge, while the run profile persists.
    rt.outputs['strong'], rt.outputs['stronger'] = ['no', 'no'], ['yes', 'no']
    rt.hub_llm = profiles._client(rt.config)
    second = compare(profile='stronger', source=SOURCE + '\n# New diagnostic implementation\n')
    assert second['ok'] and second['adoption_eligible'], second
    assert adopt(second)['ok'] and rt.config.feature_model == 'stronger'
    assert not profiles.profile_status(rt)['available_profiles']
    assert not compare()['ok']


def test_failed_tier_permits_next_trial_without_adoption_or_changed_controls(rt):
    rt.config.classifier_profiles.append({'name': 'stronger', 'feature_model': 'stronger'})
    rt.outputs['strong'], rt.outputs['stronger'] = ['no', 'no'], ['yes', 'no']
    first = compare()
    assert first['ok'] and first['outcome'] == 'no_demonstrated_improvement'
    assert rt.config.feature_model == 'base' and not rt.run.classifier_profile_settings
    # A later round/Resume retains the tested-tier cursor, not a fabricated adoption.
    rt.run.round += 1
    assert profiles.profile_status(rt)['tested_tier_cursors']
    denied = compare(profile='stronger', source=SOURCE + '\n# different hypothesis')
    assert not denied['ok'] and len(rt.calls) == 4
    second = compare(profile='stronger')
    assert second['ok'] and second['adoption_eligible'], second
    # Native diagnostics reuse the identical baseline panel rather than paying twice.
    assert rt.calls == ['base', 'base', 'strong', 'strong', 'stronger', 'stronger']
    assert compare(profile='stronger')['reused']
    assert adopt(second)['ok'] and rt.config.feature_model == 'stronger'
    assert first['comparison_id'] != second['comparison_id']


def test_passing_or_incomplete_intermediate_does_not_authorize_a_skip(rt):
    rt.config.classifier_profiles.append({'name': 'stronger', 'feature_model': 'stronger'})
    rt.outputs['base'] = ['yes', 'no']
    passing = compare()
    assert passing['ok'] and passing['outcome'] == 'baseline_controls_pass'
    assert not passing['adoption_eligible'] and passing['status'] == 'complete'
    assert not compare(profile='stronger')['ok']
    # A passing baseline never dispatches the unused fallback.
    assert rt.calls == ['base', 'base']


def test_pinned_policy_survives_changed_defaults_without_changing_judge(rt):
    from autofa.llm.endpoints import endpoint_for_config

    original = deepcopy(rt.config)
    profiles.initialize_classifier_policy(original, rt.run, new_run=True)
    changed = deepcopy(original)
    changed.classifier_profiles = []
    changed.feature_model = 'unapproved-new-default'
    changed.judge_model = 'unapproved-new-judge'
    restored = profiles.apply_run_classifier(changed, rt.run)
    assert restored.feature_model == original.feature_model
    assert restored.classifier_profiles == original.classifier_profiles
    assert endpoint_for_config(restored, 'judge') == endpoint_for_config(original, 'judge')
    assert not rt.calls


def test_legacy_run_without_snapshot_does_not_pin_current_global_routes(rt):
    assert not getattr(rt.run, 'classifier_policy', None)
    assert profiles.initialize_classifier_policy(rt.config, rt.run) == {}
    assert not getattr(rt.run, 'classifier_policy', None)


def test_new_policy_has_one_authoritative_copy(rt):
    profiles.initialize_classifier_policy(rt.config, rt.run, new_run=True)
    result = compare()
    assert result['ok']
    state = json.loads((rt.run.run_dir() / profiles.STATE_FILE).read_text())
    assert 'policy' not in state and state['policy_sha256']
    assert profiles.profile_status(rt)['comparisons_used'] == 1


def test_reordering_controls_cannot_create_fresh_trial_or_relabel_outcomes(rt):
    original = compare()
    calls = len(rt.calls)
    reordered = compare(item_ids=['n', 'p'])
    assert reordered['ok'] and reordered['reused']
    assert reordered['comparison_id'] == original['comparison_id']
    changed = compare(item_ids=['n', 'p'], expected_outputs={'n': True, 'p': False})
    assert not changed['ok'] and 'frozen' in changed['error']
    assert len(rt.calls) == calls
    assert profiles.profile_status(rt)['comparisons_used'] == 1


def test_feature_profile_preserves_inherited_judge_and_improvement_endpoints(rt):
    from autofa.llm.endpoints import endpoint_for_config, improvement_endpoint_for_config

    baseline = deepcopy(rt.config)
    before_judge = endpoint_for_config(baseline, 'judge')
    before_improvement = improvement_endpoint_for_config(baseline)
    # Exercise restoration independently of SDK dispatch: the adopted profile may use
    # another provider/API while other classifier roles keep their original route.
    rt.run.classifier_profile_settings = {
        'name': 'strong', 'comparison_id': 'a' * 64,
        'settings': {'feature_model': 'strong', 'feature_api': 'responses',
                     'feature_base_url': 'https://other.invalid/v1',
                     'feature_api_key_env': 'OTHER_API_KEY', 'feature_request_options': {}},
        'inherited_judge_settings': {
            'judge_base_url': before_judge.base_url, 'judge_api_key_env': before_judge.api_key_env,
            'judge_api': before_judge.api,
        },
    }
    restored = profiles.apply_run_classifier(baseline, rt.run)
    assert endpoint_for_config(restored).base_url == 'https://other.invalid/v1'
    assert endpoint_for_config(restored).api == 'responses'
    assert endpoint_for_config(restored, 'judge') == before_judge
    assert improvement_endpoint_for_config(restored) == before_improvement
    assert baseline.judge_api == baseline.judge_base_url == baseline.judge_api_key_env == ''


def test_native_candidate_compare_adopt_and_fresh_scoring(rt, monkeypatch):
    """Scripted tool sequence with real native controls, adoption and scoring.

    This tests execution integration; a scripted tool sequence is not evidence that a
    live discovery agent independently chooses the hypothesis or classifier change.
    """
    from autofa.session.family_discovery import initialize_family_discovery

    monkeypatch.setattr(tools_library, "get_runtime", lambda: rt)
    monkeypatch.setattr(rt, "persist_library",
                        lambda **kwargs: rt.library.save(rt.run.run_dir() / "library.json"))
    monkeypatch.setattr(rt, "ensure_hub_llm", lambda: profiles._client(rt.config))
    submitted = tools_library.submit_function(
        name="required_condition", source=SOURCE, kind="python", auto_evaluate=False,
        description="Required condition violated",
        mechanism="The response contradicts a required condition in the observed trace. "
                  "The uncorrected violation prevents delivery of the requested result.",
    )
    assert submitted["ok"], submitted
    assert not rt.library.get("required_condition").scores
    initialize_family_discovery(rt=rt)
    request = {"action": "compare", "profile": "strong", "name": "required_condition",
               "item_ids": ["p", "n"], "expected_outputs": {"p": True, "n": False},
               "diagnosis": "Source inspection finds a missed violation and a corrected boundary."}
    compared = tools_classifier.classifier_profiles(**request)
    assert compared["ok"] and compared["adoption_eligible"], compared
    server._record_tool_exposure("classifier_profiles", request, compared)
    adopted = adopt(compared)
    assert adopted["ok"] and adopted["adopted"], adopted
    assert not rt.library.get("required_condition").scores
    # The regular evaluator must execute the adopted model, not copy diagnostic scores.
    scored = tools_eval.evaluate_functions(["required_condition"], mode="capped")
    assert scored["ok"], scored
    scores = rt.library.get("required_condition").scores
    assert scores["evaluated"] and scores["evaluation_context"] == "strong"
    assert scores["recall"] == 1
    assert not scores.get("behavior_n_judged")
    assert not scores.get("behavior_purity")
    assert set(review_context(rt.run.run_dir(), "required_condition")["reviewed_item_ids"]) >= {
        "p", "n"}
    assert rt.calls[-2:] == ["strong", "strong"]
