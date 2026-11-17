# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remote services stay off until configured: eval and library server settings, onboarding,
home identity status and login/logout, with local dashboard mode never using Cloud credentials."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from autofa.cli import _is_default_tty_home
from autofa.config import Config, RunConfig, config_path, load_config
from autofa.data import remote_library_auth as auth
from autofa.data import remote_library_onboarding as onboarding
from autofa.data.remote_api import DataApiClient, DataApiError
from autofa.data.remote_library_api import (
    RemoteCliAuthStart,
    RemoteCliToken,
    RemoteCliTokenPending,
    RemoteIdentity,
    RemoteLibraryError,
    client_from_config,
)
from autofa.data.remote_library_auth import login_to_remote_library, logout_from_remote_library
from autofa.data.remote_library_onboarding import (
    FIRST_PROMPT_MARKER,
    LOGIN_PROMPT,
    home_cloud_status,
)
from autofa.data.remote_library_publish import maybe_publish_run
from autofa.local_service import service_token
from autofa.ui import settings
from autofa.ui.library_settings import validate_library_url

from support.config import save_config


class FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(str(message))


class FakeStore:
    def __init__(self, token=None, events=None):
        self.token = token
        self.events = events if events is not None else []

    @property
    def get_calls(self):
        return sum(event[0] == "get" for event in self.events)

    def get_token(self, base_url):
        self.events.append(("get", base_url))
        return self.token

    def set_token(self, base_url, token):
        self.events.append(("save", base_url, token))
        self.token = token

    def delete_token(self, base_url):
        self.events.append(("delete", base_url))
        existed = self.token is not None
        self.token = None
        return existed


def forbidden(*args, **kwargs):
    pytest.fail("This mode must not use credentials, prompts, browser sign-in or the network")


def _identity(username="alice", identity_id=7):
    return RemoteIdentity(id=identity_id, username=username, name=username.title())


def _cloud():
    return Config(remote_library_base_url="https://example.test",
                  remote_library_token_env="GITLAB_TOKEN")


# The terminal settings form configures a remote AutoFA server.

RIGHT, SAVE, CANCEL, CLEAR = "\x1b[C", "\x13", "\x03", "\x01\x0b"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    cfg = Config(remote_library_mode="cloud", remote_library_base_url="https://reports.test",
                 remote_library_token_env="GITLAB_TOKEN")
    save_config(cfg)
    return cfg


def edit(cfg, keys, *, persistent=True):
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        return settings.configure_library_server(cfg, persistent=persistent)


def test_server_settings_save_url_and_dedicated_token_variable(configured):
    path = config_path()
    path.write_text(path.read_text() + '\n# User extension\ncustom_key = "keep"\n')
    edit(configured, RIGHT + "\thttp://analysis-server:8765/" + SAVE)
    saved = load_config()
    assert saved.remote_library_mode == "server"
    assert saved.remote_library_base_url == "http://analysis-server:8765"
    assert saved.remote_library_token_env == "AUTOFA_LIBRARY_TOKEN"
    assert saved.local_library_base_url == configured.local_library_base_url
    assert '# User extension\ncustom_key = "keep"' in path.read_text()


def test_per_run_server_selection_does_not_save_defaults(configured):
    before = config_path().read_bytes()
    draft = edit(configured, RIGHT + "\thttps://analysis-server" + SAVE, persistent=False)
    assert draft.remote_library_mode == "server"
    assert draft.remote_library_base_url == "https://analysis-server"
    assert config_path().read_bytes() == before


@pytest.mark.parametrize("keys", [
    RIGHT + CANCEL,
    RIGHT + "\thttp://server:8765?token=secret" + SAVE + CANCEL,
    RIGHT + "\thttp://server:8765\t" + CLEAR + "GITLAB_TOKEN" + SAVE + CANCEL,
], ids=["cancel", "token_in_url", "cloud_token_variable"])
def test_cancelled_or_invalid_server_form_leaves_saved_config_untouched(configured, keys):
    before = config_path().read_bytes()
    assert edit(configured, keys) == configured
    assert config_path().read_bytes() == before


def test_failed_server_settings_save_keeps_previous_selection(configured, monkeypatch):
    def fail(*args):
        raise OSError("Disk full")

    monkeypatch.setattr(settings, "_save_fields", fail)
    assert edit(configured, RIGHT + "\thttp://server:8765" + SAVE + CANCEL) == configured
    assert load_config().remote_library_base_url == configured.remote_library_base_url


@pytest.mark.parametrize("url", ["http://user:pass@server", "file:///report"])
def test_invalid_server_origins_are_rejected(url):
    with pytest.raises(ValueError):
        validate_library_url(url, mode="server")


def test_server_mode_is_validated_when_loading_config(configured):
    path = config_path()
    path.write_text(path.read_text().replace('remote_library_mode = "cloud"',
                                            'remote_library_mode = "typo"'))
    with pytest.raises(ValueError, match="remote_library_mode"):
        load_config()


def test_standalone_server_needs_no_authenticated_discovery_harness(tmp_path, monkeypatch):
    from autofa import cli
    from autofa.harness import settings as harness_settings
    from autofa_library_server import local

    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    def forbidden(**kwargs):
        pytest.fail("A server-only installation must not require an LLM harness")

    started = []
    monkeypatch.setattr(harness_settings, "detect_default_harness", forbidden)
    monkeypatch.setattr(local, "serve", lambda **kwargs: started.append(kwargs))
    cli.main(["serve", "--host", "0.0.0.0", "--port", "9876"])
    assert started == [{"host": "0.0.0.0", "port": 9876}]
    assert load_config().remote_library_mode == Config().remote_library_mode


# Public defaults stay local; explicit settings enable portable remote services.


@pytest.fixture
def profile(tmp_path, monkeypatch, installed_codex):
    monkeypatch.setenv("AUTOFA_CONFIG", str(tmp_path / "autofa.toml"))
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path / "profile"))
    return tmp_path


def test_fresh_install_and_migration_do_not_enable_services(profile):
    cfg = load_config()
    assert cfg.data_api_base_url == cfg.remote_library_base_url == ""
    # Loading an older local config adds disabled defaults without selecting a service.
    config_path().write_text('harness = "codex"\n')
    cfg = load_config()
    assert cfg.data_api_base_url == cfg.remote_library_base_url == ""


def test_existing_deployment_settings_survive_migration(profile):
    config_path().write_text(
        'data_api_base_url = "https://eval.example.org/api/"\n'
        'data_api_name = "Team evaluation"\n'
        'remote_library_base_url = "https://library.example.org"\n'
        'remote_library_token_env = "TEAM_TOKEN"\n'
        'hub_base_url = "https://llm.example.org/v1"\n'
        'hub_api_key_env = "TEAM_LLM_KEY"\n'
    )
    cfg = load_config()
    assert (cfg.data_api_base_url, cfg.data_api_name) == (
        "https://eval.example.org/api", "Team evaluation")
    assert (cfg.remote_library_base_url, cfg.remote_library_token_env) == (
        "https://library.example.org", "TEAM_TOKEN")
    assert (cfg.hub_base_url, cfg.hub_api_key_env) == (
        "https://llm.example.org/v1", "TEAM_LLM_KEY")


@pytest.mark.parametrize(("saved", "shown"), [
    ("EvalRDL Remote Server", "EvalRDL remote server"),  # earlier title-case default
    ("", "EvalRDL remote server"),
    ("Team evaluation", "Team evaluation"),
])
def test_eval_server_menu_name_uses_current_default(profile, saved, shown):
    config_path().write_text(f'data_api_name = "{saved}"\n')
    assert load_config().data_api_name == shown


def test_disabled_services_never_lookup_credentials_or_publish(profile, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("autofa.data.remote_library_onboarding.KeyringCredentialStore", forbidden)
    monkeypatch.setattr("autofa.data.remote_library_publish.client_from_config", forbidden)
    cfg = load_config()
    assert home_cloud_status(cfg, prompt_fn=forbidden) == ""
    run = RunConfig(run_id="local-only", parquet_paths=[], model_id="test")
    assert maybe_publish_run(run, cfg, trigger="session_end", console=SimpleNamespace(
        print=forbidden)) is None
    assert not (profile / "profile/runs/local-only/publication.json").exists()


@pytest.mark.parametrize("mode", ["cloud", "server"])
def test_explicit_remote_requests_explain_missing_configuration(profile, mode):
    cfg = Config(remote_library_mode=mode)
    with pytest.raises(RemoteLibraryError, match="Settings.*library server"):
        client_from_config(cfg)
    with pytest.raises(RemoteLibraryError, match="remote_library_base_url"):
        login_to_remote_library(cfg)
    if mode == "cloud":
        with pytest.raises(RemoteLibraryError, match="remote_library_base_url"):
            logout_from_remote_library(cfg)
    with pytest.raises(DataApiError, match="Settings > EvalRDL remote server"):
        DataApiClient("").list_runs()


def edit_eval(cfg, keys, *, persistent=True):
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        return settings.configure_eval_server(cfg, persistent=persistent)


def test_eval_settings_enable_customize_and_disable(profile):
    cfg = load_config()
    config_path().write_text(config_path().read_text() + '\n# Keep this\ncustom = "yes"\n')
    edit_eval(cfg, "https://eval.example.org/api/\t" + CLEAR + "Team evals" + SAVE)
    enabled = load_config()
    assert enabled.data_api_base_url == "https://eval.example.org/api"
    assert enabled.data_api_name == "Team evals"
    assert '# Keep this\ncustom = "yes"' in config_path().read_text()
    edit_eval(enabled, CLEAR + SAVE)
    assert load_config().data_api_base_url == ""


@pytest.mark.parametrize("address", ["https://user:secret@server", "https://server/?token=secret"])
def test_invalid_eval_settings_and_cancel_preserve_file(profile, address):
    cfg = load_config()
    before = config_path().read_bytes()
    assert edit_eval(cfg, address + SAVE + CANCEL) == cfg
    assert config_path().read_bytes() == before


def test_eval_per_run_override_does_not_save(profile):
    cfg = load_config()
    before = config_path().read_bytes()
    result = edit_eval(cfg, "https://eval.example.org" + SAVE, persistent=False)
    assert result.data_api_base_url == "https://eval.example.org"
    assert config_path().read_bytes() == before


def test_library_settings_can_disable_a_configured_server(profile):
    cfg = Config(remote_library_base_url="https://library.example.org",
                 remote_library_mode="server")
    save_config(cfg)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\t" + CLEAR + SAVE)
        settings.configure_library_server(cfg)
    assert load_config().remote_library_base_url == ""


@pytest.mark.parametrize("field", ["data_api_base_url", "remote_library_base_url"])
def test_toml_service_urls_are_validated_before_use(profile, field):
    config_path().write_text(f'{field} = "https://user:secret@server"\n')
    with pytest.raises(ValueError, match="credentials"):
        load_config()


# Explicit browser login/logout.


def test_server_login_validates_shared_token_without_browser_or_keychain(monkeypatch):
    identity = RemoteIdentity(id=1, username="local", name="Local researcher",
                              auth_method="local_token")
    calls = []

    class Client:
        def me(self):
            calls.append("me")
            return identity

    monkeypatch.setattr(auth, "KeyringCredentialStore", forbidden)
    console = FakeConsole()
    cfg = Config(remote_library_mode="server", remote_library_base_url="http://server:8765",
                 remote_library_token_env="AUTOFA_LIBRARY_TOKEN")
    assert login_to_remote_library(cfg, client=Client(), console=console,
                                   browser_open=forbidden) == identity
    assert calls == ["me"]
    assert logout_from_remote_library(cfg, console=console,
                                       authenticated_client_factory=forbidden) is False
    assert "AUTOFA_LIBRARY_TOKEN" in "\n".join(console.messages)


def test_server_login_rejects_anonymous_browser_identity():
    class Client:
        def me(self):
            return RemoteIdentity(id=1, username="local", name="Local researcher",
                                  auth_method="session")

    with pytest.raises(RemoteLibraryError, match="did not authenticate"):
        login_to_remote_library(Config(remote_library_mode="server"), client=Client())


def _start(expires_in=30, interval=1):
    flow_id = "f" * 43
    return RemoteCliAuthStart(flow_id=flow_id, login_url=f"https://example.test/auth/cli/{flow_id}",
                              expires_in=expires_in, interval=interval)


def _status(*args, **kwargs):
    return nullcontext()


SERVICE_TOKEN = "autofa_cli_service-secret"


def _issued(identity):
    return RemoteCliToken(token=SERVICE_TOKEN, expires_at="2026-09-03T12:00:00Z",
                          expires_in=3600, identity=identity)


def _cloud_login(client, store, **kwargs):
    kwargs.setdefault("console", FakeConsole())
    kwargs.setdefault("browser_open", lambda url: True)
    return login_to_remote_library(Config(remote_library_base_url="https://example.test"),
                                   credential_store=store, client=client,
                                   status_factory=_status, **kwargs)


def _cloud_logout(store, client_factory, **kwargs):
    kwargs.setdefault("console", FakeConsole())
    config = Config(remote_library_base_url="https://example.test", **kwargs.pop("config", {}))
    return logout_from_remote_library(config, credential_store=store,
                                      authenticated_client_factory=client_factory, **kwargs)


def test_login_browser_fallback_validates_before_keychain_save():
    events = []
    issued = _identity()

    class UnauthenticatedClient:
        def start_cli_login(self, challenge):
            events.append(("start", challenge))
            return _start()

        def poll_cli_token(self, flow_id, verifier):
            events.append(("poll", flow_id, verifier))
            return _issued(issued)

    class AuthenticatedClient:
        def me(self):
            events.append(("me",))
            return issued

        def revoke_cli_token(self):
            events.append(("revoke",))

    store = FakeStore(events=events)
    console = FakeConsole()
    identity = _cloud_login(
        UnauthenticatedClient(), store, console=console, browser_open=lambda url: False,
        authenticated_client_factory=lambda value: (
            events.append(("authenticated", value)) or AuthenticatedClient()
        ),
    )

    assert identity.username == "alice"
    event_names = [event[0] for event in events]
    assert event_names.index("me") < event_names.index("save")
    assert store.token == SERVICE_TOKEN
    output = "\n".join(console.messages)
    assert "https://example.test/auth/cli/" in output
    assert SERVICE_TOKEN not in output


class PendingClient:
    def __init__(self, start):
        self.start, self.interval = start, start.interval

    def start_cli_login(self, challenge):
        return self.start

    def poll_cli_token(self, flow_id, verifier):
        return RemoteCliTokenPending(interval=self.interval)


def test_login_timeout_never_saves_token():
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    store = FakeStore()
    with pytest.raises(RemoteLibraryError, match="expired"):
        _cloud_login(PendingClient(_start(expires_in=3, interval=2)), store, sleeper=sleep,
                     monotonic=lambda: clock.now)
    assert store.token is None
    assert not any(event[0] == "save" for event in store.events)


def test_login_identity_mismatch_revokes_and_does_not_save():
    events = []

    class UnauthenticatedClient:
        def start_cli_login(self, challenge):
            return _start()

        def poll_cli_token(self, flow_id, verifier):
            return _issued(_identity("alice", 7))

    class AuthenticatedClient:
        def me(self):
            events.append(("me",))
            return _identity("mallory", 8)

        def revoke_cli_token(self):
            events.append(("revoke",))

    store = FakeStore(events=events)
    with pytest.raises(RemoteLibraryError, match="identity did not match"):
        _cloud_login(UnauthenticatedClient(), store,
                     authenticated_client_factory=lambda value: AuthenticatedClient())
    assert ("revoke",) in events
    assert not any(event[0] == "save" for event in events)


def test_logout_revokes_cli_token_then_deletes_locally(monkeypatch):
    events = []
    store = FakeStore(SERVICE_TOKEN, events)
    console = FakeConsole()
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    client = SimpleNamespace(revoke_cli_token=lambda: events.append(("revoke",)))

    assert _cloud_logout(store, lambda token: client, console=console) is True
    assert [event[0] for event in events] == ["get", "revoke", "delete"]
    assert store.token is None


def test_logout_is_idempotent_and_does_not_claim_to_revoke_environment_pat(monkeypatch):
    monkeypatch.setenv("CUSTOM_PAT", "gitlab-environment-pat")
    console = FakeConsole()

    removed = _cloud_logout(
        FakeStore(), lambda token: pytest.fail("environment PAT must not be revoked by AutoFA"),
        console=console, config={"remote_library_token_env": "CUSTOM_PAT"},
    )

    assert removed is False
    assert "gitlab-environment-pat" not in "\n".join(console.messages)


def test_logout_deletes_locally_when_remote_revoke_fails():
    store = FakeStore(SERVICE_TOKEN)
    console = FakeConsole()

    def unavailable():
        raise RemoteLibraryError("service unavailable")

    client = SimpleNamespace(revoke_cli_token=unavailable)
    assert _cloud_logout(store, lambda token: client, console=console)
    assert store.token is None


@pytest.mark.parametrize("command", ["login", "logout"])
def test_cli_dispatches_explicit_auth_commands(monkeypatch, command):
    from autofa import cli

    called = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "ensure_default_config", lambda: "/tmp/config.toml")
    monkeypatch.setattr(cli, "_login", lambda: called.append("login"))
    monkeypatch.setattr(cli, "_logout", lambda: called.append("logout"))

    cli.main([command])

    assert called == [command]


# First-run cloud onboarding and home identity status.


@pytest.mark.parametrize("connected", [True, False])
def test_server_onboarding_uses_configured_token_without_cloud_prompt(
    tmp_path, monkeypatch, connected,
):
    class Client:
        base_url = "http://analysis-server:8765"

        def me(self):
            if not connected:
                raise RemoteLibraryError("secret-token must not be rendered", status=401)
            return RemoteIdentity(id=1, username="local", name="Local researcher",
                                  auth_method="local_token")

    monkeypatch.setattr(onboarding, "KeyringCredentialStore", forbidden)
    monkeypatch.setattr(onboarding, "client_from_config", lambda *args, **kwargs: Client())
    status = onboarding.home_cloud_status(
        Config(remote_library_mode="server", remote_library_base_url=Client.base_url),
        marker_root=tmp_path, prompt_fn=forbidden, login_fn=forbidden,
    )
    assert ("ready" if connected else "invalid token") in status
    assert "secret-token" not in status
    assert list(tmp_path.iterdir()) == []


def _home(root, *, console=None, **kwargs):
    return home_cloud_status(_cloud(), marker_root=root, console=console or FakeConsole(),
                             **kwargs)


def _no_prompt(*args, **kwargs):
    pytest.fail("prompt should not run")


def test_first_run_yes_logs_in_and_writes_atomic_marker(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    prompts = []
    logins = []

    def prompt(message, *, default_yes):
        prompts.append((message, default_yes))
        return True

    def login(config, *, credential_store, console):
        logins.append((credential_store, console))
        return _identity()

    status = _home(tmp_path, credential_store=FakeStore(), prompt_fn=prompt, login_fn=login)

    assert status == "AutoFA Cloud Library: logged in as @alice"
    assert prompts == [(LOGIN_PROMPT, True)]
    assert len(logins) == 1
    marker = tmp_path / FIRST_PROMPT_MARKER
    assert json.loads(marker.read_text(encoding="utf-8")) == {"version": 1, "prompted": True}
    assert list(tmp_path.glob(".*.partial")) == []


def test_first_run_decline_is_only_asked_once(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    store = FakeStore()
    prompts = []

    def no_login(*args, **kwargs):
        pytest.fail("login should not run")

    first = _home(tmp_path, credential_store=store, login_fn=no_login,
                  prompt_fn=lambda message, default_yes: prompts.append(message) or False)
    second = _home(tmp_path, credential_store=store, login_fn=no_login,
                   prompt_fn=lambda *args, **kwargs: pytest.fail("prompt repeated"))

    assert prompts == [LOGIN_PROMPT]
    assert "not logged in" in first
    assert second == first


@pytest.mark.parametrize(("environment_token", "keychain_token", "expected_source"), [
    ("gitlab-pat", None, "env"), (None, "autofa_cli_keychain", "keychain"),
])
def test_existing_credential_skips_prompt_and_displays_identity(
    tmp_path: Path, monkeypatch, environment_token, keychain_token, expected_source,
):
    if environment_token:
        monkeypatch.setenv("GITLAB_TOKEN", environment_token)
    else:
        monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    store = FakeStore(keychain_token)
    clients = []

    def client_factory(base_url, token, source):
        clients.append((base_url, token, source))
        return SimpleNamespace(me=_identity)

    status = _home(tmp_path, credential_store=store, prompt_fn=_no_prompt,
                   client_factory=client_factory)

    assert status == "AutoFA Cloud Library: logged in as @alice"
    assert len(clients) == 1
    assert clients[0][2] == expected_source
    assert store.get_calls == (0 if environment_token else 1)


@pytest.mark.parametrize(("error", "expected"), [
    (RemoteLibraryError("invalid token with secret", status=401), "credential expired/invalid"),
    (RemoteLibraryError("network leaked secret"), "unavailable"),
])
def test_invalid_or_network_identity_status_is_compact_and_nonfatal(
    tmp_path: Path, monkeypatch, error, expected,
):
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)

    class Client:
        def me(self):
            raise error

    status = _home(tmp_path, credential_store=FakeStore("autofa_cli_keychain"),
                   prompt_fn=_no_prompt, client_factory=lambda *args: Client())

    assert expected in status
    assert "secret" not in status


def test_login_failure_or_interrupt_keeps_local_home_usable(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    console = FakeConsole()
    status = _home(
        tmp_path, credential_store=FakeStore(), console=console,
        prompt_fn=lambda *args, **kwargs: True,
        login_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            RemoteLibraryError("failure containing autofa_cli_secret")
        ),
    )

    assert "unavailable" in status
    assert "autofa_cli_secret" not in "\n".join(console.messages)

    interrupted_root = tmp_path / "interrupted"
    interrupted_console = FakeConsole()
    interrupted = _home(
        interrupted_root, credential_store=FakeStore(), console=interrupted_console,
        prompt_fn=lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    assert "not logged in" in interrupted
    assert (interrupted_root / FIRST_PROMPT_MARKER).is_file()


def test_default_home_tty_gate_suppresses_once_nontty_and_subcommands():
    tty = SimpleNamespace(isatty=lambda: True)
    not_tty = SimpleNamespace(isatty=lambda: False)
    for command, once, stdin, expected in [
        ("home", False, tty, True), (None, False, tty, True), ("home", True, tty, False),
        ("home", False, not_tty, False), ("publish", False, tty, False),
    ]:
        args = SimpleNamespace(command=command, once=once)
        assert _is_default_tty_home(args, stdin=stdin, stdout=tty) is expected


# Local dashboard mode never performs Cloud sign-in or credential operations.


@pytest.fixture
def local_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUTOFA_HOME", str(tmp_path))
    monkeypatch.setenv("GITLAB_TOKEN", "cloud-secret-must-not-be-used")
    monkeypatch.setattr(auth, "KeyringCredentialStore", forbidden)
    monkeypatch.setattr(onboarding, "KeyringCredentialStore", forbidden)
    token = service_token(create=True)
    return tmp_path, token


def test_local_home_uses_profile_token_without_cloud_onboarding(local_profile, monkeypatch):
    url = "http://localhost:9876/"
    root, token = local_profile
    calls = []

    class Client:
        def __init__(self, base_url, *, token, credential_source, timeout):
            calls.append((base_url, token, credential_source, timeout))

        def me(self):
            return RemoteIdentity(id=1, username="local", name="Local researcher")

    monkeypatch.setattr(onboarding, "RemoteLibraryClient", Client)
    status = onboarding.home_cloud_status(
        Config(remote_library_base_url=url),
        marker_root=root,
        prompt_fn=forbidden,
        login_fn=forbidden,
    )

    assert "ready" in status
    assert calls == [(url.rstrip("/"), token, "local", onboarding.HOME_IDENTITY_TIMEOUT)]
    assert not (root / onboarding.FIRST_PROMPT_MARKER).exists()


@pytest.mark.parametrize("missing_token", [False, True])
def test_local_unavailable_status_does_not_offer_cloud_login(
    local_profile, missing_token,
):
    root, token = local_profile
    if missing_token:
        (root / "server" / "token").unlink()

    class Client:
        def me(self):
            raise RemoteLibraryError(f"Authentication failed with {token}", status=401)

    status = onboarding.home_cloud_status(
        Config(remote_library_base_url="http://127.0.0.1:8765"),
        marker_root=root,
        prompt_fn=forbidden,
        login_fn=forbidden,
        client_factory=lambda *args: Client(),
    )

    assert "unavailable" in status
    assert token not in status
    assert "autofa login" not in status
    assert not (root / onboarding.FIRST_PROMPT_MARKER).exists()


def test_local_login_logout_leave_profile_token_and_cloud_credentials_untouched(local_profile):
    root, token = local_profile
    console = FakeConsole()
    cfg = Config(remote_library_base_url="http://127.0.0.1:8765")

    identity = auth.login_to_remote_library(cfg, console=console, browser_open=forbidden)
    removed = auth.logout_from_remote_library(
        cfg, console=console, authenticated_client_factory=forbidden,
    )

    assert identity.username == "local"
    assert removed is False
    assert service_token() == token
    assert list(root.iterdir()) == [root / "server"]
    output = "\n".join(console.messages)
    assert "cloud-secret-must-not-be-used" not in output and token not in output


def test_invalid_loopback_configuration_never_accesses_cloud_credentials_or_network(
    local_profile, monkeypatch,
):
    root, token = local_profile
    cfg = Config(remote_library_base_url="http://127.0.0.1:8765/path")
    monkeypatch.setattr(onboarding, "find_credential", forbidden)
    monkeypatch.setattr(onboarding, "RemoteLibraryClient", forbidden)
    monkeypatch.setattr(auth, "RemoteLibraryClient", forbidden)
    status = onboarding.home_cloud_status(
        cfg, marker_root=root, prompt_fn=forbidden, login_fn=forbidden,
        client_factory=forbidden,
    )
    assert "invalid configuration" in status
    assert not (root / onboarding.FIRST_PROMPT_MARKER).exists()
    with pytest.raises(RemoteLibraryError, match="Invalid local library configuration"):
        auth.login_to_remote_library(cfg, browser_open=forbidden)
    with pytest.raises(RemoteLibraryError, match="Invalid local library configuration"):
        auth.logout_from_remote_library(cfg, authenticated_client_factory=forbidden)
    assert service_token() == token
