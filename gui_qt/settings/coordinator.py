"""Navigation, lazy materialization, and config orchestration for Settings pages.

The coordinator is deliberately Qt-free: the host injects a page builder and a
page-show callback. ``MainWindow`` remains the owner of the single save
transaction and the global shell route during the migration; the coordinator
delegates to those host callbacks and never writes configuration itself.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .page_contract import SettingsIssue, SettingsPageActions
from .registry import SettingsPageRegistry, SettingsPageSpec


class SettingsCoordinator:
    """Own Settings page identity, lazy build, and page lifecycle boundaries."""

    def __init__(
        self,
        registry: SettingsPageRegistry,
        *,
        builder: Callable[[SettingsPageSpec], object | None],
        show_page: Callable[[str], None] | None = None,
        on_page_built: Callable[[SettingsPageSpec, object], None] | None = None,
        actions: SettingsPageActions | None = None,
    ) -> None:
        self._registry = registry
        self._builder = builder
        self._show_page = show_page
        self._on_page_built = on_page_built
        self._actions = actions
        self._pages: dict[str, object] = {}
        self._loaded: set[str] = set()
        self._active_key: str | None = None

    @property
    def registry(self) -> SettingsPageRegistry:
        return self._registry

    @property
    def actions(self) -> SettingsPageActions | None:
        return self._actions

    @property
    def active_key(self) -> str | None:
        return self._active_key

    def built_keys(self) -> tuple[str, ...]:
        return tuple(self._pages)

    def loaded_keys(self) -> frozenset[str]:
        return frozenset(self._loaded)

    def is_built(self, key: str) -> bool:
        return key in self._pages

    def is_loaded(self, key: str) -> bool:
        return key in self._loaded

    def mark_loaded(self, key: str) -> None:
        """Record that the host loaded this page through the legacy path."""

        self._registry.get(key)
        self._loaded.add(key)

    def mark_unloaded(self, key: str) -> None:
        self._loaded.discard(key)

    def page(self, key: str) -> object | None:
        return self._pages.get(key)

    def set_actions(self, actions: SettingsPageActions | None) -> None:
        self._actions = actions
        if actions is None:
            return
        for page in self._pages.values():
            setter = getattr(page, "set_action_callbacks", None)
            if callable(setter):
                setter(actions)

    def ensure_page(self, key: str, *, build: bool = True) -> object | None:
        """Build one page on first use; never build sibling pages."""

        if key in self._pages:
            return self._pages[key]
        spec = self._registry.get(key)
        if not build:
            return None
        page = self._builder(spec)
        if page is None:
            return None
        self._validate_page_contract(spec, page)
        if self._actions is not None:
            setter = getattr(page, "set_action_callbacks", None)
            if callable(setter):
                setter(self._actions)
        self._pages[key] = page
        self._loaded.discard(key)
        if self._on_page_built is not None:
            self._on_page_built(spec, page)
        return page

    def activate(self, key: str, *, build: bool = True) -> object | None:
        """Make one page current, building only that page when needed."""

        page = self.ensure_page(key, build=build)
        if page is None:
            return None
        self._active_key = key
        if self._show_page is not None:
            self._show_page(key)
        return page

    def load(
        self,
        snapshot: Mapping[str, object],
        *,
        pages: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> None:
        """Call ``load`` on already-built pages; never materialize pages here."""

        for key in self._target_keys(pages):
            page = self._pages[key]
            page_snapshot = self._snapshot_for_page(key, snapshot)
            loader = getattr(page, "load", None)
            if callable(loader):
                loader(page_snapshot)
                self._loaded.add(key)

    def collect(self) -> dict[str, object]:
        """Merge page-owned values and reject unknown or duplicate keys."""

        values: dict[str, object] = {}
        owners: dict[str, str] = {}
        for key, page in self._pages.items():
            collector = getattr(page, "collect", None)
            if not callable(collector):
                continue
            page_values = collector()
            if not isinstance(page_values, Mapping):
                raise TypeError(
                    f"settings page {key!r} collect() must return a mapping"
                )
            allowed = self._registry.config_keys_for(key)
            for config_key, value in page_values.items():
                if config_key not in allowed:
                    raise ValueError(
                        f"settings page {key!r} returned unowned key "
                        f"{config_key!r}"
                    )
                previous_owner = owners.get(config_key)
                if previous_owner is not None:
                    raise ValueError(
                        f"settings key {config_key!r} collected from both "
                        f"{previous_owner!r} and {key!r}"
                    )
                owners[config_key] = key
                values[config_key] = value
        return values

    def validate(self) -> list[SettingsIssue]:
        """Collect page-local issues; shared validation remains host-owned."""

        issues: list[SettingsIssue] = []
        for key, page in self._pages.items():
            validator = getattr(page, "validate", None)
            if not callable(validator):
                continue
            page_issues = validator()
            for issue in page_issues:
                if issue.page_key != key:
                    raise ValueError(
                        f"settings page {key!r} returned issue for page "
                        f"{issue.page_key!r}"
                    )
                issues.append(issue)
        return issues

    def reset(
        self,
        *,
        pages: Sequence[str] | set[str] | frozenset[str] | None = None,
    ) -> None:
        """Discard unsaved edits on built pages; never write configuration."""

        for key in self._target_keys(pages):
            resetter = getattr(self._pages[key], "reset", None)
            if callable(resetter):
                resetter()
            self._loaded.discard(key)

    def focus_issue(self, issue: SettingsIssue) -> bool:
        """Activate the issue page and let it focus/decorate the field."""

        page = self.ensure_page(issue.page_key)
        if page is None:
            return False
        self._active_key = issue.page_key
        if self._show_page is not None:
            self._show_page(issue.page_key)
        focus = getattr(page, "focus_issue", None)
        if not callable(focus):
            return False
        return bool(focus(issue))

    def set_task_running(self, running: bool) -> None:
        for page in self._pages.values():
            setter = getattr(page, "set_task_running", None)
            if callable(setter):
                setter(running)

    def has_unsaved_changes(self, baseline: Mapping[str, object]) -> bool:
        """Compare collected page-owned values against a host baseline."""

        for key, value in self.collect().items():
            if baseline.get(key) != value:
                return True
        return False

    def _target_keys(
        self,
        pages: Sequence[str] | set[str] | frozenset[str] | None,
    ) -> tuple[str, ...]:
        if pages is None:
            return tuple(self._pages)
        return tuple(key for key in self._pages if key in set(pages))

    def _snapshot_for_page(
        self,
        key: str,
        snapshot: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            config_key: snapshot[config_key]
            for config_key in self._registry.config_keys_for(key)
            if config_key in snapshot
        }

    def _validate_page_contract(
        self,
        spec: SettingsPageSpec,
        page: object,
    ) -> None:
        page_key = getattr(page, "page_key", None)
        if page_key != spec.key:
            raise ValueError(
                f"settings page object for {spec.key!r} reports "
                f"page_key={page_key!r}"
            )
        nav_label = getattr(page, "nav_label", None)
        if nav_label != spec.nav_label:
            raise ValueError(
                f"settings page {spec.key!r} reports nav_label={nav_label!r}; "
                f"expected {spec.nav_label!r}"
            )
        config_keys = getattr(page, "config_keys", None)
        if frozenset(config_keys or ()) != spec.config_keys:
            raise ValueError(
                f"settings page {spec.key!r} config_keys do not match registry"
            )
        immediate_ids = getattr(page, "immediate_action_ids", None)
        if frozenset(immediate_ids or ()) != spec.immediate_action_ids:
            raise ValueError(
                f"settings page {spec.key!r} immediate_action_ids do not "
                "match registry"
            )
        for method_name in (
            "load",
            "collect",
            "validate",
            "reset",
            "focus_issue",
            "set_task_running",
        ):
            if not callable(getattr(page, method_name, None)):
                raise TypeError(
                    f"settings page {spec.key!r} is missing {method_name}()"
                )


def snapshot_subset(
    snapshot: Mapping[str, object],
    config_keys: frozenset[str],
) -> dict[str, object]:
    """Return only the flat keys owned by a page (used by adapters/tests)."""

    return {key: snapshot[key] for key in config_keys if key in snapshot}
