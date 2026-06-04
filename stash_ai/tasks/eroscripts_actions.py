"""EroScripts task adapters for the dispatch seam (#4, commit 4).

The four EroScripts modes (validate auth, search, download, status) are a
self-contained subsystem: each lives in its own ``eroscripts_*`` module with a
module-level ``run()`` that writes its own ``assets/eroscripts/{prefix}_{request_id}.json``
result file. These adapters wrap those functions so the modes go through the
generic seam (uniform error handling + self-describing construction) without
disturbing the subsystem. Each declares ``result_key`` for the dispatch-seam
guard even though the underlying module owns its result-file path (same
task-internal-writer pattern as ``RecommendationsTask``).

EroScripts is slated to be spun off into its own plugin later; keeping the
adapters thin here makes that extraction easy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..stash_client import StashClient
    from .dispatch import TaskContext


class EroscriptsValidateAuthTask:
    """Validate (or clear/re-check) the EroScripts session cookie."""

    result_key = "eroscripts_auth"  # module writes assets/eroscripts/auth_{request_id}.json

    def __init__(
        self,
        args: dict[str, Any],
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.args = args
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> EroscriptsValidateAuthTask:
        return cls(args=ctx.args, log_callback=ctx.log)

    def run(self) -> None:
        from . import eroscripts_auth as task_module

        task_module.run(self.args, self.log)


class EroscriptsSearchTask:
    """Search discuss.eroscripts.com for funscripts matching a Stash scene."""

    result_key = "eroscripts_search"  # module writes assets/eroscripts/search_{request_id}.json

    def __init__(
        self,
        stash: StashClient,
        args: dict[str, Any],
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.args = args
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> EroscriptsSearchTask:
        return cls(stash=ctx.stash, args=ctx.args, log_callback=ctx.log)

    def run(self) -> None:
        from . import eroscripts_search as task_module

        task_module.run(self.stash, self.args, self.log)


class EroscriptsDownloadTask:
    """List attachments for an eroscripts topic, or download one and persist."""

    result_key = "eroscripts_download"  # module writes assets/eroscripts/download_{request_id}.json

    def __init__(
        self,
        stash: StashClient,
        args: dict[str, Any],
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.args = args
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> EroscriptsDownloadTask:
        return cls(stash=ctx.stash, args=ctx.args, log_callback=ctx.log)

    def run(self) -> None:
        from . import eroscripts_download as task_module

        task_module.run(self.stash, self.args, self.log)


class EroscriptsStatusTask:
    """Report whether a scene has a matched funscript + sidecar."""

    result_key = "eroscripts_status"  # module writes assets/eroscripts/status_{request_id}.json

    def __init__(
        self,
        stash: StashClient,
        args: dict[str, Any],
        log_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.stash = stash
        self.args = args
        self.log = log_callback or (lambda msg, level: None)

    @classmethod
    def from_context(cls, ctx: TaskContext) -> EroscriptsStatusTask:
        return cls(stash=ctx.stash, args=ctx.args, log_callback=ctx.log)

    def run(self) -> None:
        from . import eroscripts_status as task_module

        task_module.run(self.stash, self.args, self.log)
