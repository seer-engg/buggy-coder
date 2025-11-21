from __future__ import annotations

from langgraph.graph import START, END, StateGraph

try:  # pragma: no cover - compatibility for script-style imports
    from .asana_client import AsanaAPIError, AsanaClient
    from .asana_utils import extract_asana_task_gids, normalise_gids
    from .config import ConfigurationError, load_config
    from .github_client import GitHubAPIError, GitHubClient
    from .state import PRSyncState
except ImportError:  # pragma: no cover - fallback when package context is absent
    from asana_client import AsanaAPIError, AsanaClient
    from asana_utils import extract_asana_task_gids, normalise_gids
    from config import ConfigurationError, load_config
    from github_client import GitHubAPIError, GitHubClient
    from state import PRSyncState

SYSTEM_PROMPT = (
    "You are an automation agent responsible for syncing merged GitHub pull requests to Asana. "
    "Every event provides the GitHub repository name and pull request number. Inspect the pull "
    "request, confirm it is merged, gather any Asana task URLs from the body, and then mark those "
    "tasks complete while leaving a merge comment that references the pull request URL. "
    "Surface clear, actionable errors when configuration or network issues prevent progress."
)


# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------

def _with_appended_list(state: PRSyncState, key: str, value: str) -> PRSyncState:
    items = list(state.get(key, []))
    items.append(value)
    state[key] = items
    return state


def _add_log(state: PRSyncState, message: str) -> PRSyncState:
    return _with_appended_list(state, "logs", message)


def _add_error(state: PRSyncState, message: str) -> PRSyncState:
    _with_appended_list(state, "errors", message)
    return _add_log(state, f"ERROR: {message}")


# --------------------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------------------

def validate_input(state: PRSyncState) -> PRSyncState:
    new_state: PRSyncState = dict(state)

    repository_value = state.get("repository")
    repository_raw = str(repository_value).strip() if repository_value is not None else ""
    if not repository_raw:
        _add_error(new_state, "Missing 'repository' in event payload.")
    else:
        new_state["repository"] = repository_raw
        _add_log(new_state, f"Received repository input: {repository_raw}")

    pr_number_raw = state.get("pull_request_number")
    if pr_number_raw is None:
        _add_error(new_state, "Missing 'pull_request_number' in event payload.")
    else:
        try:
            pr_number_int = int(pr_number_raw)
        except (TypeError, ValueError):
            _add_error(
                new_state,
                "Pull request number must be an integer-compatible value, "
                f"got {pr_number_raw!r}.",
            )
        else:
            new_state["pull_request_number"] = pr_number_int
            _add_log(new_state, f"Targeting pull request #{pr_number_int}.")

    return new_state


def load_pull_request(state: PRSyncState) -> PRSyncState:
    new_state: PRSyncState = dict(state)

    try:
        config = load_config()
    except ConfigurationError as exc:
        return _add_error(new_state, f"Configuration error: {exc}")

    repository = new_state.get("repository")
    pull_number = new_state.get("pull_request_number")
    if repository is None or pull_number is None:
        return new_state

    client = GitHubClient(config.github)
    _add_log(new_state, f"Fetching pull request data for {repository}#{pull_number}.")
    try:
        pull_request = client.fetch_pull_request(repository, pull_number)
    except GitHubAPIError as exc:
        return _add_error(new_state, f"GitHub API error while retrieving pull request: {exc}")

    new_state["pull_request"] = pull_request.to_dict()

    if not pull_request.merged:
        return _add_error(
            new_state,
            f"Pull request {repository}#{pull_number} is not merged; skipping Asana sync.",
        )

    body = pull_request.body or ""
    task_gids = normalise_gids(extract_asana_task_gids(body))
    new_state["asana_task_gids"] = task_gids
    if task_gids:
        _add_log(
            new_state,
            f"Identified {len(task_gids)} Asana task reference(s) in pull request body.",
        )
    else:
        _add_log(new_state, "No Asana task references found in pull request body.")

    return new_state


def sync_asana_tasks(state: PRSyncState) -> PRSyncState:
    new_state: PRSyncState = dict(state)

    task_gids = list(new_state.get("asana_task_gids", []))
    if not task_gids:
        _add_log(new_state, "Skipping Asana synchronisation because no tasks were detected.")
        return new_state

    try:
        config = load_config()
    except ConfigurationError as exc:
        return _add_error(new_state, f"Configuration error: {exc}")

    client = AsanaClient(config.asana)
    pull_request = new_state.get("pull_request", {})
    pr_url = pull_request.get("html_url")
    repository = new_state.get("repository")
    pull_number = new_state.get("pull_request_number")
    pr_reference = (
        f"{repository}#{pull_number}" if repository and pull_number is not None else "Pull request"
    )

    for gid in task_gids:
        try:
            client.mark_task_complete(gid)
            _with_appended_list(new_state, "actions", f"Marked Asana task {gid} complete.")
            if pr_url:
                comment = f"{pr_reference} merged: {pr_url}"
                client.add_comment_to_task(gid, comment)
                _with_appended_list(
                    new_state,
                    "actions",
                    f"Posted merge comment to Asana task {gid}.",
                )
        except AsanaAPIError as exc:
            _add_error(new_state, f"Failed to update Asana task {gid}: {exc}")

    if not new_state.get("errors"):
        _add_log(
            new_state,
            f"Completed Asana updates for {len(task_gids)} task(s).",
        )
    return new_state


def summarise(state: PRSyncState) -> PRSyncState:
    new_state: PRSyncState = dict(state)
    repository = new_state.get("repository")
    pull_number = new_state.get("pull_request_number")
    prefix = (
        f"PR {repository}#{pull_number}"
        if repository and pull_number is not None
        else "Pull request"
    )

    errors = new_state.get("errors", [])
    if errors:
        summary = f"{prefix} sync failed: {errors[0]}"
        if len(errors) > 1:
            summary += f" (+{len(errors) - 1} more issue(s))"
    else:
        task_count = len(new_state.get("asana_task_gids", []))
        if task_count:
            summary = f"Synchronized {task_count} Asana task(s) for {prefix}."
        else:
            summary = f"No Asana task references detected for {prefix}."
    new_state["summary"] = summary

    _add_log(new_state, f"Summary generated: {summary}")
    return new_state


# --------------------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------------------

def _route_on_errors(state: PRSyncState) -> str:
    return "halt" if state.get("errors") else "continue"


graph = StateGraph(PRSyncState)

graph.add_node("validate_input", validate_input)

graph.add_node("load_pull_request", load_pull_request)

graph.add_node("sync_asana_tasks", sync_asana_tasks)

graph.add_node("summarise", summarise)

graph.add_edge(START, "validate_input")

graph.add_conditional_edges(
    "validate_input",
    _route_on_errors,
    {"halt": "summarise", "continue": "load_pull_request"},
)

graph.add_conditional_edges(
    "load_pull_request",
    _route_on_errors,
    {"halt": "summarise", "continue": "sync_asana_tasks"},
)

graph.add_conditional_edges(
    "sync_asana_tasks",
    _route_on_errors,
    {"halt": "summarise", "continue": "summarise"},
)

graph.add_edge("summarise", END)

app = graph.compile()
