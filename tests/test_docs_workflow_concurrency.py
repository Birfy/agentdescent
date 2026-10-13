"""PR docs checks must not cancel other PRs or a Pages deployment."""

from pathlib import Path

import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "docs.yml"


def test_docs_workflow_scopes_concurrency_to_each_pull_request():
    config = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert config["concurrency"]["group"] == (
        "${{ github.event_name == 'pull_request' && "
        "format('pages-pr-{0}', github.event.pull_request.number) || 'pages' }}"
    )
    assert config["concurrency"]["cancel-in-progress"] is True


def test_docs_workflow_limits_pages_permissions_to_guarded_deploy_job():
    config = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    assert config["permissions"] == {"contents": "read"}
    assert config["jobs"]["build"]["permissions"] == {"contents": "read"}
    assert config["jobs"]["deploy"]["permissions"] == {
        "pages": "write",
        "id-token": "write",
    }
    assert config["jobs"]["deploy"]["if"] == (
        "github.event_name == 'workflow_dispatch' || "
        "github.ref == 'refs/heads/main'"
    )
