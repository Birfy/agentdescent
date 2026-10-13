"""Keep the install guide's dry-run claim in sync with the port catalogue."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _algorithm_rows(section: str) -> int:
    """Count named algorithm rows in one catalogue section."""
    return sum(line.startswith("| **") for line in section.splitlines())


def test_install_guide_names_the_current_number_of_dry_run_ports():
    catalogue = (ROOT / "docs" / "self-evolution-examples.md").read_text()
    install = (ROOT / "docs" / "install.md").read_text()

    headings = (
        "### The eight benchmark-faithful ports",
        "### The eleven microports and analogues",
        "### The twentieth: Genesis, at the standard seams",
    )
    sections = []
    for heading in headings:
        start = catalogue.index(heading)
        following = re.search(r"^### ", catalogue[start + len(heading):], re.MULTILINE)
        end = start + len(heading) + following.start() if following else len(catalogue)
        sections.append(catalogue[start:end])

    total = sum(_algorithm_rows(section) for section in sections)
    assert total == 20
    assert f"All {total} algorithm ports print" in install
    assert "All nineteen" not in install
