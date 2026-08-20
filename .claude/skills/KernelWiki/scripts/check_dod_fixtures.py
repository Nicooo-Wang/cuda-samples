#!/usr/bin/env python3
"""Check Phase 3 Definition-of-Done fixtures (AC-7).

Reads data/phase3-dod-fixtures.yaml; for each entry, verifies that at least
one file matches the `required_assets` globs, the file has >= required_min_lines,
any `required_provenance_modes` restriction is met (by checking the file's
ancestor PROVENANCE.yaml's files[*].mode), and any `required_content_patterns`
regexes match across the matched files.

Exit codes:
  0 — all fixture entries pass
  1 — at least one fixture entry fails
  2 — invocation error (fixtures file missing)
"""

import argparse
import re
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_PATH = REPO_ROOT / "data" / "phase3-dod-fixtures.yaml"


def find_bundle_root_for(path):
    """Walk up from path until a PROVENANCE.yaml is found; return the directory."""
    p = path.parent
    while p != p.parent:
        if (p / "PROVENANCE.yaml").is_file():
            return p
        p = p.parent
    return None


def load_modes_for_file(file_path):
    """Return the per-file `mode` declared in the enclosing bundle's files[*]
    manifest, or None if not found."""
    root = find_bundle_root_for(file_path)
    if not root:
        return None
    prov_path = root / "PROVENANCE.yaml"
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        return None
    for entry in prov.get("files") or []:
        if isinstance(entry, dict) and entry.get("local_path") == rel:
            return entry.get("mode")
    return None


def load_bundle_mode_for_file(file_path):
    """Return the bundle-level `asset_mode` default for the enclosing bundle,
    or None if no PROVENANCE.yaml is found. This is distinct from the per-file
    mode: a bundle can carry `asset_mode: verbatim` (all files are verbatim
    upstream) or `asset_mode: derived` (all files are derived), and some DoD
    fixtures need to enforce the bundle-level contract rather than file-level."""
    root = find_bundle_root_for(file_path)
    if not root:
        return None
    prov_path = root / "PROVENANCE.yaml"
    try:
        prov = yaml.safe_load(prov_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    return prov.get("asset_mode")


def check_entry(entry):
    """Per-glob-independent fixture check.

    Every glob in `required_assets` must independently match at least one file.
    `required_min_lines` must be reached by at least one file matched by each
    glob (so the line-count requirement is enforced against the intended set,
    not just the union). `required_provenance_modes` and
    `required_content_patterns` are enforced against the overall matched set
    with the same per-glob coverage requirement.
    """
    errors = []
    question = entry.get("question", "<unnamed>")
    required_assets = entry.get("required_assets") or []
    required_min_lines = entry.get("required_min_lines", 100)
    required_modes = set(entry.get("required_provenance_modes") or [])
    required_bundle_modes = set(entry.get("required_bundle_asset_mode") or [])
    required_patterns = entry.get("required_content_patterns") or []

    # Each entry in required_assets can be either:
    #   - a string: a glob applied with the fixture-level required_min_lines
    #   - a dict {glob, min_lines?}: a glob with its own min_lines override
    # Resolve each glob independently so that a missing glob fails the fixture
    # even if other globs matched files.
    per_glob_matches = {}
    per_glob_min_lines = {}
    for a in required_assets:
        if isinstance(a, dict):
            glob = a.get("glob")
            min_lines = a.get("min_lines", required_min_lines)
        else:
            glob = a
            min_lines = required_min_lines
        if not glob:
            continue
        hits = [p for p in REPO_ROOT.glob(glob) if p.is_file()]
        per_glob_matches[glob] = hits
        per_glob_min_lines[glob] = min_lines
        if not hits:
            errors.append(f"{question!r}: required_assets glob {glob!r} matched no files")

    if errors:
        return errors

    # Per-glob line-count gate: at least one matched file per glob must reach
    # that glob's min_lines (either the fixture-level required_min_lines or
    # the glob's own override). Each asset group represents a distinct
    # required piece of evidence for the question.
    for glob, hits in per_glob_matches.items():
        floor = per_glob_min_lines[glob]
        long_enough = False
        for p in hits:
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                if len(lines) >= floor:
                    long_enough = True
                    break
            except OSError:
                pass
        if not long_enough:
            errors.append(
                f"{question!r}: no file matching {glob!r} reached "
                f"required_min_lines={floor}"
            )

    # Per-file mode restriction: every glob must have at least one file whose
    # file-level mode is in the allowed set. A fixture cannot pass by borrowing
    # mode evidence from a different glob's match.
    if required_modes:
        for glob, hits in per_glob_matches.items():
            ok = False
            for p in hits:
                m = load_modes_for_file(p)
                if m in required_modes:
                    ok = True
                    break
            if not ok:
                errors.append(
                    f"{question!r}: no file matching {glob!r} has per-file provenance "
                    f"mode in {sorted(required_modes)}"
                )

    # Bundle-level mode restriction: every glob must have at least one file
    # whose ENCLOSING BUNDLE declares asset_mode in the allowed set. This is
    # what lets a fixture demand e.g. "this question must be backed by at least
    # one verbatim-bundle asset" even when the bundle is mixed at the per-file
    # level.
    if required_bundle_modes:
        for glob, hits in per_glob_matches.items():
            ok = False
            for p in hits:
                bm = load_bundle_mode_for_file(p)
                if bm in required_bundle_modes:
                    ok = True
                    break
            if not ok:
                errors.append(
                    f"{question!r}: no file matching {glob!r} lives in a bundle with "
                    f"asset_mode in {sorted(required_bundle_modes)}"
                )

    # Content regexes must match across the full matched set. Aggregating is
    # still appropriate here because the content pattern is a statement about
    # the combined evidence, not per-glob.
    if required_patterns:
        aggregate = ""
        for hits in per_glob_matches.values():
            for p in hits:
                try:
                    aggregate += p.read_text(encoding="utf-8", errors="replace") + "\n"
                except OSError:
                    continue
        for pat in required_patterns:
            try:
                if not re.search(pat, aggregate, re.IGNORECASE):
                    errors.append(f"{question!r}: required_content_pattern {pat!r} not found")
            except re.error as e:
                errors.append(f"{question!r}: invalid regex {pat!r}: {e}")

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    args = parser.parse_args()

    if not FIXTURES_PATH.is_file():
        # R27: the module docstring classifies a missing fixture file
        # as an invocation error (the AC-7 gate is a required input).
        # `install_precommit_hook.sh` also guards against invoking this
        # script when the file is absent, so reaching this branch means
        # the caller bypassed that guard — fail loud (exit 2) instead
        # of reporting a clean skip that would silently remove the gate
        # in CI.
        print(
            f"ERROR: {FIXTURES_PATH.relative_to(REPO_ROOT)} not found; "
            f"the AC-7 DoD fixture gate requires this file. If the gate "
            f"is intentionally being retired, remove the invocation from "
            f"the pre-commit hook and delete this script instead of "
            f"deleting the fixtures file.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        data = yaml.safe_load(FIXTURES_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print(f"ERROR: could not parse {FIXTURES_PATH}: {e}", file=sys.stderr)
        sys.exit(2)

    entries = data.get("fixtures") or []
    if not entries:
        # R27: an empty `fixtures:` list collapses the AC-7 gate just
        # like a missing file. Reject as invocation error rather than
        # reporting "nothing to check" which looks like a clean pass.
        print(
            f"ERROR: {FIXTURES_PATH.relative_to(REPO_ROOT)} has no "
            f"`fixtures:` entries; AC-7 requires at least one fixture.",
            file=sys.stderr,
        )
        sys.exit(2)

    all_errors = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        all_errors.extend(check_entry(entry))

    print(f"Checked {len(entries)} DoD fixture entries.")
    if all_errors:
        for e in all_errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("All fixtures pass.")
    sys.exit(0)


if __name__ == "__main__":
    main()
