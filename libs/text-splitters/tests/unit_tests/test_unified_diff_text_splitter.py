"""Unit tests for `UnifiedDiffTextSplitter`."""

from __future__ import annotations

import re
from itertools import pairwise

import pytest
from langchain_core.documents import Document

from langchain_text_splitters import UnifiedDiffTextSplitter

_CHUNK_HEADER_RE = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(?P<suffix>.*)$")


def _parse_chunk(chunk: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Split a produced chunk back into file header and `(header, body)` hunks."""
    lines = chunk.split("\n")
    old_index = next(i for i, line in enumerate(lines) if line.startswith("--- "))
    new_index = next(i for i, line in enumerate(lines) if line.startswith("+++ "))
    header = lines[old_index : new_index + 1]
    hunks: list[tuple[str, list[str]]] = []
    index = new_index + 1
    while index < len(lines):
        hunk_header = lines[index]
        index += 1
        body: list[str] = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            body.append(lines[index])
            index += 1
        hunks.append((hunk_header, body))
    return header, hunks


def _assert_valid_piece_headers(
    pieces: list[tuple[str, list[str]]],
    *,
    old_start: int,
    new_start: int,
) -> None:
    """Check rewritten headers against an independent walk of piece bodies."""
    cursor_old = old_start
    cursor_new = new_start
    started_old = old_start > 0
    started_new = new_start > 0
    for hunk_header, body in pieces:
        match = _CHUNK_HEADER_RE.match(hunk_header)
        assert match is not None
        declared_old_start = int(match.group(1))
        declared_old_count = int(match.group(2))
        declared_new_start = int(match.group(3))
        declared_new_count = int(match.group(4))
        old_count = sum(
            1
            for line in body
            if line.startswith((" ", "-")) and not line.startswith("\\")
        )
        new_count = sum(
            1
            for line in body
            if line.startswith((" ", "+")) and not line.startswith("\\")
        )
        assert old_count == declared_old_count
        assert new_count == declared_new_count
        if old_count > 0:
            assert declared_old_start == cursor_old
        else:
            expected = cursor_old - 1 if started_old else 0
            assert declared_old_start == expected
        if new_count > 0:
            assert declared_new_start == cursor_new
        else:
            expected = cursor_new - 1 if started_new else 0
            assert declared_new_start == expected
        cursor_old += old_count
        cursor_new += new_count
        started_old = True
        started_new = True


def test_empty_input_returns_empty_list() -> None:
    splitter = UnifiedDiffTextSplitter()
    assert splitter.split_text("") == []


def test_single_small_hunk_is_one_verbatim_chunk() -> None:
    diff_text = (
        "--- a/f\n"
        "+++ b/f\n"
        "@@ -1,3 +1,3 @@ def foo():\n"
        " line1\n"
        "-line2\n"
        "+line2 changed\n"
        " line3\n"
    )
    splitter = UnifiedDiffTextSplitter(chunk_size=1000)
    chunks = splitter.split_text(diff_text)
    assert chunks == [diff_text.rstrip("\n")]


def test_every_chunk_repeats_file_header_and_contains_single_file() -> None:
    diff_text = (
        "--- a/first\n"
        "+++ b/first\n"
        "@@ -1,2 +1,2 @@\n"
        " a\n"
        "-b\n"
        "+B\n"
        "--- a/second\n"
        "+++ b/second\n"
        "@@ -1,2 +1,2 @@\n"
        " c\n"
        "-d\n"
        "+D\n"
    )
    splitter = UnifiedDiffTextSplitter(chunk_size=45)
    chunks = splitter.split_text(diff_text)
    assert len(chunks) >= 2
    for chunk in chunks:
        old_headers = [line for line in chunk.split("\n") if line.startswith("--- ")]
        new_headers = [line for line in chunk.split("\n") if line.startswith("+++ ")]
        assert len(old_headers) == 1
        assert len(new_headers) == 1
        assert chunk.startswith("--- a/")
    assert any("first" in chunk for chunk in chunks)
    assert any("second" in chunk for chunk in chunks)


def test_complete_hunks_merge_in_order_while_they_fit() -> None:
    hunk1_header = "@@ -1,2 +1,2 @@\n c1\n-x\n"
    hunk1 = hunk1_header + "+y\n"
    hunk2 = "@@ -10,2 +10,2 @@\n c2\n-z\n+w\n"
    diff_text = f"--- a\n+++ b\n{hunk1}{hunk2}"

    together = UnifiedDiffTextSplitter(chunk_size=1000).split_text(diff_text)
    assert len(together) == 1
    assert together[0] == diff_text.rstrip("\n")

    separated = UnifiedDiffTextSplitter(chunk_size=48).split_text(diff_text)
    assert len(separated) == 2
    assert hunk1.rstrip("\n") in separated[0]
    assert hunk2.rstrip("\n") in separated[1]
    # Both chunks repeat the file header and keep original hunk headers.
    assert all(chunk.startswith("--- a\n+++ b\n") for chunk in separated)
    assert "@@ -1,2 +1,2 @@" in separated[0]
    assert "@@ -10,2 +10,2 @@" in separated[1]


def test_oversized_hunk_split_keeps_body_and_rewrites_headers() -> None:
    diff_text = "--- a\n+++ b\n@@ -1,6 +1,6 @@\n c1\n-old\n+new\n c2\n c3\n c4\n c5\n"
    splitter = UnifiedDiffTextSplitter(chunk_size=40)
    chunks = splitter.split_text(diff_text)
    assert len(chunks) >= 3

    # Every chunk is a valid fragment and stays within the size budget.
    reconstructed_body: list[str] = []
    piece_count = 0
    for chunk in chunks:
        assert len(chunk) <= 40
        header, hunks = _parse_chunk(chunk)
        assert header == ["--- a", "+++ b"]
        for _, body in hunks:
            reconstructed_body.extend(body)
            piece_count += 1
    assert piece_count >= 2

    expected_body = [" c1", "-old", "+new", " c2", " c3", " c4", " c5"]
    assert reconstructed_body == expected_body

    # Re-walk the pieces: rewritten headers must carry accurate line numbers.
    all_pieces: list[tuple[str, list[str]]] = []
    for chunk in chunks:
        _, hunks = _parse_chunk(chunk)
        all_pieces.extend(hunks)
    _assert_valid_piece_headers(all_pieces, old_start=1, new_start=1)


def test_consecutive_added_and_removed_lines_are_never_split() -> None:
    diff_text = "--- a\n+++ b\n@@ -1,5 +1,5 @@\n c1\n-r1\n-r2\n+a1\n+a2\n c2\n c3\n"
    splitter = UnifiedDiffTextSplitter(chunk_size=40)
    chunks = splitter.split_text(diff_text)
    pieces: list[tuple[str, list[str]]] = []
    for chunk in chunks:
        _, hunks = _parse_chunk(chunk)
        pieces.extend(hunks)

    # The four-line +/- run must show up in exactly one piece.
    run_pieces = [
        body
        for _, body in pieces
        if "-r1" in body or "-r2" in body or "+a1" in body or "+a2" in body
    ]
    assert len(run_pieces) == 1
    assert run_pieces[0] == ["-r1", "-r2", "+a1", "+a2"]

    # No piece boundary sits between two change lines.
    for (_, body_a), (_, body_b) in pairwise(pieces):
        assert not (
            body_a[-1].startswith(("-", "+")) and body_b[0].startswith(("-", "+"))
        )


def test_no_newline_marker_stays_with_preceding_line() -> None:
    diff_text = (
        "--- a\n"
        "+++ b\n"
        "@@ -1,3 +1,3 @@\n"
        " c1\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new\n"
        "\\ No newline at end of file\n"
        " c2\n"
    )
    splitter = UnifiedDiffTextSplitter(chunk_size=40)
    chunks = splitter.split_text(diff_text)
    for chunk in chunks:
        _, hunks = _parse_chunk(chunk)
        for _, body in hunks:
            assert not body or body[0] != "\\ No newline at end of file"
            for index, line in enumerate(body):
                if line == "\\ No newline at end of file":
                    assert body[index - 1] in {"-old", "+new"}

    reconstructed: list[str] = []
    for chunk in chunks:
        _, hunks = _parse_chunk(chunk)
        for _, body in hunks:
            reconstructed.extend(body)
    assert reconstructed == [
        " c1",
        "-old",
        "\\ No newline at end of file",
        "+new",
        "\\ No newline at end of file",
        " c2",
    ]


def test_minimum_unit_above_limit_is_emitted_alone() -> None:
    long_line = " " + "x" * 100
    diff_text = "--- a\n+++ b\n@@ -1,2 +1,2 @@\n" + long_line + "\n tail\n"
    splitter = UnifiedDiffTextSplitter(chunk_size=50)
    chunks = splitter.split_text(diff_text)
    # The long context line is an oversized minimum unit: still emitted.
    oversized = [chunk for chunk in chunks if long_line in chunk]
    assert len(oversized) == 1
    assert len(oversized[0]) > 50
    assert " tail" not in oversized[0]
    # Nothing is lost.
    joined_bodies = [
        line for chunk in chunks for _, body in _parse_chunk(chunk)[1] for line in body
    ]
    assert long_line in joined_bodies
    assert " tail" in joined_bodies


def test_removed_line_looking_like_file_header_stays_body() -> None:
    diff_text = (
        "--- a\n+++ b\n@@ -1,2 +1,2 @@\n--- removed content\n+--- added content\n zz\n"
    )
    splitter = UnifiedDiffTextSplitter(chunk_size=1000)
    chunks = splitter.split_text(diff_text)
    assert len(chunks) == 1
    _, hunks = _parse_chunk(chunks[0])
    assert hunks[0][1] == ["--- removed content", "+--- added content", " zz"]


def test_git_extended_headers_are_repeated() -> None:
    diff_text = (
        "diff --git a/new b/new\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/new\n"
        "@@ -0,0 +1,1 @@\n"
        "+hello\n"
    )
    splitter = UnifiedDiffTextSplitter(chunk_size=1000)
    chunks = splitter.split_text(diff_text)
    assert len(chunks) == 1
    assert chunks[0].startswith(
        "diff --git a/new b/new\nnew file mode 100644\nindex 0000000..1111111\n"
        "--- /dev/null\n+++ b/new\n"
    )


def test_added_and_deleted_file_split_headers_use_zero_anchor() -> None:
    added = "--- /dev/null\n+++ b/new\n@@ -0,0 +1,3 @@\n+a\n+b\n+c\n"
    added_chunks = UnifiedDiffTextSplitter(chunk_size=30).split_text(added)
    # The three adds are one inseparable run, so the unit is emitted alone.
    assert any(len(chunk) > 30 for chunk in added_chunks)
    added_piece = _parse_chunk(added_chunks[0])[1][0]
    assert added_piece[0] == "@@ -0,0 +1,3 @@"
    assert added_piece[1] == ["+a", "+b", "+c"]

    deleted = "--- a/old\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a\n-b\n-c\n"
    deleted_chunks = UnifiedDiffTextSplitter(chunk_size=30).split_text(deleted)
    deleted_piece = _parse_chunk(deleted_chunks[0])[1][0]
    assert deleted_piece[0] == "@@ -1,3 +0,0 @@"
    assert deleted_piece[1] == ["-a", "-b", "-c"]


def test_zero_count_piece_uses_insertion_anchor() -> None:
    # Insertion after the first context line; the pure-add piece must point
    # its old range at the preceding line: @@ -1,0 +2,1 @@
    diff_text = "--- a\n+++ b\n@@ -1,3 +1,4 @@\n c1\n+ins\n c2\n c3\n"
    chunks = UnifiedDiffTextSplitter(chunk_size=32).split_text(diff_text)
    all_headers = [
        hunk_header for chunk in chunks for hunk_header, _ in _parse_chunk(chunk)[1]
    ]
    assert "@@ -1,1 +1,1 @@" in all_headers
    assert "@@ -1,0 +2,1 @@" in all_headers
    assert "@@ -2,1 +3,1 @@" in all_headers
    assert "@@ -3,1 +4,1 @@" in all_headers


def test_insertion_before_first_hunk_line_anchors_at_zero() -> None:
    # difflib emits `@@ -1 +1,2 @@` for an addition preceding old line 1.
    diff_text = "--- a\n+++ b\n@@ -1 +1,2 @@\n+new\n c1\n"
    chunks = UnifiedDiffTextSplitter(chunk_size=32).split_text(diff_text)
    all_headers = [
        hunk_header for chunk in chunks for hunk_header, _ in _parse_chunk(chunk)[1]
    ]
    assert "@@ -0,0 +1,1 @@" in all_headers
    assert "@@ -1,1 +2,1 @@" in all_headers


def test_hunk_suffix_is_preserved_on_rewritten_headers() -> None:
    diff_text = (
        "--- a\n"
        "+++ b\n"
        "@@ -1,6 +1,6 @@ def foo():\n"
        " c1\n"
        "-old\n"
        "+new\n"
        " c2\n"
        " c3\n"
        " c4\n"
        " c5\n"
    )
    chunks = UnifiedDiffTextSplitter(chunk_size=40).split_text(diff_text)
    for chunk in chunks:
        for hunk_header, _ in _parse_chunk(chunk)[1]:
            assert hunk_header.endswith(" def foo():")


def test_custom_length_function_counts_lines() -> None:
    diff_text = "--- a\n+++ b\n@@ -1,4 +1,4 @@\n c1\n-x\n+y\n c2\n c3\n"

    def count_lines(text: str) -> int:
        return text.count("\n") + 1

    splitter = UnifiedDiffTextSplitter(chunk_size=5, length_function=count_lines)
    chunks = splitter.split_text(diff_text)
    for chunk in chunks:
        assert count_lines(chunk) <= 5
    reconstructed = [
        line for chunk in chunks for _, body in _parse_chunk(chunk)[1] for line in body
    ]
    assert reconstructed == [" c1", "-x", "+y", " c2", " c3"]


def test_create_documents_preserves_metadata() -> None:
    splitter = UnifiedDiffTextSplitter(chunk_size=40)
    texts = [
        "--- a\n+++ b\n@@ -1,2 +1,2 @@\n c1\n-x\n+y\n",
        "--- c\n+++ d\n@@ -1,1 +1,1 @@\n z\n",
    ]
    metadatas = [{"source": "first.patch", "id": 1}, {"source": "second.patch"}]
    documents = splitter.create_documents(texts, metadatas=metadatas)
    assert documents
    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {"first.patch", "second.patch"}
    first_docs = [d for d in documents if d.metadata["source"] == "first.patch"]
    assert all(d.metadata["id"] == 1 for d in first_docs)
    # Metadata is deep-copied per chunk.
    assert all(doc.metadata is not metadatas[0] for doc in first_docs)


def test_split_documents_preserves_metadata() -> None:
    splitter = UnifiedDiffTextSplitter(chunk_size=40)
    source = Document(
        page_content=(
            "--- a\n+++ b\n@@ -1,6 +1,6 @@\n c1\n-old\n+new\n c2\n c3\n c4\n c5\n"
        ),
        metadata={"source": "patch.diff"},
    )
    documents = splitter.split_documents([source])
    assert len(documents) >= 2
    assert all(doc.metadata == {"source": "patch.diff"} for doc in documents)
    assert all(doc.metadata is not source.metadata for doc in documents)


@pytest.mark.parametrize(
    "diff_text",
    [
        # Missing +++ header
        "--- a\n@@ -1,1 +1,1 @@\n x\n",
        # Hunk outside of any file section
        "@@ -1,1 +1,1 @@\n x\n",
        # More body lines than declared (old side)
        "--- a\n+++ b\n@@ -1,1 +1,1 @@\n x\n-extra\n",
        # More body lines than declared (new side)
        "--- a\n+++ b\n@@ -1,1 +1,1 @@\n x\n+extra\n",
        # Truncated body
        "--- a\n+++ b\n@@ -1,3 +1,3 @@\n x\n",
        # Invalid hunk header
        "--- a\n+++ b\n@@ broken\n x\n",
        # Orphan no-newline marker
        "--- a\n+++ b\n@@ -1,1 +1,1 @@\n\\ No newline at end of file\n",
        # Trailing content outside any file section
        "--- a\n+++ b\n@@ -1,1 +1,1 @@\n x\nrandom trailing line\n",
    ],
)
def test_malformed_diff_raises(diff_text: str) -> None:
    splitter = UnifiedDiffTextSplitter()
    with pytest.raises(ValueError, match="Malformed unified diff"):
        splitter.split_text(diff_text)


def test_bodyless_hunk_header_emitted_alone_when_over_limit() -> None:
    diff_text = "--- a\n+++ b\n@@ -1,0 +1,0 @@\n"
    chunks = UnifiedDiffTextSplitter(chunk_size=10).split_text(diff_text)
    assert chunks == ["--- a\n+++ b\n@@ -1,0 +1,0 @@"]


def test_multi_file_split_does_not_lose_or_duplicate_body() -> None:
    files_text = ""
    for file_index in range(3):
        body = "".join(f" c{line_index}\n" for line_index in range(6))
        files_text += (
            f"--- a/f{file_index}\n+++ b/f{file_index}\n@@ -1,6 +1,6 @@\n{body}"
        )
    splitter = UnifiedDiffTextSplitter(chunk_size=45)
    chunks = splitter.split_text(files_text)
    assert len(chunks) > 3
    for file_index in range(3):
        file_chunks = [c for c in chunks if f"--- a/f{file_index}" in c]
        assert file_chunks
        bodies: list[str] = []
        for chunk in file_chunks:
            _, hunks = _parse_chunk(chunk)
            for _, body in hunks:
                bodies.extend(body)
        assert bodies == [f" c{line_index}" for line_index in range(6)]
        for chunk in file_chunks:
            assert len(chunk) <= 45
