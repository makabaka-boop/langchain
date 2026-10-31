"""Unified diff text splitter.

When code patches are retrieved as text, generic character-based splitters can
detach changed lines from the file name and the ``@@`` hunk headers that carry
their line numbers. `UnifiedDiffTextSplitter` is aware of the unified diff
format and keeps that context on every chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from typing_extensions import override

from langchain_text_splitters.base import TextSplitter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_HUNK_HEADER_PATTERN = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?P<suffix>.*)$"
)
_NO_NEWLINE_MARKER = "\\ No newline at end of file"
_OLD_FILE_PREFIX = "--- "
_NEW_FILE_PREFIX = "+++ "
_HUNK_PREFIX = "@@ "


@dataclass
class _BodyUnit:
    """An unsplittable group of consecutive diff body lines.

    A context line is a unit on its own. A maximal run of consecutive added
    and removed lines forms a single unit so paired changes are never split
    apart. No-newline markers are kept on the unit of their preceding line.
    """

    lines: list[str]
    old_count: int
    new_count: int


@dataclass
class _Hunk:
    """A parsed ``@@`` hunk together with its atomic body units."""

    header_line: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    suffix: str
    units: list[_BodyUnit]

    @property
    def body_lines(self) -> list[str]:
        """All body lines of the hunk in their original order."""
        lines: list[str] = []
        for unit in self.units:
            lines.extend(unit.lines)
        return lines


@dataclass
class _FileDiff:
    """A single file section: its file header followed by its hunks."""

    header_lines: list[str]
    hunks: list[_Hunk]


@dataclass
class _RawBodyLine:
    """One parsed body line with an optional attached no-newline marker."""

    line: str
    old_delta: int
    new_delta: int
    marker: str | None = None


def _raise_malformed(detail: str) -> NoReturn:
    """Raise a `ValueError` describing a malformed unified diff.

    Args:
        detail: Description of the structural problem.

    Raises:
        ValueError: Always, with the provided detail message.
    """
    msg = f"Malformed unified diff: {detail}"
    raise ValueError(msg)


def _parse_hunk(lines: Sequence[str], start_index: int) -> tuple[_Hunk, int]:
    """Parse one hunk (header and body) starting at `start_index`.

    Args:
        lines: All lines of the diff text.
        start_index: Index of the hunk header line.

    Returns:
        A tuple of the parsed `_Hunk` and the index immediately after the
        hunk body.

    Raises:
        ValueError: If the hunk header cannot be parsed, the body ends
            prematurely, a body line has an unexpected prefix, the declared
            line counts do not match the body, or a no-newline marker is
            not attached to a body line.
    """
    header_line = lines[start_index]
    match = _HUNK_HEADER_PATTERN.match(header_line)
    if match is None:
        _raise_malformed(f"invalid hunk header: {header_line!r}")
    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or "1")
    suffix = match.group("suffix")
    index = start_index + 1

    raw_lines: list[_RawBodyLine] = []
    consumed_old = 0
    consumed_new = 0
    while consumed_old < old_count or consumed_new < new_count:
        if index >= len(lines):
            _raise_malformed(
                f"hunk starting at {header_line!r} is truncated: expected "
                f"{old_count} old and {new_count} new body lines"
            )
        line = lines[index]
        if line == _NO_NEWLINE_MARKER:
            if not raw_lines or raw_lines[-1].marker is not None:
                _raise_malformed(
                    "no-newline marker is not attached to a preceding body line"
                )
            raw_lines[-1].marker = line
            index += 1
            continue
        if not line:
            _raise_malformed("blank line is not a valid diff body line")
        prefix = line[0]
        if prefix == " ":
            old_delta, new_delta = 1, 1
        elif prefix == "+":
            old_delta, new_delta = 0, 1
        elif prefix == "-":
            old_delta, new_delta = 1, 0
        else:
            _raise_malformed(f"unexpected diff body line: {line!r}")
        if consumed_old + old_delta > old_count or (
            consumed_new + new_delta > new_count
        ):
            _raise_malformed(
                f"hunk starting at {header_line!r} has more body lines than "
                f"its declared counts ({old_count} old, {new_count} new)"
            )
        raw_lines.append(_RawBodyLine(line, old_delta, new_delta))
        consumed_old += old_delta
        consumed_new += new_delta
        index += 1

    # A marker following the final body line is only reachable once the
    # declared counts are already satisfied.
    if index < len(lines) and lines[index] == _NO_NEWLINE_MARKER:
        if not raw_lines:
            _raise_malformed(
                "no-newline marker is not attached to a preceding body line"
            )
        raw_lines[-1].marker = lines[index]
        index += 1

    units = _group_body_units(raw_lines)
    hunk = _Hunk(
        header_line=header_line,
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        suffix=suffix,
        units=units,
    )
    return hunk, index


def _group_body_units(raw_lines: Sequence[_RawBodyLine]) -> list[_BodyUnit]:
    """Group parsed body lines into atomic units.

    Context lines each become their own unit. Consecutive added and removed
    lines are grouped into one unit together with their attached no-newline
    markers.

    Args:
        raw_lines: The parsed body lines in their original order.

    Returns:
        The list of atomic body units.
    """
    units: list[_BodyUnit] = []
    index = 0
    while index < len(raw_lines):
        raw_line = raw_lines[index]
        group: list[_RawBodyLine]
        if raw_line.old_delta == 1 and raw_line.new_delta == 1:
            group = [raw_line]
            index += 1
        else:
            group = []
            while index < len(raw_lines):
                candidate = raw_lines[index]
                if candidate.old_delta == 1 and candidate.new_delta == 1:
                    break
                group.append(candidate)
                index += 1
        unit_lines: list[str] = []
        unit_old_count = 0
        unit_new_count = 0
        for entry in group:
            unit_lines.append(entry.line)
            if entry.marker is not None:
                unit_lines.append(entry.marker)
            unit_old_count += entry.old_delta
            unit_new_count += entry.new_delta
        units.append(
            _BodyUnit(
                lines=unit_lines,
                old_count=unit_old_count,
                new_count=unit_new_count,
            )
        )
    return units


def _parse_unified_diff(text: str) -> list[_FileDiff]:
    """Parse unified diff text into per-file sections.

    Each section consists of a file header (extended header lines followed by
    the ``---`` and ``+++`` lines) and its ``@@`` hunks. A ``---`` line only
    starts a new file when the current hunk body is complete, because removed
    body lines may themselves start with ``---``.

    Args:
        text: The unified diff text.

    Returns:
        The parsed file sections in order.

    Raises:
        ValueError: If the diff structure is invalid, for example a missing
            ``+++`` header, a hunk outside a file section, a dangling extended
            header, or content that belongs to no file section.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    files: list[_FileDiff] = []
    pending_header: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        if not line.startswith(_OLD_FILE_PREFIX):
            pending_header.append(line)
            index += 1
            continue

        old_line = line
        index += 1
        if index >= total or not lines[index].startswith(_NEW_FILE_PREFIX):
            _raise_malformed("missing '+++' file header after '---' line")
        new_line = lines[index]
        index += 1
        header_lines = [*pending_header, old_line, new_line]
        pending_header = []

        hunks: list[_Hunk] = []
        while index < total:
            current = lines[index]
            if current.startswith(_OLD_FILE_PREFIX):
                break
            if current.startswith(_HUNK_PREFIX):
                hunk, index = _parse_hunk(lines, index)
                hunks.append(hunk)
                continue
            # Any other line starts the extended header of the next file.
            # It must be followed by another '---' file header; hunk headers
            # cannot follow it inside the current file section.
            pending_header.append(current)
            index += 1
            while index < total and not lines[index].startswith(_OLD_FILE_PREFIX):
                pending_header.append(lines[index])
                index += 1
            break

        files.append(_FileDiff(header_lines=header_lines, hunks=hunks))

    if pending_header:
        _raise_malformed(
            "content outside of a file section is not followed by a file header"
        )
    return files


class UnifiedDiffTextSplitter(TextSplitter):
    r"""Splitting unified diff text while keeping file and hunk context.

    Handles unified diff text built from file headers (``---``/``+++`` plus
    any extended header lines such as ``diff --git``) and ordinary ``@@``
    hunks. Multiple files as well as added and deleted files (``/dev/null``)
    are supported.

    Every chunk contains exactly one file and repeats that file's header.
    Complete hunks are merged into a chunk in order while they fit. A hunk
    that does not fit is split along its body lines, except that consecutive
    added (``+``) and removed (``-``) lines are never separated and the
    ``\\ No newline at end of file`` marker always stays with its preceding
    line. Rewritten ``@@`` headers carry accurate old/new start line numbers
    and counts, so every piece is itself a valid unified diff hunk. Body
    lines are never dropped, duplicated, or rewritten.

    Chunks never overlap. A minimum unsplittable unit that is longer than
    `chunk_size` is returned on its own rather than being dropped.
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        length_function: Callable[[str], int] = len,
        **kwargs: Any,
    ) -> None:
        """Create a new `UnifiedDiffTextSplitter`.

        Args:
            chunk_size: Maximum length of each chunk, measured on the complete
                rendered chunk via `length_function`.
            length_function: Function that measures the length of a chunk.
            **kwargs: Additional keyword arguments forwarded to
                `TextSplitter`. `chunk_overlap` is forced to ``0`` because
                diff chunks must never duplicate body lines.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=length_function,
            **kwargs,
        )

    @override
    def split_text(self, text: str) -> list[str]:
        """Split unified diff text into chunks.

        Args:
            text: The unified diff text to split.

        Returns:
            The list of chunks, each containing a single file's repeated
            file header followed by one or more complete or rewritten diff
            hunks. An empty string returns an empty list.

        Raises:
            ValueError: If the diff structure or a hunk's declared line
                counts are invalid.
        """
        if text == "":
            return []
        files = _parse_unified_diff(text)
        chunks: list[str] = []
        for file_diff in files:
            chunks.extend(self._split_file(file_diff))
        return chunks

    def _length(self, lines: Sequence[str]) -> int:
        return self._length_function("\n".join(lines))

    def _split_file(self, file_diff: _FileDiff) -> list[str]:
        """Split one parsed file section into chunks.

        Args:
            file_diff: The parsed file section.

        Returns:
            The chunks for this file, each repeating the file header.
        """
        header = file_diff.header_lines
        chunks: list[str] = []
        current: list[str] = list(header)

        for hunk in file_diff.hunks:
            hunk_lines = [hunk.header_line, *hunk.body_lines]
            if self._length([*current, *hunk_lines]) <= self._chunk_size:
                current.extend(hunk_lines)
                continue

            # The hunk does not fit the open chunk. Finish that chunk first,
            # then retry against a fresh chunk that only carries the header.
            if len(current) > len(header):
                chunks.append("\n".join(current))
                current = list(header)
            if self._length([*current, *hunk_lines]) <= self._chunk_size:
                current.extend(hunk_lines)
                continue

            # The complete hunk is too large even on its own: split it into
            # valid sub-hunks. All pieces except the last are emitted as
            # standalone chunks; the last stays open so later complete hunks
            # can still merge into it.
            pieces = self._split_oversized_hunk(header, hunk)
            for piece_header, piece_body in pieces[:-1]:
                current.extend([piece_header, *piece_body])
                chunks.append("\n".join(current))
                current = list(header)
            last_header, last_body = pieces[-1]
            current.extend([last_header, *last_body])

        if not file_diff.hunks or len(current) > len(header):
            chunks.append("\n".join(current))
        return chunks

    def _split_oversized_hunk(
        self, header: Sequence[str], hunk: _Hunk
    ) -> list[tuple[str, list[str]]]:
        """Split a hunk that does not fit into a fresh chunk.

        Units are packed greedily. Each rendered piece (file header plus
        rewritten hunk header plus body) is measured as a complete chunk. A
        single unit larger than `chunk_size` is emitted as its own piece.

        Args:
            header: The file header lines repeated on every chunk.
            hunk: The oversized hunk to split.

        Returns:
            The ordered `(rewritten hunk header, body lines)` pieces.
        """
        if not hunk.units:
            # A bodyless hunk header is irreducible; emit it on its own.
            return [(hunk.header_line, [])]

        pieces: list[tuple[str, list[str]]] = []
        units = hunk.units
        start = 0
        cursor_old = hunk.old_start
        cursor_new = hunk.new_start
        while start < len(units):
            end = start
            while end < len(units):
                trial = units[start : end + 1]
                piece_header = self._render_piece_header(
                    hunk,
                    old_before=cursor_old - hunk.old_start,
                    new_before=cursor_new - hunk.new_start,
                    old_count=sum(unit.old_count for unit in trial),
                    new_count=sum(unit.new_count for unit in trial),
                )
                body_lines = [line for unit in trial for line in unit.lines]
                candidate_length = self._length([*header, piece_header, *body_lines])
                if candidate_length <= self._chunk_size:
                    end += 1
                else:
                    break

            if end == start:
                # The minimum unsplittable unit exceeds the limit; emit it
                # alone rather than dropping or further splitting it.
                end = start + 1

            chosen = units[start:end]
            old_count = sum(unit.old_count for unit in chosen)
            new_count = sum(unit.new_count for unit in chosen)
            piece_header = self._render_piece_header(
                hunk,
                old_before=cursor_old - hunk.old_start,
                new_before=cursor_new - hunk.new_start,
                old_count=old_count,
                new_count=new_count,
            )
            body_lines = [line for unit in chosen for line in unit.lines]
            pieces.append((piece_header, body_lines))
            cursor_old += old_count
            cursor_new += new_count
            start = end
        return pieces

    @staticmethod
    def _zero_range_start(base_start: int, before: int, total_count: int) -> int:
        """Return the start line number for a zero-count hunk range.

        When lines of this side precede the piece inside the hunk, the empty
        range anchors at the last preceding line. At the very start of the
        hunk, a hunk that has no lines on this side at all already carries
        the insertion anchor in its header; otherwise the empty range sits
        just before the hunk's first line.

        Args:
            base_start: The hunk header's start for this side.
            before: Number of this side's lines preceding the piece.
            total_count: The hunk header's total count for this side.

        Returns:
            The zero-count start line number, clamped to ``0``.
        """
        if before > 0:
            return base_start + before - 1
        if total_count == 0:
            return base_start
        return max(base_start - 1, 0)

    @classmethod
    def _render_piece_header(
        cls,
        hunk: _Hunk,
        *,
        old_before: int,
        new_before: int,
        old_count: int,
        new_count: int,
    ) -> str:
        """Render a rewritten hunk header for a sub-hunk piece.

        Args:
            hunk: The original hunk.
            old_before: Number of old-side lines preceding the piece.
            new_before: Number of new-side lines preceding the piece.
            old_count: Number of old-side lines in the piece.
            new_count: Number of new-side lines in the piece.

        Returns:
            The rewritten ``@@`` hunk header, preserving the original suffix.
        """
        if old_count > 0:
            old_start = hunk.old_start + old_before
        else:
            old_start = cls._zero_range_start(
                hunk.old_start, old_before, hunk.old_count
            )
        if new_count > 0:
            new_start = hunk.new_start + new_before
        else:
            new_start = cls._zero_range_start(
                hunk.new_start, new_before, hunk.new_count
            )
        return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{hunk.suffix}"
