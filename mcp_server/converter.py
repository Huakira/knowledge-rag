"""Document conversion (fork): PDF / EPUB -> Markdown next to the source file.

Rules (kept deliberately few):
- Converted files are Markdown written in the same directory as the source.
- Never overwrite an existing file unless ``overwrite=True`` is requested.
- A conversion ledger (SQLite in the data dir, keyed by the source SHA-256)
  remembers what was converted. Converted-once sources are never converted
  again automatically: moving the source does not matter, and deleting the
  Markdown does not trigger a re-conversion.
- Failures write nothing and are NOT recorded in the ledger, so they are
  retried on the next scan and reported through ``failed_files``.
- Files modified in the last ``min_age_seconds`` are skipped (half-copied).
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ENGINES = ("pymupdf4llm", "doc2x", "pandoc")

_PYMUPDF_SCRIPT = (
    "import sys, pymupdf4llm\n"
    "md = pymupdf4llm.to_markdown(sys.argv[1])\n"
    "open(sys.argv[2], 'w', encoding='utf-8').write(md)\n"
)


class ConversionError(RuntimeError):
    pass


class ConversionLedger:
    """SQLite ledger: sha256(source) -> converted markdown."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS conversions ("
                " source_sha256 TEXT PRIMARY KEY, source_path TEXT, md_path TEXT, md_sha256 TEXT,"
                " converted_at REAL, tool TEXT, status TEXT)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER, sha256 TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30)

    def get(self, sha: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT source_sha256, source_path, md_path, md_sha256, converted_at, tool, status"
                " FROM conversions WHERE source_sha256=?",
                (sha,),
            ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row) -> Dict[str, Any]:
        keys = ("source_sha256", "source_path", "md_path", "md_sha256", "converted_at", "tool", "status")
        return dict(zip(keys, row))

    def put(self, sha: str, source_path: str, md_path: str, md_sha: str, tool: str, status: str = "done") -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO conversions VALUES (?,?,?,?,?,?,?)",
                (sha, source_path, md_path, md_sha, time.time(), tool, status),
            )

    def forget(self, sha: str) -> int:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM conversions WHERE source_sha256=?", (sha,)).rowcount

    def list(self) -> List[Dict[str, Any]]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT source_sha256, source_path, md_path, md_sha256, converted_at, tool, status"
                " FROM conversions ORDER BY converted_at DESC"
            ).fetchall()
        return [self._row(r) for r in rows]

    def cached_hash(self, path: str, mtime_ns: int, size: int) -> Optional[str]:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT mtime_ns, size, sha256 FROM hashes WHERE path=?", (path,)).fetchone()
        if row and row[0] == mtime_ns and row[1] == size:
            return row[2]
        return None

    def store_hash(self, path: str, mtime_ns: int, size: int, sha: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?,?)", (path, mtime_ns, size, sha))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Converter:
    def __init__(
        self,
        documents_dir: Path,
        data_dir: Path,
        exclude_patterns: List[str],
        convertible: List[str],
        min_age_seconds: float = 30.0,
        scanned_chars_per_page: float = 50.0,
        doc2x_enabled: bool = True,
        doc2x_max_pages: int = 500,
        doc2x_timeout: int = 1800,
        pandoc_timeout: int = 600,
        pymupdf_timeout: int = 900,
        min_output_chars: int = 50,
    ):
        self.documents_dir = Path(documents_dir)
        self.data_dir = Path(data_dir)
        self.exclude = exclude_patterns
        self.convertible = {s.lower() for s in convertible}
        self.min_age = min_age_seconds
        self.scanned_cpp = scanned_chars_per_page
        self.doc2x_enabled = doc2x_enabled
        self.doc2x_max_pages = doc2x_max_pages
        self.doc2x_timeout = doc2x_timeout
        self.pandoc_timeout = pandoc_timeout
        self.pymupdf_timeout = pymupdf_timeout
        self.min_output_chars = min_output_chars
        self.ledger = ConversionLedger(self.data_dir / "conversions.db")

    # ------------------------------------------------------------------ scan
    def _excluded(self, path: Path) -> bool:
        if not self.exclude:
            return False
        try:
            rel = path.relative_to(self.documents_dir)
        except ValueError:
            rel = path
        rel_str = str(rel).replace("\\", "/")
        for pat in self.exclude:
            if fnmatch.fnmatch(rel_str, pat) or any(fnmatch.fnmatch(part, pat) for part in rel.parts):
                return True
        return False

    def iter_sources(self):
        for root, dirs, files in os.walk(self.documents_dir):
            dirs[:] = [d for d in dirs if not self._excluded(Path(root) / d)]
            for name in files:
                p = Path(root) / name
                if p.suffix.lower() in self.convertible and not name.startswith((".rag-pending-", "~$")):
                    if not self._excluded(p):
                        yield p

    def source_hash(self, path: Path) -> str:
        st = path.stat()
        cached = self.ledger.cached_hash(str(path), st.st_mtime_ns, st.st_size)
        if cached:
            return cached
        sha = sha256_file(path)
        self.ledger.store_hash(str(path), st.st_mtime_ns, st.st_size, sha)
        return sha

    # ------------------------------------------------------------------ run
    def run(self, record_failed: Callable[[str, str], None]) -> Dict[str, Any]:
        """Convert every new convertible source. Returns a summary dict."""
        summary = {"converted": 0, "skipped": 0, "failed": 0, "deferred": 0}
        doc2x_fail_streak = 0
        for src in self.iter_sources():
            try:
                st = src.stat()
            except OSError:
                continue
            if time.time() - st.st_mtime < self.min_age:
                summary["deferred"] += 1
                continue
            try:
                sha = self.source_hash(src)
                if self.ledger.get(sha):
                    summary["skipped"] += 1
                    continue
                target = src.with_suffix(".md")
                if target.exists():
                    self.ledger.put(sha, str(src), str(target), "", "-", "target_exists")
                    summary["skipped"] += 1
                    continue
                engine, note = self.choose_engine(src, auto=True)
                if engine == "doc2x" and doc2x_fail_streak >= 3:
                    raise ConversionError("doc2x skipped: 3 consecutive doc2x failures this round")
                try:
                    self.convert(src, engine=engine, overwrite=False, sha=sha)
                except ConversionError:
                    if engine == "doc2x":
                        doc2x_fail_streak += 1
                    raise
                if engine == "doc2x":
                    doc2x_fail_streak = 0
                summary["converted"] += 1
            except Exception as exc:  # noqa: BLE001 - report every failure, keep going
                summary["failed"] += 1
                record_failed(str(src), f"conversion failed: {str(exc)[:240]}")
        return summary

    # -------------------------------------------------------------- engines
    def pdf_stats(self, path: Path) -> Dict[str, float]:
        import pymupdf

        with pymupdf.open(str(path)) as doc:
            pages = doc.page_count
            sample = range(pages) if pages <= 40 else range(0, pages, max(1, pages // 40))
            chars = sum(len(doc[i].get_text().strip()) for i in sample)
            n = max(1, len(list(sample)))
        return {"pages": pages, "chars_per_page": chars / n}

    def choose_engine(self, path: Path, auto: bool = True):
        suffix = path.suffix.lower()
        if suffix == ".epub":
            return "pandoc", "epub"
        if suffix == ".pdf":
            s = self.pdf_stats(path)
            if s["chars_per_page"] >= self.scanned_cpp:
                return "pymupdf4llm", f"text layer {s['chars_per_page']:.0f} chars/page"
            if not self.doc2x_enabled:
                raise ConversionError("scanned PDF (no text layer) and doc2x is disabled")
            if auto and s["pages"] > self.doc2x_max_pages:
                raise ConversionError(
                    f"scanned PDF with {int(s['pages'])} pages exceeds the automatic doc2x cap "
                    f"({self.doc2x_max_pages}); run the convert tool with engine=doc2x to proceed"
                )
            return "doc2x", f"scanned, {int(s['pages'])} pages"
        raise ConversionError(f"unsupported source type: {suffix}")

    def _run_engine(self, engine: str, src: Path, out_md: Path) -> None:
        if engine == "pymupdf4llm":
            r = subprocess.run(
                [sys.executable, "-I", "-c", _PYMUPDF_SCRIPT, str(src), str(out_md)],
                capture_output=True,
                text=True,
                timeout=self.pymupdf_timeout,
            )
            if r.returncode != 0:
                raise ConversionError(f"pymupdf4llm failed: {r.stderr.strip()[-200:]}")
        elif engine == "pandoc":
            r = subprocess.run(
                ["pandoc", "-f", "epub", "-t", "gfm", "--wrap=none", "-o", str(out_md), str(src)],
                capture_output=True,
                text=True,
                timeout=self.pandoc_timeout,
            )
            if r.returncode != 0:
                raise ConversionError(f"pandoc failed: {r.stderr.strip()[-200:]}")
        elif engine == "doc2x":
            outdir = out_md.parent / "doc2x-out"
            r = subprocess.run(
                ["doc2x", "parse", str(src), "--to", "md", "--out", str(outdir), "--quiet"],
                capture_output=True,
                text=True,
                timeout=self.doc2x_timeout,
            )
            if r.returncode != 0:
                raise ConversionError(f"doc2x failed: {(r.stderr or r.stdout).strip()[-200:]}")
            cands = sorted(outdir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
            if not cands:
                raise ConversionError("doc2x produced no markdown file")
            shutil.copyfile(cands[0], out_md)
        else:
            raise ConversionError(f"unknown engine: {engine}")

    def convert(
        self,
        src: Path,
        engine: Optional[str] = None,
        overwrite: bool = False,
        sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        src = Path(src)
        if not src.is_file():
            raise ConversionError(f"source not found: {src}")
        if src.suffix.lower() not in self.convertible:
            raise ConversionError(f"not a convertible format: {src.suffix}")
        if engine is None:
            engine, _ = self.choose_engine(src, auto=False)
        if engine not in ENGINES:
            raise ConversionError(f"engine must be one of {ENGINES}")
        target = src.with_suffix(".md")
        if target.exists() and not overwrite:
            raise ConversionError(f"target exists: {target} (use overwrite=true)")
        sha = sha or self.source_hash(src)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="conv-", dir=str(self.data_dir)))
        try:
            out_md = work / "out.md"
            try:
                self._run_engine(engine, src, out_md)
            except subprocess.TimeoutExpired as exc:
                raise ConversionError(f"{engine} timed out after {exc.timeout}s") from exc
            if not out_md.exists():
                raise ConversionError(f"{engine} produced no output")
            text = out_md.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < self.min_output_chars:
                raise ConversionError(f"{engine} output is empty or near-empty ({len(text.strip())} chars)")
            # Write next to the source atomically; temp name is ignored by the watcher.
            tmp = target.with_name(f".rag-pending-conv-{os.getpid()}-{target.name}")
            shutil.copyfile(out_md, tmp)
            try:
                if overwrite:
                    os.replace(tmp, target)
                else:
                    os.link(tmp, target)  # fails if target appeared meanwhile: never clobber
                    os.unlink(tmp)
            except FileExistsError as exc:
                os.unlink(tmp)
                raise ConversionError(f"target exists: {target}") from exc
            md_sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
            self.ledger.put(sha, str(src), str(target), md_sha, engine)
            return {"source": str(src), "markdown": str(target), "engine": engine, "chars": len(text)}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ----------------------------------------------------------- reporting
    def report(self) -> List[Dict[str, Any]]:
        out = []
        for row in self.ledger.list():
            row = dict(row)
            row["md_exists"] = bool(row["md_path"]) and Path(row["md_path"]).exists()
            row["source_exists"] = bool(row["source_path"]) and Path(row["source_path"]).exists()
            out.append(row)
        return out
