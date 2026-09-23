"""Rendering the compiled graph (delivery requirement of the brief).

The brief asks for ``.get_graph().draw_mermaid_png()``. That call renders the
diagram through a remote Mermaid service, which is unavailable on an offline
or firewalled machine. :class:`MermaidGraphVisualizer` therefore tries the PNG
first and degrades to the Mermaid source (saved as ``.mmd`` and printable),
so the deliverable never depends on the network to be inspectable.

This is a textbook Open/Closed extension point: another visualiser (Graphviz,
ASCII) only has to implement the same ``render`` method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MermaidGraphVisualizer:
    """Renders a compiled LangGraph, with a graceful offline fallback."""

    def __init__(self, output_dir: Path | str = "artifacts") -> None:
        self._output_dir = Path(output_dir)

    @property
    def name(self) -> str:
        return "mermaid"

    def render(self, compiled_graph: Any, output_path: str | None = None) -> dict[str, Any]:
        """Render ``compiled_graph`` and report what was produced.

        Returns a dict with ``format`` ("png" or "mermaid"), the file ``path``
        and the Mermaid ``source`` - always populated, so the caller can always
        show *something*.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        graph = compiled_graph.get_graph()
        mermaid_source = graph.draw_mermaid()

        png_path = Path(output_path) if output_path else self._output_dir / "workflow_graph.png"
        try:
            # The call the project brief explicitly asks for.
            png_bytes = graph.draw_mermaid_png()
            png_path.write_bytes(png_bytes)
            return {
                "format": "png",
                "path": str(png_path),
                "source": mermaid_source,
                "png_bytes": png_bytes,
                "note": "Rendered with draw_mermaid_png().",
            }
        except Exception as error:  # noqa: BLE001 - network/browser errors vary
            mmd_path = png_path.with_suffix(".mmd")
            mmd_path.write_text(mermaid_source, encoding="utf-8")
            return {
                "format": "mermaid",
                "path": str(mmd_path),
                "source": mermaid_source,
                "png_bytes": None,
                "note": (
                    "draw_mermaid_png() needs the mermaid.ink renderer, which is "
                    f"unreachable here ({type(error).__name__}: {error}). The Mermaid "
                    "source was written instead - paste it into any Mermaid viewer, "
                    "or re-run this cell on a machine with internet access."
                ),
            }


__all__ = ["MermaidGraphVisualizer"]
