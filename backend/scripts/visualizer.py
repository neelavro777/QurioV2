"""
visualizer.py
-------------
Renders the compiled Qurio LangGraph as a PNG image using LangGraph's
built-in Mermaid → PNG pipeline (no Graphviz required).

Usage:
    python visualizer.py                   # saves to graph_visualization.png
    python visualizer.py --output my.png   # saves to custom path
    python visualizer.py --show            # saves AND opens the image

How it works:
    graph.get_graph().draw_mermaid_png() converts the graph's internal
    Mermaid representation to PNG bytes via the MermaidDrawMethod.API
    renderer (uses the free mermaid.ink cloud endpoint — internet needed).

    The resulting bytes are written to disk and optionally opened with the
    system image viewer (xdg-open on Linux).
"""

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize the Qurio LangGraph as a PNG diagram."
    )
    p.add_argument(
        "--output", "-o",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "graph_visualization.png"),
        help="Output file path (default: project root / graph_visualization.png)",
    )
    p.add_argument(
        "--show", "-s",
        action="store_true",
        help="Open the image after saving (uses xdg-open on Linux).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_path = os.path.abspath(args.output)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Qurio LangGraph Visualizer                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # --- Import the compiled graph ---
    print("⟳ Importing compiled graph from graph.py …")
    
    # Add backend/ to sys.path so 'app.' imports work
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from app.agent.graph import graph  # noqa: F401 — triggers graph compilation
    except Exception as e:
        print(f"\n✗ Failed to import graph: {e}")
        sys.exit(1)
    print("✓ Graph imported successfully.\n")

    # --- Print Mermaid source for reference ---
    print("─── Mermaid source ───────────────────────────────────────")
    try:
        mermaid_src = graph.get_graph().draw_mermaid()
        print(mermaid_src)
    except Exception as e:
        print(f"  (could not render Mermaid source: {e})")
    print("──────────────────────────────────────────────────────────\n")

    # --- Render PNG ---
    print(f"⟳ Rendering PNG via mermaid.ink …")
    print("  (requires internet access — uses the free mermaid.ink API)\n")

    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        png_bytes = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    except ImportError:
        # Older LangGraph versions don't expose MermaidDrawMethod
        try:
            png_bytes = graph.get_graph().draw_mermaid_png()
        except Exception as e:
            print(f"✗ draw_mermaid_png() failed: {e}")
            print(
                "\nFallback: saving Mermaid source as .md instead.\n"
                "Paste it into https://mermaid.live to render manually."
            )
            md_path = output_path.replace(".png", "_mermaid.md")
            with open(md_path, "w") as f:
                f.write("```mermaid\n" + mermaid_src + "\n```\n")
            print(f"✓ Mermaid source saved to: {md_path}")
            sys.exit(0)
    except Exception as e:
        print(f"✗ PNG rendering failed: {e}")
        print(
            "\nCommon causes:\n"
            "  - No internet access (mermaid.ink requires a network connection)\n"
            "  - mermaid.ink rate-limited (try again in a few seconds)\n"
            "\nFallback: copy the Mermaid source above into https://mermaid.live"
        )
        sys.exit(1)

    # --- Save PNG ---
    try:
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        size_kb = len(png_bytes) / 1024
        print(f"✓ PNG saved to: {output_path}  ({size_kb:.1f} KB)\n")
    except OSError as e:
        print(f"✗ Failed to write file: {e}")
        sys.exit(1)

    # --- Annotate the graph structure in the terminal ---
    print("─── Graph structure ──────────────────────────────────────")
    print()
    print("  START")
    print("    │")
    print("    ▼")
    print("  classify_intent          [Low-temp structured LLM routing]")
    print("    │")
    print("    ├── 'chat'              ──► prompt_llm_chat ──► END")
    print("    ├── 'search_knowledge_base' ─► prompt_llm_search_knowledge_base ──► END")
    print("    └── 'query_user_information'")
    print("          │")
    print("          ▼")
    print("  agent_execute            [True ReAct node: tool_choice='auto']")
    print("    │")
    print("    ├── [has tool_calls?]")
    print("    │     YES ──► tool_node ──► agent_execute  (ReAct loop)")
    print("    │               ↑               │")
    print("    │               └───────────────┘")
    print("    └── NO  ──► END")
    print()
    print("  tool_choice strategy inside agent_execute:")
    print("    True ReAct: always 'auto' so the LLM decides autonomously when to call tools.")
    print("    Loop guard fired (duplicate call detected) → None (hard stop to prevent loop).")
    print()
    print("──────────────────────────────────────────────────────────\n")

    # --- Open image if requested ---
    if args.show:
        print("⟳ Opening image …")
        try:
            subprocess.Popen(["xdg-open", output_path])
            print("✓ Opened with system image viewer.")
        except FileNotFoundError:
            print("  xdg-open not found. Open the file manually:")
            print(f"  {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
