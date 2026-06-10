import json
import logging
import sys
import uuid
import datetime
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import graph

# Setup basic logging for the orchestrator
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "e2e_dataset.json"
REPORTS_DIR = Path(__file__).parent / "reports"

def run_experiment():
    if not DATASET_PATH.exists():
        logger.error(f"Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_filename = f"eval_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = REPORTS_DIR / report_filename

    total_turns_tested = 0
    passed_turns = 0
    results = []

    logger.info(f"Starting Multi-Turn Orchestrator. Loaded {len(dataset)} sequences.")

    for seq_idx, sequence in enumerate(dataset):
        seq_id = sequence["id"]
        description = sequence.get("description", "")
        turns = sequence["turns"]
        
        logger.info(f"\n[{seq_idx+1}/{len(dataset)}] Executing Sequence: {seq_id}")
        logger.info(f"Description: {description}")

        # Unique thread ID for the entire sequence (persists memory across turns)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        seq_results = {"id": seq_id, "description": description, "turns": []}

        for turn_idx, turn in enumerate(turns):
            total_turns_tested += 1
            query = turn["query"]
            expected_intent = turn["expected_intent"]
            expected_tool = turn["expected_tool"]
            expected_sql_keywords = turn.get("expected_sql_keywords", [])
            expected_content_keywords = turn.get("expected_content_keywords", [])
            prohibited_keywords = turn.get("prohibited_keywords", [])

            logger.info(f"  Turn {turn_idx+1}: '{query}'")

            initial_state = {"messages": [HumanMessage(content=query)]}
            
            try:
                # Get state before turn to isolate this turn's messages
                state_before = graph.get_state(config)
                num_messages_before = len(state_before.values.get("messages", [])) if (state_before and state_before.values) else 0

                # Invoke the graph
                final_state = graph.invoke(initial_state, config=config)
                
                messages = final_state.get("messages", [])
                new_messages = messages[num_messages_before:]
                actual_intent = final_state.get("message_intent")
                
                # Extract Trajectory (tool calls and generated SQL) from new messages only
                tools_called = set()
                actual_sqls = []
                
                for msg in new_messages:
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc_info in msg.tool_calls:
                            tool_name = tc_info.get("name")
                            tools_called.add(tool_name)
                            # Extract SQL if query_database was called
                            if tool_name == "query_database":
                                sql = tc_info.get("args", {}).get("sql", "")
                                if sql:
                                    actual_sqls.append(sql)

                # Extract the final AI response Outcome
                final_response = ""
                if new_messages and isinstance(new_messages[-1], AIMessage):
                    final_response = new_messages[-1].content or ""

                # Evaluation Logic
                failure_reasons = []

                # 1. Check Intent
                if actual_intent != expected_intent:
                    failure_reasons.append(f"Intent mismatch: Expected '{expected_intent}', got '{actual_intent}'")

                # 2. Check Tool Call
                if expected_tool:
                    if expected_tool not in tools_called:
                        failure_reasons.append(f"Tool missing: Expected '{expected_tool}', but called {list(tools_called)}")
                else:
                    if tools_called:
                        failure_reasons.append(f"Unexpected tools called: {list(tools_called)}")

                # 3. Trajectory Check: SQL Keywords
                for sql_kw in expected_sql_keywords:
                    found_in_any = any(sql_kw.lower() in s.lower() for s in actual_sqls)
                    if not found_in_any:
                        failure_reasons.append(f"Trajectory Error: Missing SQL keyword '{sql_kw}'. Generated SQLs: {actual_sqls}")

                # 4. Outcome Check: Content Keywords
                missing_keywords = [kw for kw in expected_content_keywords if kw.lower() not in final_response.lower()]
                if missing_keywords:
                    failure_reasons.append(f"Outcome Error: Missing keywords in response: {missing_keywords}")

                # 5. Outcome Check: Prohibited Keywords (Anti-Hallucination)
                found_prohibited = [kw for kw in prohibited_keywords if kw.lower() in final_response.lower()]
                if found_prohibited:
                    failure_reasons.append(f"Silent Failure (Hallucination): Found prohibited keywords in response: {found_prohibited}")

                passed = len(failure_reasons) == 0
                if passed:
                    passed_turns += 1

                seq_results["turns"].append({
                    "query": query,
                    "passed": passed,
                    "actual_intent": actual_intent,
                    "tools_called": list(tools_called),
                    "actual_sqls": actual_sqls,
                    "final_response": final_response,
                    "failure_reasons": failure_reasons
                })

                if passed:
                    logger.info(f"    -> PASS")
                else:
                    logger.info(f"    -> FAIL: {failure_reasons}")

            except Exception as e:
                logger.error(f"    -> ERROR during execution: {str(e)}")
                seq_results["turns"].append({
                    "query": query,
                    "passed": False,
                    "actual_intent": "ERROR",
                    "tools_called": [],
                    "actual_sqls": [],
                    "final_response": f"Exception: {str(e)}",
                    "failure_reasons": ["Exception raised during graph execution"]
                })

        results.append(seq_results)

    # Generate Markdown Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Qurio Multi-Turn Agentic Evaluation Report\n\n")
        f.write(f"**Date Run:** {datetime.datetime.now().isoformat()}\n")
        f.write(f"**Total Sequences:** {len(dataset)}\n")
        f.write(f"**Total Turns Evaluated:** {total_turns_tested}\n")
        f.write(f"**Passed Turns:** {passed_turns}\n")
        accuracy = (passed_turns / total_turns_tested) * 100 if total_turns_tested > 0 else 0
        f.write(f"**Overall Accuracy:** {accuracy:.1f}%\n\n")

        f.write("## Test Sequences\n\n")
        for seq in results:
            f.write(f"### {seq['id']}\n")
            f.write(f"*{seq['description']}*\n\n")
            
            for i, turn in enumerate(seq['turns']):
                status_icon = "✅ PASS" if turn["passed"] else "❌ FAIL"
                f.write(f"#### Turn {i+1}: `{turn['query']}`\n")
                f.write(f"**Status:** {status_icon}\n\n")
                if not turn["passed"]:
                    f.write(f"**Failure Reasons:**\n")
                    for reason in turn["failure_reasons"]:
                        f.write(f"- {reason}\n")
                
                f.write(f"\n- **Intent:** `{turn['actual_intent']}`\n")
                f.write(f"- **Tools Called:** `{turn['tools_called']}`\n")
                if turn['actual_sqls']:
                    f.write(f"- **Generated SQL:**\n")
                    for s in turn['actual_sqls']:
                        f.write(f"  ```sql\n  {s}\n  ```\n")
                
                f.write("\n**Final Model Response:**\n")
                f.write(f"> {turn['final_response'].replace(chr(10), chr(10)+'> ')}\n\n")
            
            f.write("---\n")

    # Copy to latest_eval_report.md
    latest_report_path = REPORTS_DIR / "latest_eval_report.md"
    try:
        import shutil
        shutil.copy2(report_path, latest_report_path)
        logger.info(f"Report copied to static path: {latest_report_path}")
    except Exception as e:
        logger.error(f"Failed to copy report to static path: {str(e)}")

    logger.info(f"\nEvaluation complete. Report generated at: {report_path}")

if __name__ == "__main__":
    run_experiment()
