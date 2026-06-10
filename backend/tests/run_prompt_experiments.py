import subprocess
import shutil
from pathlib import Path
import os
import sys

def main():
    tests_dir = Path(__file__).resolve().parent
    agent_dir = tests_dir.parent / "app" / "agent"
    reports_dir = tests_dir / "reports"
    prompts_dataset_dir = tests_dir / "datasets" / "prompts"
    
    variants = ["baseline", "markdown", "cot"]
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Running Experiment: {variant.upper()} PROMPT")
        print(f"{'='*60}")
        
        source_yaml = prompts_dataset_dir / f"prompts_{variant}.yaml"
        target_yaml = agent_dir / "prompts.yaml"
        
        if not source_yaml.exists():
            print(f"Error: {source_yaml} does not exist.")
            continue
            
        shutil.copy2(source_yaml, target_yaml)
        print(f"Swapped prompts.yaml with {source_yaml.name}")
        
        try:
            print("Executing orchestrator.py (this will take a few minutes)...")
            
            result = subprocess.run(
                [sys.executable, "orchestrator.py"],
                cwd=tests_dir,
            )
            
            if result.returncode != 0:
                print(f"Error running orchestrator for {variant} (exit code {result.returncode})")
            
            latest_report = reports_dir / "latest_eval_report.md"
            if latest_report.exists():
                new_report_name = reports_dir / f"eval_report_{variant}.md"
                shutil.copy2(latest_report, new_report_name)
                print(f"\nReport saved as {new_report_name.name}")
                
                with open(new_report_name, "r") as f:
                    for line in f:
                        if "**Passed Turns:**" in line or "**Overall Accuracy:**" in line:
                            print(f"[{variant.upper()}] {line.strip()}")
            else:
                print(f"Warning: latest_eval_report.md not found for {variant}")
                
        except Exception as e:
            print(f"Experiment failed for {variant}: {e}")

    print("\nRestoring baseline prompts.yaml...")
    shutil.copy2(prompts_dataset_dir / "prompts_baseline.yaml", agent_dir / "prompts.yaml")
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
