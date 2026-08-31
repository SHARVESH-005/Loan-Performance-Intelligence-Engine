import argparse
import sys
from src.data import profiler, advanced_profiler
from src.features import engineer
from src.models import baseline, submission, logistic_baseline, improved, calibration, metrics_report
from src.survival import discrete_hazard
from src.anomaly import detector
from src.scenario import simulator
from src.explainability import explainer
from src.llm_copilot import copilot

def main():
    parser = argparse.ArgumentParser(description="Run the Loan Performance Intelligence Engine pipeline.")
    parser.add_argument("--stage", type=str, default="all", help="Pipeline stage to run (day1, day2, all)")
    args = parser.parse_args()

    print(f"Running pipeline stage: {args.stage}")
    
    if args.stage in ['day1', 'day2', 'all']:
        print("=== Step 1: Data Profiling ===")
        profiler.generate_report()
        
        print("\n=== Step 2: Feature Engineering ===")
        engineer.run()
        
    if args.stage in ['day2', 'all']:
        print("\n=== Step 2.5: Advanced Profiling (Day 2) ===")
        advanced_profiler.run()
        
    if args.stage in ['day1']:
        print("\n=== Step 3: Baseline Models ===")
        baseline.run()
        
    if args.stage in ['day2', 'all']:
        print("\n=== Step 3.1: Logistic Regression Baselines ===")
        logistic_baseline.run()
        
        print("\n=== Step 3.2: Improved LightGBM Tuning ===")
        improved.run()
        
        print("\n=== Step 3.3: Probability Calibration ===")
        calibration.run()
        
        print("\n=== Step 3.4: §12 Metrics Report ===")
        metrics_report.run()
        
    if args.stage in ['day3', 'all']:
        print("\n=== Step 5: Survival Modeling (FR-3) ===")
        discrete_hazard.run()
        
        print("\n=== Step 6: Anomaly & Exception Detection (FR-4) ===")
        detector.run()

    if args.stage in ['day4', 'all']:
        print("\n=== Step 8: Scenario Simulation (FR-5) ===")
        simulator.run()
        
        print("\n=== Step 9: Explainability (FR-6) ===")
        explainer.run()
        
        print("\n=== Step 10: LLM Copilot (FR-7) ===")
        copilot.run()

    if args.stage in ['day1', 'day2', 'day3', 'day4', 'all']:
        print("\n=== Step 11: Submission ===")
        submission.run()

if __name__ == "__main__":
    main()
