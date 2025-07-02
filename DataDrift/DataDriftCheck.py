import os
import json
import re
import pandas as pd
import mlflow
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

def sanitize(name: str) -> str:
    """
    Sanitizes metric names to conform to MLflow's allowed characters.
    Allows only alphanumerics, underscores (_), dashes (-), periods (.), spaces, and slashes.
    """
    safe = re.sub(r"[^0-9A-Za-z_\-./ /]", "_", name)
    return re.sub(r"_+", "_", safe).strip("_")

def generate_all_reports(
    reference_path="../data/Personal_Loan.csv",
    current_path="../data/newData.csv",
    target_col="Personal Loan",
    output_dir=".",
    mlflow_log=True
):
    # Ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data and drop unwanted columns
    ref_df = pd.read_csv(reference_path).drop(columns=[target_col, "ZIP Code"], errors="ignore")
    cur_df = pd.read_csv(current_path).drop(columns=[target_col, "ZIP Code"], errors="ignore")

    report_paths = {}
    run = None

    if mlflow_log:
        mlflow.set_experiment("Data Monitoring Reports")
        run = mlflow.start_run(run_name=f"Evidently_Report_{datetime.now():%Y%m%d_%H%M%S}")

    try:
        # --- Data Drift Report ---
        drift_report = Report([DataDriftPreset(method="psi")], include_tests=True)
        report1 = drift_report.run(reference_data=ref_df, current_data=cur_df)
        drift_html = os.path.join(output_dir, "datadrift_report.html")
        report1.save_html(drift_html)
        report_paths["Data Drift"] = drift_html

        # Parse and log drift metrics
        drift_dict = json.loads(report1.json())
        for entry in drift_dict.get("metrics", []):
            metric_id = sanitize(entry.get("metric_id") or entry.get("metric", "unknown"))
            val = entry.get("value")
            if isinstance(val, dict):
                for sub, sub_val in val.items():
                    if isinstance(sub_val, (int, float)):
                        mlflow.log_metric(f"{metric_id}_{sanitize(sub)}", sub_val)
            elif isinstance(val, (int, float)):
                mlflow.log_metric(metric_id, val)

        # --- Data Summary Report ---
        summary_report = Report([DataSummaryPreset()])
        report2 = summary_report.run(reference_data=ref_df, current_data=cur_df)
        summary_html = os.path.join(output_dir, "datasummary_report.html")
        report2.save_html(summary_html)
        report_paths["Data Summary"] = summary_html

        # Parse and log summary metrics
        summary_dict = json.loads(report2.json())
        for entry in summary_dict.get("metrics", []):
            metric_id = sanitize(entry.get("metric_id") or entry.get("metric", "unknown"))
            val = entry.get("value")
            if isinstance(val, dict):
                for sub, sub_val in val.items():
                    if isinstance(sub_val, (int, float)):
                        mlflow.log_metric(f"{metric_id}_{sanitize(sub)}", sub_val)
            elif isinstance(val, (int, float)):
                mlflow.log_metric(metric_id, val)

    finally:
        if run:
            mlflow.end_run()

    return report_paths
