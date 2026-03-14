def execute_rolling_evaluation(target_equity, forecast_start_date, total_steps, use_actuals=True, model_weights=None):
    mode_str = "ONE-STEP (using Actuals)" if use_actuals else "RECURSIVE ROLLING"
    print(f"\n--- Starting {total_steps}-Day {mode_str} Evaluation for {MODEL_TYPE} ---")
