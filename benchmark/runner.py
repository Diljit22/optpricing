import importlib
import sys
from time import perf_counter
from typing import List, Any, Dict

def run_single_technique(technique, config):
    """Runs all metrics for a single technique from the config."""
    results = {opt: {} for opt in config.options}
    timings = {opt: {} for opt in config.options}
    # Define the order of metrics
    metrics_to_run = config.metrics_to_run

    for opt in config.options:
        # --- Step 1: Price ---

        if "Price" not in metrics_to_run:
            continue # Should not happen

        t0 = perf_counter()
        # Pass technique_kwargs here for models like BlacksApprox
        pr = technique.price(opt, config.stock, config.model, config.rate, **config.technique_kwargs)
        timings[opt]["Price"] = perf_counter() - t0
        results[opt]["Price"] = pr.price

        # --- Step 2: Greeks and Implied Vol ---
        for metric in metrics_to_run:
            if metric == "Price":
                continue
            if metric == "ImpliedVol":
                method_to_call = technique.implied_volatility
            else:
                method_name = metric.lower()
            
            # Skip if the technique doesn't support the greek (e.g., a custom technique)
                if not hasattr(technique, method_name):
                    results[opt][metric] = float('nan')
                    timings[opt][metric] = 0.0
                    continue

                method_to_call = getattr(technique, method_name)
            
            t0 = perf_counter()
            
            # Handle the special case for implied_volatility
            if metric == "ImpliedVol":
                val = method_to_call(opt, config.stock, config.model, config.rate, target_price=pr.price, **config.technique_kwargs)
            else:
                val = method_to_call(opt, config.stock, config.model, config.rate, **config.technique_kwargs)
            
            timings[opt][metric] = perf_counter() - t0
            results[opt][metric] = val
            
    return results, timings

def run_from_config(config_module_path: str):
    """Loads a config, runs the benchmark, and prints the report."""
    from benchmark.report import print_benchmark_report
    
    try:
        config_module = importlib.import_module(config_module_path)
        config = config_module.config
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error: Could not load config from '{config_module_path}': {e}")
        sys.exit(1)

    all_results = []
    all_timings = []

    for tech, name in config.techniques:
        print(f"Running {name}...")
        results, timings = run_single_technique(tech, config)
        all_results.append(results)
        all_timings.append(timings)

    print_benchmark_report(config, all_results, all_timings)
    

def collect_benchmark_data(config):
    """
    Runs a benchmark config, calculating ONLY the price for each option/technique,
    and returns the collected data without printing a report.
    """
    #  will store results like: {technique_name: {option: {metric: value}}}
    collated_results = {}
    collated_timings = {}

    for tech, name in config.techniques:
        print(f"Running {name}...")
        
        collated_results[name] = {opt: {} for opt in config.options}
        collated_timings[name] = {opt: {} for opt in config.options}

        for opt in config.options:
            t0 = perf_counter()
            pr = tech.price(opt, config.stock, config.model, config.rate, **config.technique_kwargs)
            collated_timings[name][opt]["Price"] = perf_counter() - t0
            collated_results[name][opt]["Price"] = pr.price
            
    return collated_results, collated_timings

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m benchmark.runner <config_module_path>")
        print("Example: python -m benchmark.runner benchmark.configs.bsm_all_techs")
        sys.exit(1)
    
    run_from_config(sys.argv[1])