#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
import sys
import os

# python3 /home/matteo/franka_ros2_ws/src/simulation_rmp_1/scripts/analysis_simulations_rmp.py

# --------- tiny helpers (no psutil) ----------
def available_ram_gb():
    """Linux-friendly available RAM using /proc/meminfo; fallback to 1e9 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            kv = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k, v = parts
                    kv[k.strip()] = v.strip()
        # MemAvailable in kB
        avail_kb = int(kv.get("MemAvailable", "0 kB").split()[0])
        return avail_kb / (1024**2)
    except Exception:
        return 1.0  # best-effort fallback

def file_size_mb(p: Path):
    try:
        return p.stat().st_size / (1024**2)
    except Exception:
        return 0.0
# ---------------------------------------------

class SafeSimulationAnalyzer:
    def __init__(self, json_file_path):
        self.json_file_path = Path(json_file_path)
        self.data = None
        self.summary_df = None
        self.analysis_tables = {}  # Store all analytical tables

        # soft guardrails
        self.max_memory_mb = 2048   # ~2GB
        self.max_file_size_mb = 500 # suggest streaming for bigger, but we won't require ijson

        self.check_inputs()
        self.load_data()

    def check_inputs(self):
        if not self.json_file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file_path}")

        print(" System Check (no extra pkgs):")
        print(f"   Available RAM: {available_ram_gb():.1f} GB")
        print(f"   JSON file size: {file_size_mb(self.json_file_path):.1f} MB")
        if file_size_mb(self.json_file_path) > self.max_file_size_mb:
            print("   Note: Large file detected; this script loads whole JSON into RAM.")
            print("         If it fails, consider converting to JSONL/Parquet later.")

    def _normalize_raw(self, raw):
        """
        Normalize input to a list of per-run dicts.
        Supports:
        - list of runs
        - dict of {'simulation_XXXX': {...}}
        - dict with top-level list 'runs' or 'simulations'
        """
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            sims = [v for k, v in raw.items() if isinstance(k, str) and k.startswith("simulation_")]
            if sims:
                return sims
            for key in ("runs", "simulations", "data"):
                if key in raw and isinstance(raw[key], list):
                    return raw[key]
        raise ValueError("Unrecognized JSON structure: expected list of runs or dict of 'simulation_*' entries.")

    def load_data(self):
        print(f" Loading data from: {self.json_file_path}")
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.data = self._normalize_raw(raw)
        print(f" ✓ Loaded {len(self.data)} simulations")

    def _process_single_simulation(self, sim_data, sim_id=None):
        # Basic fields
        timestamp = sim_data.get("timestamp", "")
        exec_time = sim_data.get("execution_time", 0)
        n_obs = sim_data.get("num_obstacles", 0)

        # Goal-related
        goal_reached = bool(sim_data.get("goal_reached", False))
        goal_reach_time = sim_data.get("goal_reach_time", None)

        # Position checks
        pos_ok = sim_data.get("successful_position_checks", 0) or 0
        pos_cnt = sim_data.get("position_check_count", 0) or 0
        pos_rate = (pos_ok / pos_cnt) if pos_cnt > 0 else np.nan

        # Sub-objects
        dist = sim_data.get("distance_analysis", {}) or {}
        path = sim_data.get("path_metrics", {}) or {}
        joint = sim_data.get("joint_velocity_analysis", {}) or {}

        # Target position → distance from base
        target_pos = sim_data.get("target_position", [0, 0, 0]) or [0, 0, 0]
        if isinstance(target_pos, (list, tuple)) and len(target_pos) >= 3:
            try:
                base_dist = float(np.sqrt(sum(float(x) ** 2 for x in target_pos[:3])))
            except Exception:
                base_dist = np.nan
        else:
            base_dist = np.nan

        # Enhanced metrics
        trajectory_smoothness = path.get("average_curvature", np.nan)
        max_curvature = path.get("max_curvature", np.nan)
        
        # Execution efficiency 
        execution_efficiency = np.nan
        if goal_reach_time and exec_time and exec_time > 0:
            execution_efficiency = goal_reach_time / exec_time

        # Safety metrics
        danger_zone_percentage = 0
        warning_zone_percentage = 0
        safety_score = 100  # Default perfect score

        #Collision detection from raw distance data
        collision_count = 0
        collision_threshold = 0.01  # 1cm threshold for collision
        absolute_min_distance = float('inf')
        
        raw_distance_data = sim_data.get("raw_distance_data", [])
        if raw_distance_data:
            danger_count = sum(1 for entry in raw_distance_data 
                             if entry.get('overall_min_distance', float('inf')) < 0.50)
            warning_count = sum(1 for entry in raw_distance_data 
                              if entry.get('overall_min_distance', float('inf')) < 0.10)
            total_samples = len(raw_distance_data)
            
            #Count actual collisions and find absolute minimum distance
            collision_distances = []
            for entry in raw_distance_data:
                min_dist = entry.get('overall_min_distance', float('inf'))
                if min_dist != float('inf'):
                    absolute_min_distance = min(absolute_min_distance, min_dist)
                    if min_dist < collision_threshold:
                        collision_distances.append(min_dist)
                        collision_count += 1
            
            # If no finite distances found, reset to NaN
            if absolute_min_distance == float('inf'):
                absolute_min_distance = np.nan
        
            if total_samples > 0:
                danger_zone_percentage = (danger_count / total_samples) * 100
                warning_zone_percentage = (warning_count / total_samples) * 100
                safety_score = max(0, 100 - (danger_count / total_samples * 200))

        return {
            "simulation_id": sim_id,
            "timestamp": timestamp,
            "execution_time": exec_time,
            "num_obstacles": n_obs,
            "goal_reached": goal_reached,
            "goal_reach_time": goal_reach_time,
            "position_check_success_rate": pos_rate,
            "min_distance_achieved": dist.get("min_distance_achieved", np.nan),
            "avg_distance": dist.get("avg_distance", np.nan),
            "close_calls_5cm": dist.get("close_calls", 0),
            "safety_violations_2cm": dist.get("safety_violations", 0),
            "total_distance_traveled": path.get("total_distance_traveled", np.nan),
            "path_efficiency_ratio": path.get("path_efficiency_ratio", np.nan),
            "total_curvature": path.get("total_curvature", np.nan),
            "trajectory_smoothness": trajectory_smoothness,
            "max_curvature": max_curvature,
            "overall_max_velocity": joint.get("overall_max_velocity", np.nan),
            "overall_avg_velocity": joint.get("overall_avg_velocity", np.nan),
            "target_distance_from_base": base_dist,
            "execution_efficiency": execution_efficiency,
            "danger_zone_percentage": danger_zone_percentage,
            "warning_zone_percentage": warning_zone_percentage,
            "safety_score": safety_score,
            "collision_count": collision_count,
            "absolute_min_distance": absolute_min_distance,
            "had_collision": collision_count > 0,
        }

    def create_summary_dataframe(self):
        print(" Creating summary DataFrame...")
        rows = []
        # simple chunk-ish loop to allow GC
        for idx, run in enumerate(self.data, 1):
            rows.append(self._process_single_simulation(run, sim_id=idx))
            if idx % 1000 == 0:
                gc.collect()
        df = pd.DataFrame(rows)

        # enforce numeric dtypes where useful
        numeric_cols = [
            "execution_time", "num_obstacles", "goal_reach_time",
            "position_check_success_rate", "min_distance_achieved",
            "avg_distance", "close_calls_5cm", "safety_violations_2cm",
            "total_distance_traveled", "path_efficiency_ratio", "total_curvature",
            "trajectory_smoothness", "max_curvature",
            "overall_max_velocity", "overall_avg_velocity", "target_distance_from_base",
            "execution_efficiency", "danger_zone_percentage", "warning_zone_percentage", "safety_score", 
            "collision_count", "absolute_min_distance"
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        self.summary_df = df
        print(f" ✓ Summary DataFrame created: {len(self.summary_df)} rows")
        return self.summary_df

    def analyze_failure_patterns(self):
        """Analyze patterns in failed simulations"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        failed_df = self.summary_df[self.summary_df['goal_reached'] == False].copy()
        if len(failed_df) == 0:
            print("No failed simulations to analyze")
            return
        
        print(f"\n FAILURE PATTERN ANALYSIS ({len(failed_df)} failures)")
        print("="*60)
        
        # Create failure analysis tables
        failure_tables = {}
        
        # Collision analysis in failures
        if "collision_count" in failed_df.columns:
            failed_with_collisions = failed_df[failed_df["collision_count"] > 0]
            collision_summary = {
                'metric': ['Total failures', 'Failures with collisions', 'Collision rate in failures', 
                          'Avg collisions per failed run', 'Max collisions in failed run'],
                'value': [
                    len(failed_df),
                    len(failed_with_collisions),
                    f"{len(failed_with_collisions)/len(failed_df)*100:.1f}%",
                    f"{failed_with_collisions['collision_count'].mean():.1f}" if len(failed_with_collisions) > 0 else "0",
                    int(failed_with_collisions["collision_count"].max()) if len(failed_with_collisions) > 0 else 0
                ]
            }
            failure_tables['collision_analysis'] = pd.DataFrame(collision_summary)
            
            print(f" Collision involvement in failures:")
            print(f"   Failures with collisions: {len(failed_with_collisions)}/{len(failed_df)} ({len(failed_with_collisions)/len(failed_df)*100:.1f}%)")
            
            if len(failed_with_collisions) > 0:
                avg_collisions = failed_with_collisions["collision_count"].mean()
                max_collisions = failed_with_collisions["collision_count"].max()
                print(f"   Average collisions per failed run: {avg_collisions:.1f}")
                print(f"   Maximum collisions in a failed run: {int(max_collisions)}")

        # Failure by obstacle count
        obstacle_failure_data = []
        print(" Failure rates by obstacle complexity:")
        for n_obs in sorted(self.summary_df["num_obstacles"].dropna().unique()):
            total = len(self.summary_df[self.summary_df["num_obstacles"] == n_obs])
            failures = len(failed_df[failed_df["num_obstacles"] == n_obs])
            rate = (failures / total * 100) if total > 0 else 0
            print(f"   {int(n_obs)} obstacles: {failures}/{total} ({rate:.1f}% failure)")
            
            obstacle_failure_data.append({
                'obstacle_count': int(n_obs),
                'total_simulations': total,
                'failures': failures,
                'failure_rate_percent': rate
            })
        
        failure_tables['failure_by_obstacles'] = pd.DataFrame(obstacle_failure_data)
        
        # Distance analysis for failures
        if "min_distance_achieved" in failed_df.columns:
            valid_distances = failed_df["min_distance_achieved"].dropna()
            success_distances = self.summary_df[self.summary_df['goal_reached'] == True]["min_distance_achieved"].dropna()
            
            if len(valid_distances) > 0:
                distance_analysis = {
                    'metric': ['Avg min distance in failures (m)', 'Closest approach in failures (m)', 
                              'Avg min distance in successes (m)', 'Distance difference (m)'],
                    'value': [
                        f"{valid_distances.mean():.3f}",
                        f"{valid_distances.min():.3f}",
                        f"{success_distances.mean():.3f}" if len(success_distances) > 0 else "N/A",
                        f"{valid_distances.mean() - success_distances.mean():.3f}" if len(success_distances) > 0 else "N/A"
                    ]
                }
                failure_tables['distance_analysis'] = pd.DataFrame(distance_analysis)
                
                print(f"\n Distance characteristics of failures:")
                print(f"   Avg min distance in failures: {valid_distances.mean():.3f}m")
                print(f"   Closest approach in failures: {valid_distances.min():.3f}m")
                
                if len(success_distances) > 0:
                    print(f"   Avg min distance in successes: {success_distances.mean():.3f}m")
                    print(f"   Difference: {valid_distances.mean() - success_distances.mean():.3f}m")

        # Safety analysis for failures
        if "safety_score" in failed_df.columns:
            failed_safety = failed_df["safety_score"].dropna()
            success_safety = self.summary_df[self.summary_df['goal_reached'] == True]["safety_score"].dropna()
            
            if len(failed_safety) > 0:
                safety_analysis = {
                    'metric': ['Avg safety score in failures', 'Avg safety score in successes'],
                    'value': [
                        f"{failed_safety.mean():.1f}",
                        f"{success_safety.mean():.1f}" if len(success_safety) > 0 else "N/A"
                    ]
                }
                failure_tables['safety_analysis'] = pd.DataFrame(safety_analysis)
                
                print(f"\n Safety characteristics:")
                print(f"   Avg safety score in failures: {failed_safety.mean():.1f}")
                if len(success_safety) > 0:
                    print(f"   Avg safety score in successes: {success_safety.mean():.1f}")
        
        # Store failure analysis tables
        self.analysis_tables.update(failure_tables)

    def analyze_performance_metrics(self):
        """Analyze robot performance across different scenarios"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        df = self.summary_df
        print(f"\n PERFORMANCE METRICS ANALYSIS")
        print("="*60)
        
        performance_tables = {}
        
        # Path efficiency analysis
        if "path_efficiency_ratio" in df.columns:
            valid_efficiency = df["path_efficiency_ratio"].dropna()
            if len(valid_efficiency) > 0:
                success_eff = df[df['goal_reached'] == True]["path_efficiency_ratio"].dropna()
                fail_eff = df[df['goal_reached'] == False]["path_efficiency_ratio"].dropna()
                
                efficiency_data = {
                    'metric': ['Mean efficiency ratio', 'Best efficiency', 'Worst efficiency',
                              'Success avg efficiency', 'Failure avg efficiency'],
                    'value': [
                        f"{valid_efficiency.mean():.2f}",
                        f"{valid_efficiency.min():.2f}",
                        f"{valid_efficiency.max():.2f}",
                        f"{success_eff.mean():.2f}" if len(success_eff) > 0 else "N/A",
                        f"{fail_eff.mean():.2f}" if len(fail_eff) > 0 else "N/A"
                    ]
                }
                performance_tables['path_efficiency'] = pd.DataFrame(efficiency_data)
                
                print(f" Path Efficiency:")
                print(f"   Mean efficiency ratio: {valid_efficiency.mean():.2f}")
                print(f"   Best efficiency: {valid_efficiency.min():.2f} (closer to 1.0 is better)")
                print(f"   Worst efficiency: {valid_efficiency.max():.2f}")
                
                if len(success_eff) > 0 and len(fail_eff) > 0:
                    print(f"   Success avg efficiency: {success_eff.mean():.2f}")
                    print(f"   Failure avg efficiency: {fail_eff.mean():.2f}")
        
        # Velocity analysis
        if "overall_avg_velocity" in df.columns:
            valid_vel = df["overall_avg_velocity"].dropna()
            if len(valid_vel) > 0:
                velocity_data = {
                    'metric': ['Mean avg velocity (rad/s)', 'Max avg velocity (rad/s)', 'Min avg velocity (rad/s)'],
                    'value': [
                        f"{valid_vel.mean():.3f}",
                        f"{valid_vel.max():.3f}",
                        f"{valid_vel.min():.3f}"
                    ]
                }
                performance_tables['velocity_analysis'] = pd.DataFrame(velocity_data)
                
                print(f"\n Joint Velocity Analysis:")
                print(f"   Mean avg velocity: {valid_vel.mean():.3f} rad/s")
                print(f"   Max avg velocity: {valid_vel.max():.3f} rad/s")
                print(f"   Min avg velocity: {valid_vel.min():.3f} rad/s")
        
        # Trajectory smoothness
        if "trajectory_smoothness" in df.columns:
            valid_smooth = df["trajectory_smoothness"].dropna()
            if len(valid_smooth) > 0:
                smoothness_data = {
                    'metric': ['Mean curvature', 'Max curvature', 'Min curvature'],
                    'value': [
                        f"{valid_smooth.mean():.3f}",
                        f"{valid_smooth.max():.3f}",
                        f"{valid_smooth.min():.3f}"
                    ]
                }
                performance_tables['trajectory_smoothness'] = pd.DataFrame(smoothness_data)
                
                print(f"\n Trajectory Smoothness:")
                print(f"   Mean curvature: {valid_smooth.mean():.3f}")
                print(f"   Max curvature: {valid_smooth.max():.3f}")
                print(f"   Min curvature: {valid_smooth.min():.3f}")
        
        # Safety analysis
        total_close_calls = df["close_calls_5cm"].fillna(0).sum()
        total_violations = df["safety_violations_2cm"].fillna(0).sum()
        safety_scores = df["safety_score"].dropna()
        
        safety_data = {
            'metric': ['Total close calls (<5cm)', 'Total safety violations (<2cm)', 
                      'Safety violation rate (%)', 'Average safety score (/100)'],
            'value': [
                int(total_close_calls),
                int(total_violations),
                f"{total_violations/len(df)*100:.2f}",
                f"{safety_scores.mean():.1f}" if len(safety_scores) > 0 else "N/A"
            ]
        }
        performance_tables['safety_performance'] = pd.DataFrame(safety_data)
        
        print(f"\n Safety Performance:")
        print(f"   Total close calls (<5cm): {int(total_close_calls)}")
        print(f"   Total safety violations (<2cm): {int(total_violations)}")
        print(f"   Safety violation rate: {total_violations/len(df)*100:.2f}% of simulations")
        
        if len(safety_scores) > 0:
            print(f"   Average safety score: {safety_scores.mean():.1f}/100")
        
        # Store performance analysis tables
        self.analysis_tables.update(performance_tables)

    def create_correlation_analysis(self):
        """Analyze correlations between different metrics"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        numeric_cols = [
            "num_obstacles", "execution_time", "min_distance_achieved",
            "path_efficiency_ratio", "total_curvature", "overall_avg_velocity",
            "target_distance_from_base", "close_calls_5cm", "safety_violations_2cm",
            "trajectory_smoothness", "safety_score", "execution_efficiency"
        ]
        
        available_cols = [col for col in numeric_cols if col in self.summary_df.columns]
        if len(available_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return
        
        corr_data = self.summary_df[available_cols].corr()
        
        print(f"\n CORRELATION ANALYSIS")
        print("="*60)
        print(" Strong correlations (|r| > 0.5):")
        
        correlation_results = []
        found_correlations = False
        for i, col1 in enumerate(available_cols):
            for j, col2 in enumerate(available_cols[i+1:], i+1):
                corr_val = corr_data.loc[col1, col2]
                if abs(corr_val) > 0.5:
                    print(f"   {col1} ↔ {col2}: {corr_val:.3f}")
                    correlation_results.append({
                        'variable_1': col1,
                        'variable_2': col2,
                        'correlation': f"{corr_val:.3f}"
                    })
                    found_correlations = True
        
        if not found_correlations:
            print("   No strong correlations found (|r| > 0.5)")
            correlation_results.append({
                'variable_1': 'No strong correlations',
                'variable_2': 'found (|r| > 0.5)',
                'correlation': 'N/A'
            })
        
        self.analysis_tables['correlations'] = pd.DataFrame(correlation_results)

    def create_detailed_summary(self):
        """Enhanced summary with more detailed breakdowns"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        df = self.summary_df
        
        print("\n" + "="*80)
        print(" DETAILED SIMULATION ANALYSIS")
        print("="*80)
        
        # Basic stats
        basic_stats = {
            'metric': ['Total simulations', 'Successful simulations', 'Failed simulations', 
                      'Success rate (%)', 'Failure rate (%)'],
            'value': [
                len(df),
                df['goal_reached'].sum(),
                (~df['goal_reached']).sum(),
                f"{df['goal_reached'].mean()*100:.1f}",
                f"{(~df['goal_reached']).mean()*100:.1f}"
            ]
        }
        self.analysis_tables['basic_statistics'] = pd.DataFrame(basic_stats)
        
        print(f" Dataset Overview:")
        print(f"   Total simulations: {len(df)}")
        print(f"   Successful: {df['goal_reached'].sum()} ({df['goal_reached'].mean()*100:.1f}%)")
        print(f"   Failed: {(~df['goal_reached']).sum()} ({(~df['goal_reached']).mean()*100:.1f}%)")
        
        # Performance by complexity
        print(f"\n Success Rate by Scenario Complexity:")
        complexity_data = []
        
        for n_obs in sorted(df['num_obstacles'].dropna().unique()):
            sub_df = df[df['num_obstacles'] == n_obs]
            total = len(sub_df)
            success = sub_df['goal_reached'].sum()
            rate = sub_df['goal_reached'].mean() * 100
            avg_dist = sub_df['min_distance_achieved'].mean()
            avg_time = sub_df[sub_df['goal_reached'] == True]['goal_reach_time'].mean()
            
            complexity_data.append({
                'obstacle_count': int(n_obs),
                'total_sims': total,
                'successful_sims': success,
                'success_rate_percent': f"{rate:.1f}",
                'avg_min_distance_m': f"{avg_dist:.3f}" if not pd.isna(avg_dist) else "N/A",
                'avg_completion_time_s': f"{avg_time:.1f}" if not pd.isna(avg_time) else "N/A"
            })
            
            print(f"   {int(n_obs)} obstacles: {success}/{total} ({rate:.1f}%) | "
                  f"Avg dist: {avg_dist:.3f}m | Avg time: {avg_time:.1f}s")
        
        self.analysis_tables['complexity_analysis'] = pd.DataFrame(complexity_data)
        
        # Time analysis
        successful_times = df[df['goal_reached'] == True]['goal_reach_time'].dropna()
        if len(successful_times) > 0:
            time_stats = {
                'metric': ['Fastest completion (s)', 'Slowest completion (s)', 'Median completion (s)',
                          '25th percentile (s)', '75th percentile (s)'],
                'value': [
                    f"{successful_times.min():.2f}",
                    f"{successful_times.max():.2f}",
                    f"{successful_times.median():.2f}",
                    f"{successful_times.quantile(0.25):.2f}",
                    f"{successful_times.quantile(0.75):.2f}"
                ]
            }
            self.analysis_tables['completion_time_stats'] = pd.DataFrame(time_stats)
            
            print(f"\n Completion Time Analysis (successful runs only):")
            print(f"   Fastest: {successful_times.min():.2f}s")
            print(f"   Slowest: {successful_times.max():.2f}s")
            print(f"   Median: {successful_times.median():.2f}s")
            print(f"   25th percentile: {successful_times.quantile(0.25):.2f}s")
            print(f"   75th percentile: {successful_times.quantile(0.75):.2f}s")

    def save_analysis_report_csv(self, output_path=None):
        """Save all analytical tables to a comprehensive CSV report"""
        if not self.analysis_tables:
            print("No analysis tables to save. Run the analysis methods first.")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.json_file_path.parent / f"simulation_analysis_report_{timestamp}.csv"
        
        # Create a comprehensive report by combining all tables with section headers
        report_rows = []
        
        # Add metadata
        report_rows.append(['SIMULATION ANALYSIS REPORT', ''])
        report_rows.append(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        report_rows.append(['Source file:', str(self.json_file_path)])
        report_rows.append(['', ''])
        
        # Add each analysis table with headers
        for table_name, table_df in self.analysis_tables.items():
            # Add section header
            report_rows.append([f'=== {table_name.upper().replace("_", " ")} ===', ''])
            
            # Add table headers
            headers = list(table_df.columns)
            report_rows.append(headers)
            
            # Add table data
            for _, row in table_df.iterrows():
                report_rows.append(list(row))
            
            # Add separator
            report_rows.append(['', ''])
        
        # Convert to DataFrame and save
        max_cols = max(len(row) for row in report_rows)
        padded_rows = [row + [''] * (max_cols - len(row)) for row in report_rows]
        
        report_df = pd.DataFrame(padded_rows)
        report_df.to_csv(output_path, index=False, header=False)
        
        print(f" ✓ Analysis report CSV saved: {output_path}")
        print(f"   Contains {len(self.analysis_tables)} analysis sections")
        
        return output_path

    def save_failure_only_csv(self, output_path=None):
        """Create CSV with only failed simulations"""
        if self.summary_df is None:
            self.create_summary_dataframe()
        
        failed_df = self.summary_df[self.summary_df['goal_reached'] == False].copy()
        
        if len(failed_df) == 0:
            print(" No failed simulations found - cannot create failure CSV")
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.json_file_path.parent / f"simulation_failures_only_{len(failed_df)}_{timestamp}.csv"
        
        failed_df.to_csv(output_path, index=False)
        print(f" ✓ Failure-only CSV saved: {output_path}")
        print(f"   Contains {len(failed_df)} failed simulations out of {len(self.summary_df)} total")
        
        return output_path

    def save_summary_csv(self, output_path=None):
        if self.summary_df is None:
            self.create_summary_dataframe()
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.json_file_path.parent / f"simulation_summary_{len(self.summary_df)}_{timestamp}.csv"
        self.summary_df.to_csv(output_path, index=False)
        print(f" ✓ Summary CSV saved: {output_path}")
        return output_path

    def print_quick_summary(self):
        if self.summary_df is None:
            self.create_summary_dataframe()
        df = self.summary_df

        print("\n" + "="*60)
        print(" QUICK ANALYSIS SUMMARY")
        print("="*60)
        print(f" Dataset: {len(df)} simulations")
        if "goal_reached" in df.columns and len(df) > 0:
            print(f" Overall success: {df['goal_reached'].mean()*100:.1f}% ({df['goal_reached'].sum()}/{len(df)})")

        if {"num_obstacles", "goal_reached"}.issubset(df.columns):
            print("\n Success by obstacle complexity:")
            for n_obs in sorted(df["num_obstacles"].dropna().unique()):
                sub = df[df["num_obstacles"] == n_obs]
                rate = sub["goal_reached"].mean() * 100.0 if len(sub) else 0.0
                print(f"   {int(n_obs)} obstacles: {rate:5.1f}% ({sub['goal_reached'].sum():3d}/{len(sub):3d})")
                
        # Collision summary
        if "collision_count" in df.columns:
            collision_data = df["collision_count"].fillna(0)
            total_collisions = int(collision_data.sum())
            sims_with_collisions = len(df[df["collision_count"] > 0])
            print(f"\n Collision summary:")
            print(f"   Total collisions across all sims: {total_collisions}")
            print(f"   Simulations with collisions: {sims_with_collisions}/{len(df)} ({sims_with_collisions/len(df)*100:.1f}%)")
            if sims_with_collisions > 0:
                avg_collisions = collision_data[collision_data > 0].mean()
                max_collisions = int(collision_data.max())
                print(f"   Average collisions per affected sim: {avg_collisions:.1f}")
                print(f"   Maximum collisions in single sim: {max_collisions}")

        if "min_distance_achieved" in df.columns:
            valid = df["min_distance_achieved"].dropna()
            if len(valid) > 0:
                print("\n Safety summary:")
                print(f"   Closest approach: {valid.min():.3f} m")
                print(f"   Average min distance: {valid.mean():.3f} m")
            if "close_calls_5cm" in df.columns:
                print(f"   Close calls (<5cm): {int(df['close_calls_5cm'].fillna(0).sum())}")
            if "safety_violations_2cm" in df.columns:
                print(f"   Safety violations (<2cm): {int(df['safety_violations_2cm'].fillna(0).sum())}")

        if {"goal_reached", "goal_reach_time"}.issubset(df.columns):
            succ = df[(df["goal_reached"] == True) & (df["goal_reach_time"].notna())]
            if len(succ) > 0:
                print("\n Timing summary:")
                print(f"   Average completion time: {succ['goal_reach_time'].mean():.2f} s")
                print(f"   Fastest completion: {succ['goal_reach_time'].min():.2f} s")

        print("="*60)


def main():
    # change this default path if you want
    json_file_path = "/home/matteo/Simulation_rmp/Run_1000_2_260925/evaluation_results_1000_2.json"

    print("Starting ENHANCED simulation analysis...")
    try:
        analyzer = SafeSimulationAnalyzer(json_file_path)

        # Enhanced summary - this populates analysis_tables
        analyzer.create_detailed_summary()
        
        # New analyses - these also populate analysis_tables
        analyzer.analyze_failure_patterns()
        analyzer.analyze_performance_metrics() 
        analyzer.create_correlation_analysis()

        print("\n Continue with CSV generation? This will create:")
        print("   - Complete summary CSV (raw data)")
        print("   - Failures-only CSV (raw data)")
        print("   - Analysis report CSV (all printed insights)")
        response = input("Continue? [y/N]: ").lower().strip()

        if response in ("y", "yes"):
            # Create all CSV files
            complete_csv = analyzer.save_summary_csv()
            failure_csv = analyzer.save_failure_only_csv()
            analysis_report_csv = analyzer.save_analysis_report_csv()
            
            print("\n Analysis complete!")
            print(f" Complete summary: {complete_csv}")
            if failure_csv:
                print(f" Failures only: {failure_csv}")
            print(f" Analysis report: {analysis_report_csv}")
        else:
            print("\n Enhanced analysis complete! Re-run with 'y' to generate CSV files.")
            
    except Exception as e:
        print(f"\n Analysis failed: {e}")


if __name__ == "__main__":
    main()