"""
==========================================================================
 CD-LPL / BEACON Multi-Agent Discovery Pipeline - Unified Test Runner
==========================================================================
 This is an automated discovery pipeline for Carbon Dot materials 
 (BEACON: aBductive Extrapolation under Anchored CONstraints), supporting:
 1. Standard Full Pipeline (utilizing model defined at top of config.yaml)
 2. Component Ablation (disable Summary, Scout, or Critic loops)
 3. Cross-Model Ablation (benchmarking multiple LLMs on Scout dependency)
 4. Checkpoint Resume (resume from a specific iteration)

 🚀 Running Instructions:
 -----------------------
 (1) Single Run (Standard Full Pipeline):
     Command: python test_runner.py --config full
     - Uses default model from config.yaml.
     - Saves intermediate outputs in Iteration_X folders.

 (2) Ablation Suite (Single Model):
     Command: python test_runner.py --config ablation_all
     - Runs 3 configurations sequentially: full, no_summary, no_scout.
     - Saves outputs in config_xxxx subdirectories.

 (3) Cross-Model Multi-Ablation Study:
     Command: python test_runner.py --config model_ablation
     - Reads all models configured in config.yaml's model_ablation_pool.
     - Runs 3 configurations for each model: full, no_summary, no_scout.
     - Prints a consolidated cross-ablation summary table at the end.

 (4) Checkpoint Resume:
     Command: python test_runner.py --resume-dir "./experiments/DIR" --resume-iter 3
"""

import json
import os
import re
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

from src.utils import get_run_dir, set_run_subdir, get_base_run_dir, ConfigLoader, get_llm_client, get_prompt, get_llm_config
from src.llm_agents.planner import PlannerAgent
from src.llm_agents.deep_analysis_tool import DeepAnalysisRunner
from src.llm_agents.summary import SummaryAgent
from src.llm_agents.architect import ArchitectAgent
from src.llm_agents.critic import CriticAgent
from src.llm_client import retry_with_backoff, TokenTracker

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)

# =========================================================
# Hardware Optimization Setup
# =========================================================
_cpu_count = str(os.cpu_count() or 28)
os.environ['OMP_NUM_THREADS'] = _cpu_count
os.environ['MKL_NUM_THREADS'] = _cpu_count
os.environ['NUMEXPR_NUM_THREADS'] = _cpu_count
os.environ['OPENBLAS_NUM_THREADS'] = _cpu_count

try:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    kernel32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS
    print(f"⚡ Process Priority: HIGH | CPU Threads: {_cpu_count} | BLAS Threads: {_cpu_count}")
except Exception:
    pass

# =========================================================
# Global Default Parameters
# =========================================================
MOLECULE_SEARCH_LIMIT = 500000   # Scout molecule search space limit
MOLECULE_MAX_MW = 500            # Precursor molecular weight ceiling

def clean_json_str(content):
    """Clean the JSON string output by LLM, handling control characters and other common issues"""
    if not isinstance(content, str):
        return "{}"
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    # Remove control characters that would break json.loads (0x00-0x1F) while keeping tabs/newlines
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
    return content

def _count_cid_smiles():
    """Count total molecules"""
    try:
        conf = ConfigLoader.get_instance()
        cid_path = conf.get_data_path('cid_smiles')
        if cid_path and os.path.exists(cid_path):
            with open(cid_path, 'rb') as f:
                total_molecules = 0
                while True:
                    buffer = f.read(1024 * 1024 * 4) # 4MB chunks
                    if not buffer: break
                    total_molecules += buffer.count(b'\n')
            return total_molecules
    except Exception:
        pass
    return 0

def _generate_recipe_without_scout_optimizer(summary_report, planner_context, initial_query, critic_feedback):
    """
    Alternative approach when Scout+Optimizer are ablated:
    skip molecule retrieval and recipe optimization, and let LLM propose a recipe directly based on Summary rules.
    """
    client, model, temperature = get_llm_client()
    template = get_prompt('architect_agent_template')

    dummy_molecules = [{"Note": "No Scout search performed (ablated). Please propose candidate molecules based on the design rules alone."}]
    dummy_params = {"Recipe_Strategy": "To be determined by LLM", "Temperature": "To be determined", "Time": "To be determined"}

    raw_experiments = planner_context.get("retrieved_experiments", []) if planner_context else []
    raw_features = planner_context.get("feature_importance_top5", {}) if planner_context else {}

    try: key_feature = summary_report.get('critical_features_analysis', [{}])[0].get('feature_name', 'Key Substructure')
    except: key_feature = 'Target Functional Group'

    user_goal_section = ""
    if initial_query:
        user_goal_section = f"### 🎯🚨 Original Design Requirements (CRITICAL - MUST FOLLOW)\n{initial_query}\n---\n"
    if critic_feedback:
        user_goal_section += f"### ⚠️🚨 Previous Review Feedback\n{critic_feedback}\n---\n"

    prompt = user_goal_section + template.format(
        summary_report_json=json.dumps(summary_report, ensure_ascii=False),
        raw_features=json.dumps(raw_features, ensure_ascii=False, indent=2),
        raw_experiments=json.dumps(raw_experiments, ensure_ascii=False, indent=2),
        molecule_candidates_optimized=json.dumps(dummy_molecules, ensure_ascii=False),
        process_params=json.dumps(dummy_params, ensure_ascii=False),
        recipe_strategy="TBD (No Scout)",
        temperature="TBD",
        time="TBD",
        key_feature=key_feature
    )

    try:
        ablation_msgs = [{"role": "user", "content": prompt}]
        response = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model, messages=ablation_msgs, temperature=0.2
            ),
            max_retries=5, initial_wait=10, description=f"AblationArchitect:{model}",
            input_messages=ablation_msgs
        )
        recipe_content = response.choices[0].message.content
        report_path = os.path.join(get_run_dir(), "Candidate_Recipe_Report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(recipe_content)
        return recipe_content
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        return None

# ==========================================================================
# Unified Pipeline Function
# ==========================================================================

def run_pipeline(
    config_name="Full_Pipeline",
    initial_query="Help me design a near-infrared emitting carbon dot-based long afterglow material with an afterglow emission wavelength exceeding 750 nm.",
    ablation_flags=None,
    max_iterations=5,
    resume_dir=None,
    resume_iter=None
):
    """
    General execution engine: supports standard runs, resume from checkpoint, and module ablation.
    """
    if ablation_flags is None:
        ablation_flags = {"use_summary": True, "use_scout_optimizer": True, "use_critic_loop": True}

    use_summary = ablation_flags.get("use_summary", True)
    use_scout_optimizer = ablation_flags.get("use_scout_optimizer", True)
    use_critic_loop = ablation_flags.get("use_critic_loop", True)

    removed = []
    if not use_summary: removed.append("Summary")
    if not use_scout_optimizer: removed.append("Scout+Optimizer")
    if not use_critic_loop: removed.append("Critic")
    removed_str = ", ".join(removed) if removed else "None (Full Pipeline)"

    print("\n" + "=" * 70)
    print(f"🚀 [BEACON Execution Engine] Config: {config_name}")
    print(f"   Ablated Modules: {removed_str}")
    print(f"   Target Iterations: {max_iterations}")
    print("=" * 70)

    result = {
        "config_name": config_name,
        "removed_modules": removed,
        "iterations_completed": 0,
        "final_avg_score": 0,
        "final_pass_count": 0,
        "final_passed": False,
        "score_history": [],
        "error": None,
    }

    current_query = initial_query
    last_critic_feedback = None
    rejected_history = []
    start_iteration = 0

    # ==========================================
    # Checkpoint Resume Logic
    # ==========================================
    if resume_dir and resume_iter:
        print(f"\n🔄 Resuming execution from checkpoint...")
        print(f"📁 Target Directory: {resume_dir}")
        print(f"🔢 Base Iteration: Round {resume_iter} (New run will start from Round {resume_iter + 1})")
        
        ConfigLoader.get_instance()._run_dir = os.path.abspath(resume_dir)
        
        token_csv_path = os.path.join(resume_dir, config_name, "Token_and_Cost_Summary.csv")
        if os.path.exists(token_csv_path):
            print(f"📊 Merging token usage stats from history: {token_csv_path}")
            TokenTracker.get_instance().merge_from_csv(token_csv_path)
        
        base_resume_dir = resume_dir
        if config_name not in resume_dir: 
            check_path = os.path.join(resume_dir, config_name)
            if os.path.exists(check_path):
                base_resume_dir = check_path
        
        for i in range(1, resume_iter + 1):
            iter_dir = os.path.join(base_resume_dir, f"Iteration_{i}")
            recipe_path = os.path.join(iter_dir, "Candidate_Recipe_Report.md")
            summary_path = os.path.join(iter_dir, "final_summary_report.json")
            critic_summary_path = os.path.join(iter_dir, "Critic_Reviews", f"Round_{i}", "Round_Summary.json")
            
            if not (os.path.exists(recipe_path) and os.path.exists(critic_summary_path)):
                print(f"⚠️ Fatal Error: History files for iteration {i} are incomplete, resume failed.")
                print(f"  Missing: {recipe_path} or {critic_summary_path}")
                return result
                
            with open(recipe_path, "r", encoding="utf-8") as f: recipe_content = f.read()
            summary_report = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f: 
                        summary_report = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Warning: JSON decode error while reading {summary_path}: {e}")
                    summary_report = {}
            
            critic_data = {}
            try:
                with open(critic_summary_path, "r", encoding="utf-8") as f: 
                    critic_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Serious Warning: Failed to read {critic_summary_path} (JSON parse error): {e}")
                print(f"⚠️ Skipping feedback concatenation for iteration {i}. Highly recommend checking the file contents!")
                critic_data = {"details": [], "avg_score": 0, "pass_count": 0}
            
            current_round_feedback = f"Feedback for Iteration {i}:\n"
            for res in critic_data.get('details', []):
                current_round_feedback += f"- [{res.get('judge_model')}]: {res.get('score')} - {'Valid' if res.get('is_reasonable') else 'Invalid'} - {res.get('reason')}\n"
                
            rejected_history.append({"iteration": i, "recipe": recipe_content, "feedback": current_round_feedback, "summary_context": json.dumps(summary_report, ensure_ascii=False)})
            result["score_history"].append({"iteration": i, "avg_score": critic_data.get('avg_score', 0), "passed": critic_data.get('pass_count', 0) >= 6})
            
        print(f"✅ Successfully extracted feedback from {len(rejected_history)} prior rounds.")
        
        consolidated_feedback = "[⚠️ Cumulative Failure History]\nThe following plans were REJECTED in previous iterations. You MUST verify why they failed, review the exact recipe proposed, and make concrete improvements based on the feedback.\n\n"
        for hist in rejected_history:
            consolidated_feedback += f"=== Rejected Round {hist['iteration']} ===\n[Full Rejected Recipe]:\n{hist['recipe']}\n\n[Reviewer Feedback]:\n{hist['feedback']}\n" + "-"*40 + "\n"
            
        current_query = consolidated_feedback
        last_critic_feedback = consolidated_feedback
        start_iteration = resume_iter
        
    else:
        ConfigLoader.get_instance().base_run_dir

    print(f"📂 Experiment base directory: {get_base_run_dir()}")
    
    iteration = start_iteration
    while iteration < max_iterations:
        iteration += 1
        
        if config_name: 
            set_run_subdir(os.path.join(config_name, f"Iteration_{iteration}"))
        else:
            set_run_subdir(f"Iteration_{iteration}")

        print(f"\n🔄 [{config_name}] Iteration {iteration}/{max_iterations}")
        print(f"📂 Round output subdirectory: {get_run_dir()}")

        try:
            # =================================================================
            # Step 1: Planner Agent (Always runs)
            # =================================================================
            print("\n---------------------------------------------------------------")
            print("🧠 [Step 1] Planner Agent")
            print("---------------------------------------------------------------")
            planner = PlannerAgent()
            query_with_goal = f"[🎯 Original Design Requirements (MUST FOLLOW)]\n{initial_query}\n\n---\n\n{current_query}"
            
            planner_json_str, planner_context = planner.run(query_with_goal)
            planner_data = json.loads(clean_json_str(planner_json_str), strict=False)

            if "key_bits_to_decode" not in planner_data or not planner_data["key_bits_to_decode"]:
                print("  ⚠️ Planner produced no key_bits_to_decode, terminating this iteration.")
                result["error"] = "Planner produced no key_bits_to_decode"
                break

            # =================================================================
            # Step 2: Deep Analysis (Cannot be ablated)
            # =================================================================
            print("\n---------------------------------------------------------------")
            print("⛏️ [Step 2] Deep Analysis")
            print("---------------------------------------------------------------")
            analyzer = DeepAnalysisRunner()
            analysis_result = analyzer.analyze(planner_data)
            status = analysis_result.get('status')
            output_dirs = analysis_result.get('output_dirs', [])
            
            if status not in ('success', 'partial_success'):
                print("  ❌ Deep Analysis failed.")
                result["error"] = "DeepAnalysis failed"
                break

            # =================================================================
            # Step 3: Summary Agent (Ablatable)
            # =================================================================
            print("\n---------------------------------------------------------------")
            print("👁️ [Step 3] Summary Agent")
            print("---------------------------------------------------------------")
            if use_summary:
                summarizer = SummaryAgent()
                summary_report = summarizer.summarize(
                    planner_data, output_dirs, critic_feedback=last_critic_feedback, initial_query=initial_query
                )
                if not summary_report or "error" in summary_report:
                    print("  ❌ Summary generation failed.")
                    result["error"] = "Summary failed"
                    break
            else:
                print("  ⏭️ [Skipped] Only using raw data structures...")
                raw_bits = planner_data.get("key_bits_to_decode", [])
                feature_dicts = [{"feature_name": b, "chemical_meaning": f"chemical structure corresponding to {b}"} for b in raw_bits]
                summary_report = {
                    "target_property": planner_data.get("target_property", "emission"),
                    "critical_structures": feature_dicts,
                    "critical_features_analysis": feature_dicts,
                    "design_guidelines": {"structural_rules": [], "process_rules": []},
                }
                
                with open(os.path.join(get_run_dir(), "final_summary_report.json"), "w", encoding="utf-8") as f:
                    json.dump(summary_report, f, ensure_ascii=False, indent=2)

            # =================================================================
            # Step 4: Architect Agent (Scout+Optimizer Ablatable)
            # =================================================================
            print("\n---------------------------------------------------------------")
            print("🏗️ [Step 4] Architect Agent")
            print("---------------------------------------------------------------")
            if use_scout_optimizer:
                architect = ArchitectAgent()
                recipe_content = architect.generate_recipe(
                    summary_report_json=summary_report,
                    planner_context=planner_context,
                    scout_limit=MOLECULE_SEARCH_LIMIT,
                    scout_max_mw=MOLECULE_MAX_MW,
                    initial_query=initial_query,
                    critic_feedback=last_critic_feedback
                )
            else:
                print("  ⏭️ [Skipped] Skipping Scout & Optimizer")
                recipe_content = _generate_recipe_without_scout_optimizer(
                    summary_report, planner_context, initial_query, last_critic_feedback
                )

            if not recipe_content:
                print("  ❌ Recipe generation failed.")
                result["error"] = "Recipe generation failed"
                break

            # =================================================================
            # Step 5: Critic Agent (Ablatable)
            # =================================================================
            print("\n---------------------------------------------------------------")
            print("⚖️ [Step 5] Critic Agent")
            print("---------------------------------------------------------------")
            critic_evaluator = CriticAgent()
            review_result = critic_evaluator.evaluate(
                recipe_content, summary_report, iteration=iteration, log_dir=get_run_dir(), initial_query=initial_query
            )

            avg_score = review_result.get('avg_score', 0)
            pass_count = review_result.get('pass_count', 0)
            passed = review_result.get('passed', False)
            
            result["iterations_completed"] = iteration
            result["final_avg_score"] = avg_score
            result["final_pass_count"] = pass_count
            result["final_passed"] = passed
            result["score_history"].append({
                "iteration": iteration, "avg_score": round(avg_score, 2), "pass_count": pass_count
            })

            if not use_critic_loop:
                print("  ⏭️ [Skipped] Skipping Critic loop. Terminating immediately after first iteration.")
                break

            if passed:
                print(f"🎉 Recipe approved by jury! (Votes: {pass_count}/7, Average Score: {avg_score:.2f})")
                print(f"📁 Final successful recipe directory: {get_run_dir()}")
                break
            else:
                print(f"⚠️ Recipe rejected (Votes: {pass_count}/7, Average Score: {avg_score:.2f}). Proceeding to next round...")
                current_round_feedback = f"Feedback for Iteration {iteration}:\n"
                for res in review_result.get('details', []):
                    current_round_feedback += f"- [{res.get('judge_model')}]: {res.get('score')} - {'Valid' if res.get('is_reasonable') else 'Invalid'} - {res.get('reason')}\n"
                
                rejected_history.append({
                    "iteration": iteration,
                    "recipe": recipe_content,
                    "feedback": current_round_feedback
                })

                consolidated_feedback = "[⚠️ Cumulative Failure History]\nThe following plans were REJECTED in previous iterations. You MUST verify why they failed, review the exact recipe proposed, and make concrete improvements based on the feedback.\n\n"
                for hist in rejected_history:
                    consolidated_feedback += f"=== Rejected Round {hist['iteration']} ===\n[Full Rejected Recipe]:\n{hist['recipe']}\n\n[Reviewer Feedback]:\n{hist['feedback']}\n" + "-"*40 + "\n"
                
                current_query = consolidated_feedback
                last_critic_feedback = consolidated_feedback

        except Exception as e:
            print(f"❌ Serious exception in this round: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            break
            
        print("\n===============================================================")
        print(f"📊 [Iteration {iteration}] Token Usage Summary & Real-time Status")
        print("===============================================================")
        TokenTracker.get_instance().print_summary()
        TokenTracker.get_instance().save_to_csv(os.path.join(get_base_run_dir(), config_name))

    # Save final config summary results
    config_result_path = os.path.join(get_base_run_dir(), config_name, "config_run_result.json")
    os.makedirs(os.path.dirname(config_result_path), exist_ok=True)
    with open(config_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    tracker = TokenTracker.get_instance()
    tracker.print_summary()
    tracker.save_to_csv(os.path.join(get_base_run_dir(), config_name))
        
    return result


# ==========================================================================
# Command Line Integration Entry Point
# ==========================================================================

def _switch_model(model_name, base_url):
    """Dynamically switch models at runtime (overwrites openai config in active memory)"""
    conf = ConfigLoader.get_instance()
    conf.config['llm']['provider'] = 'openai'
    conf.config['llm']['openai'] = {
        'model_name': model_name,
        'temperature': 0.1,
        'base_url': base_url,
    }
    print(f"\n🔄 [Model Switch] Switched to model: {model_name} (via {base_url})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEACON Unified System Runner")
    parser.add_argument("--query", "-q", type=str, default="Help me design a near-infrared emitting carbon dot-based long afterglow material with an afterglow emission wavelength exceeding 750 nm.", help="Original design query of the user")
    parser.add_argument("--config", "-c", type=str, choices=["full", "no_summary", "no_scout", "ablation_all", "model_ablation"], default="full", help="Pipeline execution mode configuration")
    parser.add_argument("--max-iter", "-m", type=int, default=5, help="Maximum iteration ceiling for single pipeline run")
    parser.add_argument("--resume-dir", type=str, default=None, help="Absolute path to historical run directory for checkpoint resume")
    parser.add_argument("--resume-iter", type=int, default=None, help="The final completed iteration index from which to resume")
    
    args = parser.parse_args()

    # Define ablation presets
    all_configs = {
        "full": {"name": "config_full", "flags": {"use_summary": True, "use_scout_optimizer": True, "use_critic_loop": True}},
        "no_summary": {"name": "config_no_summary", "flags": {"use_summary": False, "use_scout_optimizer": True, "use_critic_loop": True}},
        "no_scout": {"name": "config_no_scout_optimizer", "flags": {"use_summary": True, "use_scout_optimizer": False, "use_critic_loop": True}},
    }

    print("=================================================")
    print(" 🚀 BEACON (aBductive Extrapolation under Anchored CONstraints) Launched")
    print("=================================================")
    _count_cid_smiles()
    
    start_global = time.time()

    # ==================================================================
    # Multi-Model Ablation Mode (Cross-Ablation Study)
    # ==================================================================
    if args.config == "model_ablation":
        conf = ConfigLoader.get_instance()
        model_pool = conf.config.get('model_ablation_pool', [])
        
        if not model_pool:
            print("❌ No model_ablation_pool found in config.yaml!")
            exit(1)
        
        print(f"\n{'='*70}")
        print(f" 🔬 Cross-Model Ablation Mode (Full)")
        print(f" Models Count: {len(model_pool)} | 3 configs per model (full, no_summary, no_scout)")
        print(f" Total Tasks: {len(model_pool) * 3} pipeline runs")
        print(f"{'='*70}\n")
        
        for i, m in enumerate(model_pool, 1):
            print(f"  [{i}] {m['alias']}: {m['model_name']}")
        print()
        
        all_results = []
        
        for model_idx, model_conf in enumerate(model_pool, 1):
            alias = model_conf['alias']
            model_name = model_conf['model_name']
            base_url = model_conf.get('base_url', 'https://api.poe.com/v1')
            
            print(f"\n{'#'*70}")
            print(f" 📦 Evaluating Model [{model_idx}/{len(model_pool)}]: {model_name}")
            print(f"{'#'*70}")
            
            _switch_model(model_name, base_url)
            TokenTracker.reset()
            conf.set_run_dir()
            
            for ab_key, ab_conf in all_configs.items():
                config_label = f"{alias}_{ab_conf['name']}"
                
                print(f"\n{'='*60}")
                print(f" 🏃 [{alias}] Running config: {ab_key}")
                print(f"{'='*60}")
                
                res = run_pipeline(
                    config_name=config_label,
                    initial_query=args.query,
                    ablation_flags=ab_conf["flags"],
                    max_iterations=args.max_iter,
                )
                res['model_alias'] = alias
                res['model_name'] = model_name
                res['ablation_type'] = ab_key
                all_results.append(res)
            
            print(f"\n📊 [{alias}] Evaluation complete for all configs of this model.")
            TokenTracker.get_instance().print_summary()
            TokenTracker.get_instance().save_to_csv(conf.base_run_dir)
        
        # consolidated cross-ablation summary table
        print(f"\n{'='*90}")
        print(f" 📊 Cross-Model Ablation Comparison Table")
        print(f"{'='*90}")
        print(f"  {'Model':<15} {'Ablation Mode':<20} {'Pass':>5} {'Iter':>5} {'AvgScore':>10}")
        print(f"  {'-'*15} {'-'*20} {'-'*5} {'-'*5} {'-'*10}")
        for r in all_results:
            status = "✅" if r.get('final_passed') else "❌"
            print(f"  {r['model_alias']:<15} {r['ablation_type']:<20} {status:>5} {r['iterations_completed']:>5} {r['final_avg_score']:>10.2f}")
        print(f"{'='*90}\n")
        
        summary_path = os.path.join(conf.base_run_dir, "model_ablation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Cross-ablation comparison saved: {summary_path}")
        
    else:
        # ==================================================================
        # Standard Single / Multi-Ablation configurations
        # ==================================================================
        configs_to_run = []
        if args.config == "ablation_all":
            configs_to_run = list(all_configs.values())
        else:
            configs_to_run = [all_configs[args.config]]
        
        results = []
        for cfg in configs_to_run:
            res = run_pipeline(
                config_name=cfg["name"],
                initial_query=args.query,
                ablation_flags=cfg["flags"],
                max_iterations=args.max_iter,
                resume_dir=args.resume_dir,
                resume_iter=args.resume_iter
            )
            results.append(res)
        
        print(f"\n✅ All tasks finished! Total time elapsed: {time.time() - start_global:.1f} s")
        print("\n📊 Results Summary:")
        for r in results:
            status = "✅ Pass" if r.get('final_passed') else "❌ Failed"
            print(f" - [{r['config_name']}]: {status} (Iterations: {r['iterations_completed']}, Avg Score: {r['final_avg_score']:.2f})")
        
        tracker = TokenTracker.get_instance()
        if tracker.total_tokens > 0:
            tracker.print_summary()
            tracker.save_to_csv(get_base_run_dir())
    
    print(f"\n🏁 Complete! Total time elapsed: {time.time() - start_global:.1f} s")
