# src/llm_agents/architect.py

import os
import json
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from src.llm_agents.scout import ScoutAgent
from src.llm_agents.optimizer import OptimizerAgent
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client, log_to_global_file
from datetime import datetime

class ArchitectAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        
        # LLM Initialization
        self.client, self.model, self.temperature = get_llm_client()
        self.scout = ScoutAgent()
        self.optimizer = OptimizerAgent()

    def _save_readable_log(self, input_messages, raw_response, step_info="Architect"):
        """Save human-readable interaction logs (globally merged)"""
        try:
             # Format Input
             input_str = ""
             for m in input_messages:
                 role = str(m.get('role', 'unknown'))
                 content = str(m.get('content', ''))
                 if len(content) > 10000:
                     content = content[:10000] + "... [TRUNCATED]"
                 input_str += f"[{role.upper()}]:\n{content}\n\n"
             
             # Format Output
             resp_content = raw_response.choices[0].message.content if raw_response.choices else ""
             
             log_to_global_file("ArchitectAgent", input_str, resp_content, step_info)
        except Exception as e:
             print(f"⚠️ Log save failed: {e}")

    def generate_recipe(self, summary_report_json, planner_context=None, scout_limit=100000, scout_max_mw=None, initial_query=None, critic_feedback=None):
        """
        Generate standard operating procedure (SOP) in standard format
        """
        if planner_context is None:
            planner_context = {}
            
        print("\n🏗️ [Architect] Intelligence received, starting to formulate Standard Operating Procedure (SOP)...")
        if initial_query:
            print(f"🎯 User original requirement: {initial_query}")

        # --- 1. Split & Execute (Parallel Pathways) ---
        
        try:
            molecule_candidates_raw = self.scout.search_molecules(
                summary_report_json, 
                limit=scout_limit,
                max_mw=scout_max_mw,
                initial_query=initial_query
            )
        except Exception as e:
            print(f"❌ [Architect] Scout execution failed: {e}")
            molecule_candidates_raw = []

        if not molecule_candidates_raw:
            print("⚠️ [Architect] No candidate molecules found, will attempt to use a fallback strategy.")

        try:
            optimizer_result = self.optimizer.optimize(summary_report_json, molecule_candidates_raw)
        except Exception as e:
            print(f"❌ [Architect] Optimizer execution failed: {e}")
            optimizer_result = {}

        # --- 2. Merge (Synthesizing & Composition) ---
        
        molecule_candidates_optimized = optimizer_result.get("Molecules_With_Params", molecule_candidates_raw)
        process_params = {k:v for k,v in optimizer_result.items() if k != "Molecules_With_Params"}

        print("🔄 [Architect] Generating report based on the standard template...")
        
        try:
            key_feature = summary_report_json.get('critical_features_analysis', [{}])[0].get('feature_name', 'Key Chemical Group')
        except:
            key_feature = 'Target Functional Group'

        # Extract raw data from planner context
        raw_experiments = planner_context.get("retrieved_experiments", [])
        raw_features = planner_context.get("feature_importance_top5", {})
        
        # Load Template
        template = get_prompt('architect_agent_template')
        
        # Inject original design requirements and review feedback into the prompt
        user_goal_section = ""
        if initial_query:
            user_goal_section = f"""
### 🎯🚨 Original Design Requirements (CRITICAL - MUST FOLLOW)
**Below are the user's core design requirements, and your entire experimental recipe must directly address this goal:**

{initial_query}

⚠️ WARNING: Please ensure that the selected precursors, process parameters, and experimental conditions are designed to achieve the above user requirements!

---
"""

        if critic_feedback:
            user_goal_section += f"""
### ⚠️🚨 Expert Review Feedback & Past Failures (CRITICAL - MUST AVOID PAST MISTAKES)
**Below are the details of the previously rejected recipe and the expert's critique:**

{critic_feedback}

⚠️ WARNING: You must carefully read the above rejected [complete recipe], identify its specific flaws in parameters or mechanism, and implement **substantial quantitative improvements** in this new design. You MUST NOT generate the same ratio, time, temperature, or precursor combinations as the rejected recipe! Your new recipe must reflect thoughtful reflection on the previous failures.

---
"""
        
        # Fill Template
        prompt = user_goal_section + template.format(
            summary_report_json=json.dumps(summary_report_json, ensure_ascii=False),
            raw_features=json.dumps(raw_features, ensure_ascii=False, indent=2),
            raw_experiments=json.dumps(raw_experiments, ensure_ascii=False, indent=2),
            molecule_candidates_optimized=json.dumps(molecule_candidates_optimized[:10], ensure_ascii=False),
            process_params=json.dumps(process_params, ensure_ascii=False),
            recipe_strategy=process_params.get('Recipe_Strategy', 'Hydrothermal'),
            temperature=process_params.get('Temperature', '200 ℃'),
            time=process_params.get('Time', '8 h'),
            key_feature=key_feature
        )

        try:
            from src.llm_client import retry_with_backoff
            
            messages = [{"role": "user", "content": prompt}]
            response = retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2 
                ),
                max_retries=5,
                initial_wait=30,
                description=f"ArchitectAgent:{self.model}",
                input_messages=messages
            )
            
            # Save Readable Log
            self._save_readable_log(messages, response)
            
            recipe_content = response.choices[0].message.content
        except Exception as e:
            print(f"❌ [Architect] LLM report generation failed: {e}")
            return None
        
        # Save Report
        try:
            report_path = os.path.join(get_run_dir(), "Candidate_Recipe_Report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(recipe_content)
            print(f"✅ [Architect] Standard experimental recipe has been generated: {report_path}")
        except Exception as e:
            print(f"❌ [Architect] Failed to save report file: {e}")

        return recipe_content