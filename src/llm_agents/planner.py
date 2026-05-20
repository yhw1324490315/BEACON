# src/llm_agents/planner.py
#
# Detailed comments have been added inside to help readability and maintenance.

import os
import json
import re
import base64
from openai import OpenAI
from dotenv import load_dotenv
from src.llm_agents.data_tools import DataToolkit
from datetime import datetime
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client, log_to_global_file

# ----------------------------
# Environment and Configuration
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)


class PseudoToolCall:
    """
    Wrap the implicit tool call structure parsed from the model in a lightweight object.

    This class does not actually make remote calls, but converts the parsed JSON structure into a form similar to
    an openai tool call (containing id, function.name, function.arguments, etc.) object for convenient
    unified interaction with downstream processing logic (e.g., messages list).
    """
    def __init__(self, data):
        self.id = data['id']
        self.function = type('obj', (object,), data['function'])
        self.type = 'function'


class PlannerAgent:
    """
    PlannerAgent's responsibility:
      - Receive user query (user_query)
      - Call LLM to obtain the tool calls to execute (e.g., get experimental data / query feature importance)
      - Execute these tools through DataToolkit and inject the results into conversation history
      - Request LLM to generate the final JSON report

    Design description:
      - Use get_llm_client() to encapsulate LLM client creation (compatible with various LLM clients)
      - Use ConfigLoader to manage prompts and other configurations
      - Maintain strict logging of messages (conversation history) for replay / logging
    """

    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        self.prompts = self.config_loader.prompts

        # Data toolkit: encapsulates query capabilities with experimental databases/files
        self.toolkit = DataToolkit()

        # Open LLM client and model configuration (returns client, model_name, temperature)
        self.client, self.model, self.temperature = get_llm_client()
        
        # Declare the tool schemas available to LLM
        self.tools_schema = [
             {
                "type": "function",
                "function": {
                    "name": "analyze_material_data",
                    "description": "Retrieve experimental data, which is essential to understanding the relationships between precursors, processing, and performance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_type": {"type": "string", "enum": ["emission", "lifetime"]},
                            "min_value": {"type": "number"},
                            "max_value": {"type": "number"}
                        },
                        "required": ["target_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_feature_importance",
                    "description": "Retrieve feature importance and SHAP analysis plots to determine critical chemical groups (Bits) and process conditions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "enum": ["lifetime", "emission"]}
                        },
                        "required": ["target"]
                    }
                }
            }
        ]

    # --- Helper Functions ---
    def _encode_image(self, image_path):
        """
        Return base64 encoded string of the image at the specified path.
        Returns None if image does not exist or fails to read.

        Used when embedding local images in messages (e.g., sending SHAP plot to LLM).
        """
        if not os.path.exists(image_path): return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except: return None

    def _serialize_message(self, msg):
        """
        Convert dialogue message to a Python primitive type suitable for logging.
        """
        if isinstance(msg, dict): return msg
        if hasattr(msg, 'to_dict'): return msg.to_dict()
        if hasattr(msg, 'model_dump'): return msg.model_dump()
        return str(msg)

    def _save_log(self, input_messages, raw_response, step):
        """
        Save an interaction with LLM (input messages and raw response) to run_dir/logs.
        Includes:
        1. jsonl format (machine readable)
        2. txt format (human readable)
        """
        try:
            log_dir = os.path.join(get_run_dir(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # --- 1. JSONL Logging ---
            path = os.path.join(log_dir, "planner_interaction_log.jsonl")
            s_inputs = []
            for m in input_messages:
                m_d = self._serialize_message(m)
                if isinstance(m_d.get('content'), list):
                    clean = []
                    for it in m_d['content']:
                        if it.get('type') == 'image_url': clean.append({"type":"image_url", "url":"[Base64 Truncated]"})
                        else: clean.append(it)
                    m_d = m_d.copy()
                    m_d['content'] = clean
                s_inputs.append(m_d)
            resp_d = raw_response.model_dump() if hasattr(raw_response, 'model_dump') else str(raw_response)
            entry = {"dt":str(datetime.now()), "step":step, "in":s_inputs, "out":resp_d}
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

            # --- 2. Readable Global Logging ---
            input_str = ""
            for m in input_messages:
                 if isinstance(m, dict):
                     role = m.get('role', 'unknown')
                     content = m.get('content', '')
                 else:
                     role = getattr(m, 'role', 'unknown')
                     content = getattr(m, 'content', '')
                 
                 if isinstance(content, list):
                     text_parts = [str(p.get('text', '')) for p in content if isinstance(p, dict) and p.get('type')=='text']
                     content = " ".join(text_parts) + " [Image content hidden]"
                 input_str += f"[{str(role).upper()}]:\n{str(content)}\n\n"
            
            resp_content = raw_response.choices[0].message.content if hasattr(raw_response, 'choices') else str(raw_response)
            
            log_to_global_file("PlannerAgent", input_str, resp_content, f"Step {step}")

        except Exception as e:
            print(f"⚠️ Log save failed: {e}")

    def _parse_tool_calls(self, text):
        """
        Parse tool call snippets output by LLM as free text.
        """
        found = []
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            try:
                if line.endswith(','): line = line[:-1]
                data = json.loads(line)
                if not isinstance(data, dict): continue
                if "name" in data:
                    name = data["name"].replace("functions.", "")
                    args = json.dumps(data.get("arguments")) if isinstance(data.get("arguments"), dict) else data.get("arguments")
                    found.append({"id":f"call_{len(found)}", "function":{"name":name, "arguments":args}})
                elif "target_type" in data:
                    found.append({"id":f"call_{len(found)}", "function":{"name":"analyze_material_data", "arguments":line}})
                elif "target" in data:
                    found.append({"id":f"call_{len(found)}", "function":{"name":"query_feature_importance", "arguments":line}})
            except: continue
        return found

    def _clean_json(self, content):
        """
        Clean up code block marks or redundant text in LLM-generated text to a clean JSON string.
        """
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        try:
            json.loads(content)
            return content
        except:
            m = re.search(r"(\{.*\})", content, re.DOTALL)
            return m.group(1) if m else content

    def _print_tool_execution_details(self, func_name, args, result, image_path=None):
        """
        Print input, output, and statistics of tool execution in a readable format to the console for manual review.
        """
        print("\n" + "="*80)
        print(f"🕵️  [Transparency Report] Agent is querying data")
        print(f"🛠️  Tool Name: {func_name}")
        print(f"📥  Input Parameters: {json.dumps(args, ensure_ascii=False)}")
        print("-" * 80)
        
        if func_name == "analyze_material_data":
            count = result.get("count", 0)
            data_preview = result.get("data", [])
            print(f"📊  [Experimental Data] Retrieved a total of {count} records.")
            print(f"    (Data Preview - Top 6):")
            print(json.dumps(data_preview[:6], ensure_ascii=False, indent=2))
            
        elif func_name == "query_feature_importance":
            bits = (result.get("feature_importance_list_BITS_TOP_20_CSV") or 
                   result.get("candidate_bits_from_csv") or 
                   result.get("top_bits_from_csv", []))
            conds = (result.get("feature_importance_list_CONDITIONS_TOP_20_CSV") or 
                    result.get("candidate_conditions_from_csv") or 
                    result.get("top_conditions_from_csv", []))
            
            print(f"📋  [Feature List] Candidate Bits ({len(bits)}):")
            print(json.dumps(bits, ensure_ascii=False, indent=2))
            print(f"📋  [Critical Conditions] Candidate Conditions ({len(conds)}):")
            print(json.dumps(conds, ensure_ascii=False, indent=2))
            
            if image_path:
                print(f"🖼️  [Visual Input] Showing SHAP image: {image_path}")
            else:
                print("    (No corresponding SHAP image found)")
                
        print("="*80 + "\n")

    def run(self, user_query):
        """
        Main run entrance:
          1. Construct system + user messages
          2. Force calling two core tools (analyze_material_data + query_feature_importance)
          3. Append tool results to messages
          4. Trigger LLM to generate final JSON report
        """
        system_prompt = get_prompt('planner_agent_system', '')
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        print(f"🤖 [Planner] Received instruction: {user_query}")

        execution_context = {
            "retrieved_experiments": [],    # Store analyze_material_data results
            "feature_importance_top5": {},  # Store query_feature_importance results
            "user_query": user_query
        }

        # ====================================================================
        # Force calling core tools
        # ====================================================================
        
        # 1. Determine target type (emission or lifetime)
        target_type = "emission"  # Default
        query_lower = user_query.lower()
        if "lifetime" in query_lower or "lifetime" in query_lower or "afterglow" in query_lower:
            target_type = "lifetime"
        if "emission" in query_lower or "wavelength" in query_lower or "emission" in query_lower:
            target_type = "emission"
        
        print(f"🎯 [Planner] Detected target type: {target_type}")
        
        # 2. Force calling analyze_material_data
        print("\n" + "="*80)
        print("🔧 [Planner] Forcing tool call: analyze_material_data")
        material_result = self.toolkit.get_experiment_data_with_sampling(
            target_type=target_type,
            min_val=None,
            max_val=None
        )
        self._print_tool_execution_details("analyze_material_data", {"target_type": target_type}, material_result)
        
        if material_result.get("data"):
            execution_context["retrieved_experiments"].extend(material_result["data"])
        
        messages.append({
            "role": "assistant",
            "content": f"I have called the analyze_material_data tool to retrieve experimental data."
        })
        messages.append({
            "role": "user",
            "content": f"Tool returned results (analyze_material_data):\n{json.dumps(material_result, ensure_ascii=False)}"
        })
        
        # 3. Force calling query_feature_importance
        print("🔧 [Planner] Forcing tool call: query_feature_importance")
        feature_result = self.toolkit.query_feature_importance(target_type, top_n=20)
        image_to_inject = feature_result.get('shap_image_path') if feature_result.get('shap_image_available') else None
        
        feature_summary = {
            "target": feature_result.get("target"),
            "feature_importance_list_BITS_TOP_20_CSV": feature_result.get("top_bits_from_csv"),
            "feature_importance_list_CONDITIONS_TOP_20_CSV": feature_result.get("top_conditions_from_csv"),
            "note": "SHAP image attached. Please use it to refine the selection." if image_to_inject else "No SHAP image."
        }
        self._print_tool_execution_details("query_feature_importance", {"target": target_type}, feature_summary, image_to_inject)
        
        execution_context["feature_importance_top5"] = {
            "top_bits": feature_result.get("top_bits_from_csv"),
            "top_conditions": feature_result.get("top_conditions_from_csv")
        }
        
        messages.append({
            "role": "user",
            "content": f"Tool returned results (query_feature_importance) - Candidates:\n{json.dumps(feature_summary, ensure_ascii=False)}"
        })
        
        # 4. If SHAP image is available, inject into messages
        if image_to_inject:
            b64 = self._encode_image(image_to_inject)
            if b64:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"This is the {target_type}'s SHAP summary plot (SHAP Summary Plot). The plot shows the direction and magnitude of the impact of each feature on the target value."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                })
        
        # 5. Inform LLM data collection is complete and generate final report
        messages.append({
            "role": "user", 
            "content": """Data collection complete. 
I have provided candidate lists of important features (Bits and Conditions) from the global analysis CSV and the SHAP summary plot.

**CRITICAL TASK**:
1. Analyze the SHAP plot to identify which features have the most distinct and consistent impact.
2. Select the **Final 10 Most Critical Features** from the candidates.
   - **MANDATORY**: You MUST include the Top 3 Bits and Top 3 Conditions from the **"feature_importance_list_BITS_TOP_20_CSV"** and **"feature_importance_list_CONDITIONS_TOP_20_CSV"** provided above. Do not ignore them.
   - Combine these with insights from the SHAP plot.
   
3. **Generate the final JSON report** with the following keys (ensure to include the new 'raw_feature_importance_top_csv' key):
{
    "target_property": "...",
    "design_constraints": [...],
    "key_bits_to_decode": [ ... mixed selection ... ],
    "key_Importance": [ ... mixed selection ... ],
    "raw_feature_importance_top_csv": {
         "bits": [ ... copy top 5 from CSV list ... ],
         "conditions": [ ... copy top 5 from CSV list ... ]
    },
    "data_insights": "...",
    "task_type": "..."
}"""
        })

        # ====================================================================
        # Trigger LLM to generate final JSON report
        # ====================================================================
        print("🤖 [Planner] Generating JSON...")
        try:
            from src.llm_client import retry_with_backoff
            
            final = retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model, 
                    messages=messages, 
                    response_format={"type": "json_object"},
                    temperature=self.temperature
                ),
                max_retries=5,
                initial_wait=30,
                description=f"PlannerAgent:{self.model}",
                input_messages=messages
            )
            self._save_log(messages, final, 1)
            
            cleaned_json_str = self._clean_json(final.choices[0].message.content)
            
            return cleaned_json_str, execution_context
            
        except Exception as e:
            print(f"❌ [Planner] LLM failed to generate report: {e}")
            # Fallback JSON
            fallback_json = json.dumps({
                "target_property": target_type,
                "design_constraints": [],
                "key_bits_to_decode": execution_context["feature_importance_top5"].get("top_bits", [])[:8],
                "key_Importance": execution_context["feature_importance_top5"].get("top_conditions", [])[:5],
                "data_insights": f"Based on {len(execution_context['retrieved_experiments'])} records of experimental data analysis",
                "error": str(e)
            }, ensure_ascii=False)
            return fallback_json, execution_context
