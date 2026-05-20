import os
import json
import time
import re
import concurrent.futures
from openai import OpenAI
from src.utils import ConfigLoader, log_to_global_file
from src.llm_client import GeminiClientWrapper

class CriticAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        
        # Load judges from config.yaml
        critic_config = self.config_loader.config.get('critic', {})
        self.judges = critic_config.get('judges', [])
        
        if not self.judges:
            print("⚠️ [CriticAgent] No judges configured in config.yaml, using defaults.")
            self.judges = [
                {"name": "Default", "provider": "openai", "model": "gpt-4o", 
                 "api_key_env": "OPENAI_API_KEY", "base_url": "", "temperature": 0.1}
            ]
        
        print(f"📋 [CriticAgent] Loaded {len(self.judges)} judge model configurations")
        for j in self.judges:
            print(f"   - {j.get('name')}: {j.get('model')} @ {j.get('base_url', 'default')}")
            
        self.shap_knowledge = self.config_loader.prompts.get('shap_knowledge', '')

    def evaluate(self, architect_recipe, summary_report, iteration=1, log_dir="experiments", initial_query=None):
        # Create output directory for this iteration
        review_dir = os.path.join(log_dir, "Critic_Reviews", f"Round_{iteration}")
        os.makedirs(review_dir, exist_ok=True)
        
        print(f"\n⚖️ [CriticAgent] Convening {len(self.judges)} LLM judges to review the recipe (Round {iteration})...")
        if initial_query:
            print(f"🎯 User original requirement: {initial_query}")
        print(f"📂 Review details will be saved in: {review_dir}")
        
        results = []
        
        # 🚨 Key: Construct user requirements reminder
        user_goal_reminder = ""
        if initial_query:
            user_goal_reminder = f"""
### 🎯🚨 Original Design Requirements (CRITICAL - MUST VERIFY)
**Below are the user's core design requirements. When determining whether a recipe is reasonable, you must first verify whether it serves this goal:**

{initial_query}

⚠️ If the experimental recipe deviates from the user's original requirements (e.g., the user requests "designing a long-lifetime material" but the recipe optimizes "emission wavelength"), you must strictly deduct points and point out the issue!

---
"""
        
        def call_judge(judge_config):
            # Read individual judge configuration
            judge_name = judge_config.get("name", "Unknown")
            model_name = judge_config.get("model", "Gemini-3-Pro")
            provider = judge_config.get("provider", "openai")
            api_key_env = judge_config.get("api_key_env", "OPENAI_API_KEY")
            base_url = judge_config.get("base_url", "")
            temperature = judge_config.get("temperature", 0.1)
            
            # Get API key from environment
            api_key = os.getenv(api_key_env)
            if not api_key:
                print(f"⚠️ [{judge_name}] API key not found in env: {api_key_env}")
                return {
                    "judge_model": judge_name,
                    "is_reasonable": False,
                    "score": 0,
                    "reason": f"API key not configured: {api_key_env}"
                }
            
            prompt = f"""
You are a rigorous materials science review expert. Your task is to first reason about the corresponding knowledge of carbon dot-based long afterglow materials based on the user's task requirements, and then combine this knowledge with the following information to judge whether the [Experimental Recipe] is reasonable and provide a score (0-10 points).

{user_goal_reminder}
### SHAP Rules
{self.shap_knowledge}

### Task Summary
{json.dumps(summary_report, ensure_ascii=False)}

### Recipe under Review
{architect_recipe}

---
### Evaluation Requirements
1. 🚨 **MOST IMPORTANT**: First, verify whether the experimental recipe directly serves the user's original design requirements. If the recipe target does not match the user requirements (e.g., the user requests a long lifetime, but the recipe optimizes the emission wavelength), directly judge it as unreasonable (0-3 points).
2. Strictly compare whether the [precursor selection], [temperature], and [solvent] in the experimental recipe conform to the SHAP rules and the suggestions in the data summary.
3. If the recipe selects "negatively correlated" parameters to "increase" the target value, it must be judged as unreasonable (0-5 points).
4. If the recipe ignores key Bit features or gating rules (such as MW limits), it must be judged as unreasonable (0-5 points).
5. If the recipe is logically rigorous and conforms to all rules, give a high score (8-10 points).
6. If there are issues, you must provide reasonable suggestions for correction.
7. You must strictly judge the rationality of the precursors and experimental conditions, and your scoring must be reasonable!
8. 🚨 **FATAL GUARDRAIL REQUIREMENT**: Your output MUST be a valid pure JSON! In the text content of "reason" and "suggest", **it is strictly forbidden to use any unescaped double quotes**! If you need to quote text, please be sure to use [single quotes] or full-width double quotes (“”), otherwise it will cause JSON parsing failure and program crash.

### Return output strictly in JSON format (ensure no unescaped double quotes):
{{
    "judge_model": "{judge_name}",
    "is_reasonable": true/false,
    "score": 5.5, // 0-10 score, supports decimals
    "reason": "Detailed reason, pointing out exactly where it conforms or violates rules",
    "suggest": "Detailed suggestions based on user requirements"
}}
```"""
            
            content = None
            
            from src.llm_client import retry_with_backoff
            
            try:
                if provider == 'openai':
                    # Use individual base_url for this judge
                    client = OpenAI(
                        api_key=api_key, 
                        base_url=base_url if base_url else None
                    )
                    critic_msgs = [{"role": "user", "content": prompt}]
                    response = retry_with_backoff(
                        lambda: client.chat.completions.create(
                            model=model_name,
                            messages=critic_msgs,
                            temperature=temperature,
                            max_tokens=4096
                        ),
                        max_retries=5,
                        initial_wait=30,
                        description=f"Critic:{judge_name}",
                        input_messages=critic_msgs
                    )
                    content = response.choices[0].message.content
                    
                elif provider == 'gemini':
                    gemini_conf = self.config_loader.config.get('llm', {}).get('gemini', {})
                    client = GeminiClientWrapper(
                        api_key=api_key, 
                        model=model_name, 
                        temperature=temperature,
                        thinking_level=gemini_conf.get('thinking_level'),
                        media_resolution=gemini_conf.get('media_resolution')
                    )
                    critic_msgs_g = [{"role": "user", "content": prompt}]
                    response = retry_with_backoff(
                        lambda: client.create(
                             model=model_name,
                             messages=critic_msgs_g
                        ),
                        max_retries=5,
                        initial_wait=30,
                        description=f"Critic:{judge_name}",
                        input_messages=critic_msgs_g
                    )
                    content = response.choices[0].message.content
                    
            except Exception as e:
                print(f"⚠️ Judge {judge_name} ultimately failed: {e}")
                return {
                    "judge_model": judge_name,
                    "is_reasonable": False,
                    "score": 0,
                    "reason": f"API Error after retries: {str(e)}"
                }

            # --- Save Readable Interaction Log for verify (Consolidated) ---
            if content:
                log_to_global_file(
                      f"CriticAgent::{judge_name}", 
                      prompt, 
                      content, 
                      f"Critic Review Iteration {iteration}"
                )
            
            # ====================================================================================
            #                          ROBUST JSON PARSING (FIXED v2)
            # ====================================================================================
            if not content:
                return {
                    "judge_model": judge_name,
                    "is_reasonable": False,
                    "score": 0,
                    "reason": "Empty response from LLM"
                }

            def _sanitize_json_string(raw_json):
                """
                Multi-layered cleanup of the JSON string returned by LLM, core fixes:
                  - Remove // comments inside JSON
                  - Remove trailing commas (trailing comma before } or ])
                  - Fix unescaped double quotes and illegal literal newlines inside string values
                  - Ultra-robust extraction fallback strategy to guarantee reason and suggest are not lost
                """
                s = raw_json.strip()
                
                # Step 0: Remove invisible control characters (keep \t \n \r)
                s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
                
                # Step 1: Remove single-line comments in JSON (// ...)
                s = re.sub(r'(?<!["\\\w])//.*?$', '', s, flags=re.MULTILINE)
                
                # Step 2: Remove trailing comma
                s = re.sub(r',\s*([}\]])', r'\1', s)
                
                # Step 3: Try parsing directly
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
                
                # Step 4: Try replacing raw newlines in string literals
                s_no_newline = s.replace('\n', '\\n').replace('\r', '\\r')
                try:
                    return json.loads(s_no_newline)
                except json.JSONDecodeError:
                    pass
                
                # Step 5: State-machine fix for unescaped double quotes and string literal newlines
                try:
                    result = []
                    i = 0
                    in_string = False
                    
                    while i < len(s):
                        ch = s[i]
                        
                        if ch == '\\' and in_string:
                            result.append(ch)
                            if i + 1 < len(s):
                                result.append(s[i + 1])
                                i += 2
                            else:
                                i += 1
                            continue
                        
                        if in_string:
                            if ch == '"':
                                rest = s[i + 1:].lstrip()
                                looks_like_close = False
                                if not rest:
                                    looks_like_close = True
                                elif rest[0] in (':', '}', ']'):
                                    looks_like_close = True
                                elif rest[0] == ',':
                                    if re.match(r',\s*(?:"|\{|\[|true|false|null|-?\d)', rest):
                                        looks_like_close = True
                                
                                if looks_like_close:
                                    in_string = False
                                    result.append(ch)
                                else:
                                    result.append('\u201c')
                            elif ch == '\n':
                                result.append('\\n')
                            elif ch == '\r':
                                result.append('\\r')
                            elif ch == '\t':
                                result.append('\\t')
                            else:
                                result.append(ch)
                        else:
                            if ch == '"':
                                in_string = True
                            result.append(ch)
                        
                        i += 1
                    
                    sanitized = ''.join(result)
                    parsed = json.loads(sanitized)
                    return parsed
                except (json.JSONDecodeError, Exception):
                    pass
                
                # Step 6: Regex extraction fallback
                try:
                    reconstructed = {}
                    
                    jm = re.search(r'"judge_model"\s*:\s*"(.*?)"', s, re.IGNORECASE)
                    reconstructed["judge_model"] = jm.group(1) if jm else judge_name
                    
                    is_r = re.search(r'"is_reasonable"\s*:\s*(true|false)', s, re.IGNORECASE)
                    reconstructed["is_reasonable"] = (is_r.group(1).lower() == 'true') if is_r else False
                    
                    score_m = re.search(r'"score"\s*:\s*([0-9.]+)', s, re.IGNORECASE)
                    reconstructed["score"] = float(score_m.group(1)) if score_m else 0.0
                    
                    keys_lookahead = r'(?=\s*,\s*"(?:judge_model|is_reasonable|score|reason|suggest|suggestions|suggestion)"\s*:|\s*\}?\s*$)'
                    
                    reason_pattern = r'"reason"\s*:\s*(.*?)' + keys_lookahead
                    reason_m = re.search(reason_pattern, s, re.IGNORECASE | re.DOTALL)
                    if reason_m:
                        res_val = reason_m.group(1).strip()
                        res_val = re.sub(r'^\[?\s*"?|"?\s*\]?$', '', res_val)
                        reconstructed["reason"] = res_val.replace('"', "'").replace('\n', '\\n')
                    else:
                        reconstructed["reason"] = "Extracted via regex fallback: reason not found in text."
                        
                    suggest_pattern = r'"(?:suggest|suggestions|suggestion)"\s*:\s*(.*?)' + keys_lookahead
                    suggest_m = re.search(suggest_pattern, s, re.IGNORECASE | re.DOTALL)
                    if suggest_m:
                        s_val = suggest_m.group(1).strip()
                        s_val = re.sub(r'^\[?\s*"?|"?\s*\]?$', '', s_val)
                        reconstructed["suggest"] = s_val.replace('"', "'").replace('\n', '\\n')
                    else:
                        reconstructed["suggest"] = ""
                        
                    return reconstructed
                except Exception:
                    pass
                
                return None
            
            try:
                return json.loads(content)
            except:
                pass

            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(code_block_pattern, content)
            
            if matches:
                for match_content in reversed(matches):
                    try:
                        parsed = json.loads(match_content.strip())
                        if isinstance(parsed, dict) and ('score' in parsed or 'is_reasonable' in parsed):
                            return parsed
                    except:
                        pass
                    sanitized = _sanitize_json_string(match_content.strip())
                    if sanitized and isinstance(sanitized, dict) and ('score' in sanitized or 'is_reasonable' in sanitized):
                        print(f"   🔧 [{judge_name}] JSON parsed successfully after cleanup (code block)")
                        return sanitized

            try:
                open_braces = [m.start() for m in re.finditer(r'\{', content)]
                for start in reversed(open_braces):
                    depth = 0
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                candidate = content[start:i+1]
                                try:
                                    parsed = json.loads(candidate)
                                    if isinstance(parsed, dict) and ('score' in parsed or 'is_reasonable' in parsed):
                                        return parsed
                                except:
                                    pass
                                sanitized = _sanitize_json_string(candidate)
                                if sanitized and isinstance(sanitized, dict) and ('score' in sanitized or 'is_reasonable' in sanitized):
                                    print(f"   🔧 [{judge_name}] JSON parsed successfully after cleanup (balanced braces)")
                                    return sanitized
                                break
            except Exception as e:
                pass

            sanitized = _sanitize_json_string(content)
            if sanitized and isinstance(sanitized, dict) and ('score' in sanitized or 'is_reasonable' in sanitized):
                print(f"   🔧 [{judge_name}] JSON parsed successfully after cleanup (full content)")
                return sanitized

            tail_content = content[-200:].replace('\n', ' ')
            print(f"⚠️ Judge {judge_name} failed all JSON parsing methods. Tail: {tail_content}")
            
            return {
                "judge_model": judge_name,
                "is_reasonable": False,
                "score": 0,
                "reason": f"Could not parse JSON from response. Response tail: {tail_content}"
            }

        # [Performance Optimization] Parallel review using ThreadPoolExecutor, drastically reducing wait time
        print(f"\n   🚀 Launching {len(self.judges)} judge models in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(self.judges))) as executor:
            future_to_judge = {}
            for i, judge in enumerate(self.judges):
                future = executor.submit(call_judge, judge)
                future_to_judge[future] = (i, judge)
                time.sleep(1) # Micro-interval to avoid hitting the same API simultaneously
            
            for future in concurrent.futures.as_completed(future_to_judge):
                i, judge = future_to_judge[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {"judge_model": judge.get("name", "Unknown"), "is_reasonable": False, "score": 0, "reason": str(e)}
                results.append(res)
                judge_name = res.get('judge_model', 'unknown')
                print(f"   ✅ Review completed: {judge_name} | Score: {res.get('score', 'N/A')}")
                safe_name = judge_name.replace('/', '_').replace(':', '')
                with open(os.path.join(review_dir, f"Review_{safe_name}.json"), "w", encoding='utf-8') as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
        
        # Statistics
        pass_count = sum(1 for r in results if r.get('is_reasonable') == True)
        scores = [r.get('score', 0) for r in results if isinstance(r.get('score'), (int, float))]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"\n📊 Review Results: {pass_count}/{len(results)} Passed | Average Score: {avg_score:.1f}")
        
        for r in results:
            icon = "✅" if r.get('is_reasonable') else "❌"
            print(f"   {icon} [{r.get('judge_model')}]: {r.get('score')} - {r.get('reason')}")

        # Save Round Summary
        summary_stats = {
            "iteration": iteration,
            "pass_count": pass_count,
            "fail_count": len(results) - pass_count,
            "avg_score": avg_score,
            "details": results
        }
        with open(os.path.join(review_dir, "Round_Summary.json"), "w", encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)

        return {
            "pass": avg_score >= 8.5, 
            "passed": avg_score >= 8.5, 
            "pass_count": pass_count,
            "avg_score": avg_score,
            "details": results
        }