import os
import json
import base64
import re
import time
import threading
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)


# ==============================================================================
# --- Global Token Usage Tracker ---
# ==============================================================================
class TokenTracker:
    """Global singleton Token usage tracker, automatically recording Token consumption for every LLM API call."""
    _instance = None
    _cls_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset tracker (for a new run)"""
        with cls._cls_lock:
            cls._instance = cls()
    
    def __init__(self):
        self.records = []
        self.total_input = 0
        self.total_output = 0
        self.total_tokens = 0
        self._lock = threading.Lock()
    
    def record(self, agent, model, input_tokens, output_tokens, total_tokens=None):
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        with self._lock:
            self.records.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "agent": agent,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            })
            self.total_input += input_tokens
            self.total_output += output_tokens
            self.total_tokens += total_tokens
            
            # Immediately print consumption after each model call
            print(f"\n[{time.strftime('%H:%M:%S')}] \U0001f4b0 Token Consumption (Agent: {agent} | Model: {model}) -> This: {total_tokens:,} | Total: {self.total_tokens:,}")

    def record_raw(self, record_dict):
        """Directly record a raw dictionary (containing timestamp, agent, model, input_tokens, output_tokens, total_tokens)"""
        with self._lock:
            # Compatibility check and backfill
            r = {
                "timestamp": record_dict.get("timestamp", record_dict.get("time", time.strftime("%H:%M:%S"))),
                "agent": record_dict.get("agent", "Unknown"),
                "model": record_dict.get("model", "Unknown"),
                "input_tokens": int(record_dict.get("input_tokens", 0)),
                "output_tokens": int(record_dict.get("output_tokens", 0)),
                "total_tokens": int(record_dict.get("total_tokens", record_dict.get("total", 0))),
            }
            if r["total_tokens"] == 0:
                r["total_tokens"] = r["input_tokens"] + r["output_tokens"]
            
            self.records.append(r)
            self.total_input += r["input_tokens"]
            self.total_output += r["output_tokens"]
            self.total_tokens += r["total_tokens"]

    def merge_from_csv(self, csv_path):
        """Merge records from existing Token_and_Cost_Summary.csv"""
        import csv
        if not os.path.exists(csv_path):
            return False
        
        try:
            new_records = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip summary row
                    if row.get("time") == "TOTAL":
                        continue
                    new_records.append(row)
            
            # Since there might be multiple resumes, we merge sequentially
            for row in new_records:
                self.record_raw(row)
            return True
        except Exception as e:
            print(f"⚠️ Failed to merge tokens from {csv_path}: {e}")
            return False
    
    def print_summary(self):
        print(f"\n{'='*65}")
        print(f"  \U0001f4b0 Token Usage Summary Report")
        print(f"{'='*65}")
        agent_stats = {}
        for r in self.records:
            a = r["agent"]
            if a not in agent_stats:
                agent_stats[a] = {"calls": 0, "input": 0, "output": 0, "total": 0}
            agent_stats[a]["calls"] += 1
            agent_stats[a]["input"] += r["input_tokens"]
            agent_stats[a]["output"] += r["output_tokens"]
            agent_stats[a]["total"] += r["total_tokens"]
        
        print(f"  {'Agent':<25} {'Calls':>6} {'Input':>10} {'Output':>10} {'Total':>10}")
        print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
        for agent, s in sorted(agent_stats.items()):
            print(f"  {agent:<25} {s['calls']:>6} {s['input']:>10,} {s['output']:>10,} {s['total']:>10,}")
        print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'TOTAL':<25} {len(self.records):>6} {self.total_input:>10,} {self.total_output:>10,} {self.total_tokens:>10,}")
        print(f"{'='*65}\n")
    
    def save_to_csv(self, save_dir):
        """Save detailed records as an independent Token usage summary CSV"""
        import csv
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "Token_and_Cost_Summary.csv")
        
        # Rate dictionary (estimated price per 1M Tokens / USD)
        # Using rough average price, variations exist across different models
        PRICES = {
            "flash": {"in": 0.1, "out": 0.3},    # Gemini Flash series, etc.
            "pro":   {"in": 1.25, "out": 3.75}, # Gemini Pro / GPT-4 series
            "default": {"in": 0.5, "out": 1.5}   # Default intermediate price
        }

        def get_rate(model_name):
            m = str(model_name).lower()
            if "lite" in m or "flash" in m or "deepseek" in m: return PRICES["flash"]
            if "pro" in m or "ultra" in m or "gpt-4" in m or "gpt-5" in m: return PRICES["pro"]
            return PRICES["default"]

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["time", "agent", "model", "input_tokens", "output_tokens", "total", "est_cost_usd"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                total_cost = 0
                for r in self.records:
                    rate = get_rate(r["model"])
                    cost = (r["input_tokens"] / 1_000_000 * rate["in"]) + (r["output_tokens"] / 1_000_000 * rate["out"])
                    total_cost += cost
                    
                    writer.writerow({
                        "time": r["timestamp"],
                        "agent": r["agent"],
                        "model": r["model"],
                        "input_tokens": r["input_tokens"],
                        "output_tokens": r["output_tokens"],
                        "total": r["total_tokens"],
                        "est_cost_usd": f"{cost:.6f}"
                    })
                
                # Write summary row
                writer.writerow({
                    "time": "TOTAL",
                    "agent": f"{len(self.records)} calls",
                    "model": "-",
                    "input_tokens": self.total_input,
                    "output_tokens": self.total_output,
                    "total": self.total_tokens,
                    "est_cost_usd": f"{total_cost:.4f}"
                })
            print(f"💰 Token cost report successfully exported separately: {path}")
        except Exception as e:
            print(f"⚠️ Failed to save Token cost report: {e}")

def _estimate_tokens(text):
    """Roughly estimate text Token count (approx. 1.5 characters/token for mixed text)"""
    if not text: return 0
    return max(1, int(len(text) / 1.5))


def _extract_token_usage(response, description="", input_messages=None):
    """Extract Token usage from API response and record to global tracker.
    
    Prioritize using API returned usage field; if not returned (e.g., Poe proxy),
    make a rough estimation based on input messages and output text (marked as estimated).
    """
    try:
        model_name = getattr(response, 'model', '') or ''
        agent_name = description.split(":")[0] if ":" in description else description
        input_t = 0
        output_t = 0
        total_t = 0
        is_estimated = False
        
        # 1. Try to extract from API native usage
        if hasattr(response, 'usage') and response.usage is not None:
            usage = response.usage
            input_t = getattr(usage, 'prompt_tokens', 0) or 0
            output_t = getattr(usage, 'completion_tokens', 0) or 0
            total_t = getattr(usage, 'total_tokens', 0) or (input_t + output_t)
        
        # 2. Fallback: estimate based on text length
        if input_t == 0 and output_t == 0:
            is_estimated = True
            try:
                content = response.choices[0].message.content or ""
                output_t = _estimate_tokens(content)
            except Exception:
                output_t = 0
            if input_messages:
                total_input_text = ""
                for msg in input_messages:
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        total_input_text += c
                    elif isinstance(c, list):
                        for item in c:
                            total_input_text += item.get("text", "")
                input_t = _estimate_tokens(total_input_text)
            total_t = input_t + output_t
            if ":" in description:
                model_name = description.split(":")[1] + " (est.)"
            else:
                model_name = "(est.)"
        
        if input_t > 0 or output_t > 0:
            TokenTracker.get_instance().record(agent_name, model_name, input_t, output_t, total_t)
    except Exception:
        pass


def retry_with_backoff(func, max_retries=5, initial_wait=30, backoff_factor=2, description="LLM API", input_messages=None):
    """
    Centralized retry wrapper with exponential backoff for rate-limited API calls.
    Automatically tracks Token usage of successful calls.
    
    Args:
        input_messages: Optional, list of input messages to estimate input token count when API does not return usage.
    """
    last_exception = None
    wait_time = initial_wait
    
    for attempt in range(1, max_retries + 1):
        try:
            result = func()
            _extract_token_usage(result, description, input_messages)
            return result
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            is_rate_limit = (
                '429' in error_str or
                'rate_limit' in error_str or
                'resource_exhausted' in error_str or
                'too many requests' in error_str or
                'rate limit' in error_str
            )
            
            if is_rate_limit and attempt < max_retries:
                print(f"\n\u23f3 [{description}] Rate limit triggered (429), retry {attempt}/{max_retries}...")
                print(f"   Waiting {wait_time} seconds before retrying (exponential backoff: {initial_wait}s \u00d7 {backoff_factor}^{attempt-1})...")
                time.sleep(wait_time)
                wait_time = min(wait_time * backoff_factor, 600)
            elif not is_rate_limit:
                raise
            else:
                print(f"\n\u274c [{description}] All {max_retries} retries exhausted.")
                raise
    
    raise last_exception

class OpenAIResponseAdapter:
    """Adapts Gemini response to OpenAI-like response object"""
    def __init__(self, gemini_response):
        self.gemini_response = gemini_response
        
        # Prepare content
        self.content = gemini_response.text if hasattr(gemini_response, 'text') else ""
        
        # Prepare tool calls
        self.tool_calls = []
        if hasattr(gemini_response, 'function_calls'):
            for fc in gemini_response.function_calls:
                # Gemini FC has 'name' and 'args' (dict)
                # OpenAI expects 'function' object with 'name' and 'arguments' (string)
                self.tool_calls.append({
                    "id": "gemini_tool_call", # dummy id
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args)
                    },
                    "type": "function"
                })
        
        # Structure to mimic resp.choices[0].message
        class Message:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls
                
            def to_dict(self):
                return {"content": self.content, "tool_calls": self.tool_calls}

        class Choice:
            def __init__(self, message):
                self.message = message

        self.choices = [Choice(Message(self.content, self.tool_calls if self.tool_calls else None))]
        
    def model_dump(self):
        # Rough emulation for logging
        return {"content": self.content, "tool_calls": self.tool_calls}


class GeminiClientWrapper:
    def __init__(self, api_key, model, temperature=0.1, thinking_level=None, media_resolution=None):
        # Use v1alpha for media_resolution support
        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.model = model
        self.temperature = temperature
        self.thinking_level = thinking_level
        self.media_resolution = media_resolution
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, model, messages, tools=None, tool_choice=None, response_format=None, temperature=None):
        """
        Mimics openai.chat.completions.create
        """
        
        # 1. Convert Messages
        system_instruction = None
        gemini_contents = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            parts = []
            if content:
                if isinstance(content, str):
                    parts.append(types.Part.from_text(text=content))
                elif isinstance(content, list):
                    # Handle multimodal content (text + image)
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item.get("text")))
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            # Parse data:image/png;base64,...
                            match = re.match(r"data:(.*?);base64,(.*)", url)
                            if match:
                                mime_type = match.group(1)
                                b64_data = match.group(2)
                                try:
                                    image_bytes = base64.b64decode(b64_data)
                                    blob = types.Blob(mime_type=mime_type, data=image_bytes)
                                    part_args = {"inline_data": blob}
                                    if self.media_resolution:
                                        part_args["media_resolution"] = {"level": self.media_resolution}
                                    parts.append(types.Part(**part_args))
                                except Exception as e:
                                    print(f"Error decoding image: {e}")
            
            if role == "system":
                system_instruction = content
                
            elif role == "user":
                gemini_contents.append(types.Content(role="user", parts=parts))
                
            elif role == "assistant":
                # Handle tool calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        fname = func.get("name")
                        fargs = func.get("arguments")
                        if fname:
                            try:
                                args_dict = json.loads(fargs) if isinstance(fargs, str) else fargs
                            except:
                                args_dict = {}
                            parts.append(types.Part.from_function_call(name=fname, args=args_dict))
                
                if not parts:
                    # Provide empty text if nothing (Gemini might error on empty content)
                    parts.append(types.Part.from_text(text=" "))
                    
                gemini_contents.append(types.Content(role="model", parts=parts))
                
            elif role == "tool":
                # Fallback: Represent tool output as user text to ensure model sees it
                text_content = f"Tool Execution Output (ID {msg.get('tool_call_id')}): {content}"
                gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text_content)])) 

        # 2. Tool Configuration
        gemini_tools = None
        if tools:
            funcs = []
            for t in tools:
                if t.get("type") == "function":
                    f_def = t.get("function")
                    funcs.append(f_def)
            
            if funcs:
                gemini_tools = [types.Tool(function_declarations=funcs)]

        # 3. Response Format (JSON mode)
        mime_type = "text/plain"
        if response_format and response_format.get("type") == "json_object":
            mime_type = "application/json"

        # 4. Generate
        gen_config_args = {
            "temperature": temperature if temperature is not None else self.temperature,
            "system_instruction": system_instruction,
            "tools": gemini_tools,
            "response_mime_type": mime_type
        }
        
        if self.thinking_level:
             gen_config_args["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)

        config = types.GenerateContentConfig(**gen_config_args)
        
        response = retry_with_backoff(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=config
            ),
            max_retries=5,
            initial_wait=30,
            description=f"Gemini:{self.model}"
        )
        
        return OpenAIResponseAdapter(response)


class LLMClientFactory:
    _instance = None
    
    @staticmethod
    def get_client(config_loader):
        full_conf = config_loader.config
        llm_conf = full_conf.get('llm', {})
        provider = llm_conf.get('provider', 'openai')
        
        if provider == 'openai':
            conf = llm_conf.get('openai', {})
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=conf.get('base_url')
            ), conf.get('model_name'), conf.get('temperature', 0.1)
            
        elif provider == 'gemini':
            conf = llm_conf.get('gemini', {})
            api_key = os.getenv("GEMINI_API_KEY")
            model_name = conf.get('model_name', 'gemini-2.0-flash-exp')
            temp = conf.get('temperature', 0.1)
            thinking_level = conf.get('thinking_level')
            media_resolution = conf.get('media_resolution')
            
            return GeminiClientWrapper(api_key, model_name, temp, thinking_level, media_resolution), model_name, temp
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
