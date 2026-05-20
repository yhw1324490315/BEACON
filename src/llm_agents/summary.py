import os
import json
import base64
import glob
import pandas as pd
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client, log_to_global_file

from PIL import Image
import io

# SVG to PNG conversion - try multiple backends
HAS_SVG_CONVERTER = False
SVG_BACKEND = None

# Try 1: svglib + reportlab (needs Cairo on some systems)
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    HAS_SVG_CONVERTER = True
    SVG_BACKEND = "svglib"
except Exception as e:
    pass

# Try 2: wand (ImageMagick binding)
if not HAS_SVG_CONVERTER:
    try:
        from wand.image import Image as WandImage
        HAS_SVG_CONVERTER = True
        SVG_BACKEND = "wand"
    except Exception:
        pass

# Load environment variables from the project config directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(project_root, "config", "secrets.env")
load_dotenv(env_path)

class SummaryAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        
        # Initialize LLM Client
        self.client, self.model, self.temperature = get_llm_client()

        # Load System Prompt from yaml
        self.system_prompt = get_prompt('summary_agent_system', '')

    # ================= Helper Functions =================

    def _encode_image(self, image_path):
        """Convert local image to Base64"""
        if not os.path.exists(image_path): return None
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except: return None

    def _read_file_content(self, file_path):
        if not os.path.exists(file_path): return None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except: return None

    def _get_browser_executable(self):
        """Find Edge or Chrome browser path in system"""
        paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    def _convert_svg_using_browser(self, svg_path, png_path):
        """Use system browser screenshot to convert SVG to PNG (Ultimate Fallback)"""
        import subprocess
        import shutil
        import time
        
        browser_exe = self._get_browser_executable()
        if not browser_exe:
            print("⚠️ Edge or Chrome browser not found, unable to perform fallback conversion.")
            return False
            
        try:
            # Ensure absolute paths are used
            abs_svg_path = os.path.abspath(svg_path)
            abs_png_path = os.path.abspath(png_path)
            
            # Browser-generated default screenshot filename is usually "screenshot.png" in CWD
            # Alternatively, specify --screenshot=path (supported in newer versions)
            cmd = [
                browser_exe,
                "--headless",
                "--disable-gpu",
                "--window-size=500,500",
                "--hide-scrollbars",
                f"--screenshot={abs_png_path}",
                f"file:///{abs_svg_path.replace(os.sep, '/')}"
            ]
            
            # Run browser command
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            # Check if the file is generated
            if os.path.exists(abs_png_path) and os.path.getsize(abs_png_path) > 0:
                return True
                
            # Try searching for default screenshot.png (if --screenshot parameter doesn't support custom paths)
            cwd_screenshot = os.path.join(os.getcwd(), "screenshot.png")
            if os.path.exists(cwd_screenshot):
                shutil.move(cwd_screenshot, abs_png_path)
                return True
                
            print(f"⚠️ Browser screenshot failed, output file not found. Stderr: {result.stderr.decode('utf-8', errors='ignore')}")
            return False
            
        except Exception as e:
            print(f"⚠️ Browser conversion exception: {e}")
            return False

    def _svg_to_png(self, svg_path, png_path=None):
        """
        Convert SVG file to PNG and return Base64 encoding.
        Strategy Priority:
        1. svglib/wand (if libraries available)
        2. System browser screenshot (General Fallback)
        """
        if not os.path.exists(svg_path):
            print(f"⚠️ SVG file does not exist: {svg_path}")
            return None
        
        # If PNG path not specified, save to the same directory as the SVG
        if png_path is None:
            png_path = svg_path.replace('.svg', '.png')

        # Prioritize using existing PNG generated by RDKit
        if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            try:
                with open(png_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                print(f"⚠️ Failed to read existing PNG: {e}")

        conversion_success = False

        # Strategy 1: Python library
        if HAS_SVG_CONVERTER:
            try:
                if SVG_BACKEND == "svglib":
                    drawing = svg2rlg(svg_path)
                    if drawing:
                        renderPM.drawToFile(drawing, png_path, fmt="PNG")
                        conversion_success = True
                elif SVG_BACKEND == "wand":
                    with WandImage(filename=svg_path) as img:
                        img.format = 'png'
                        img.save(filename=png_path)
                        conversion_success = True
            except Exception as e:
                print(f"⚠️ Python library conversion failed ({SVG_BACKEND}): {e}")
        
        # Strategy 2: Browser screenshot (if Strategy 1 fails or is unavailable)
        if not conversion_success:
            print(f"🔄 Attempting system browser conversion: {os.path.basename(svg_path)}")
            conversion_success = self._convert_svg_using_browser(svg_path, png_path)
            
        if conversion_success and os.path.exists(png_path):
            try:
                # Read and encode to Base64
                with open(png_path, 'rb') as f:
                    png_data = f.read()
                base64_str = base64.b64encode(png_data).decode('utf-8')
                print(f"✅ SVG successfully converted to PNG: {png_path}")
                return base64_str
            except Exception as e:
                 print(f"⚠️ Failed to read PNG: {e}")
                 return None
        else:
            print(f"❌ Failed to convert SVG to PNG (due to missing environment dependencies)")
            return None


    def _save_readable_log(self, input_messages, raw_response, step_info="Summary"):
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
             
             log_to_global_file("SummaryAgent", input_str, resp_content, step_info)
        except Exception as e:
             print(f"⚠️ Log save failed: {e}")

    # ================= Core Analysis Module =================

    def summarize(self, planner_json, output_dirs, critic_feedback=None, initial_query=None):
        print("🤖 [SummaryAgent] Start summarizing analysis...")
        if initial_query:
            print(f"🎯 User original requirement: {initial_query}")
        
        evidence_text = ""
        evidence_count = 0
        evidence_list = []  # Store detailed information of each evidence
        svg_images = []  # Store SVG image info for LLM vision input

        # Collect Evidence from analysis directories (RECURSIVE SEARCH)
        for d in output_dirs:
            if not os.path.exists(d): continue
            
            # 1. First collect SVG image directory (molecular structure diagrams), read SMILES from source_info.txt to generate PNG
            svg_pattern = os.path.join(d, "**", "*.svg")
            svg_files = glob.glob(svg_pattern, recursive=True)
            for svg_file in svg_files:
                svg_dir = os.path.dirname(svg_file)
                svg_info = {
                    "file_path": svg_file,
                    "file_name": os.path.basename(svg_file),
                    "rel_path": os.path.relpath(svg_file, d),
                    "source_dir": os.path.basename(d)
                }
                
                # Try to read corresponding source_info.txt for SMILES info
                source_info_path = os.path.join(svg_dir, "source_info.txt")
                if os.path.exists(source_info_path):
                    try:
                        with open(source_info_path, 'r', encoding='utf-8') as f:
                            svg_info["source_info"] = f.read()
                    except Exception as e:
                        print(f"⚠️ Failed to read source_info: {e}")
                
                # Directly convert SVG to PNG (SVG already generated by deep_analysis_tool.py)
                png_base64 = self._svg_to_png(svg_file)
                if png_base64:
                    svg_info["png_base64"] = png_base64
                
                # Even if image conversion fails, append information (including SMILES text) for fallback handling
                svg_images.append(svg_info)
            
            # 2. Recursively collect all text/csv/md files
            for ext in ["*.txt", "*.csv", "*.md"]:
                pattern = os.path.join(d, "**", ext)
                files = glob.glob(pattern, recursive=True)
                for f in files:
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            content = file.read()
                            # Truncate overly long content to prevent context overflow
                            if len(content) > 5000: 
                                content = content[:5000] + "\n...(truncated)..."
                            
                            # Get relative path for better context
                            rel_path = os.path.relpath(f, d)
                            source_label = f"{os.path.basename(d)}/{rel_path}"
                            
                            evidence_text += f"\n\n### Evidence Source: {source_label}\n"
                            evidence_text += content
                            evidence_count += 1
                            
                            # Add to evidence list
                            evidence_list.append({
                                "index": evidence_count,
                                "source": source_label,
                                "file_path": f,
                                "content_length": len(content)
                            })
                    except Exception as e:
                        print(f"Error reading {f}: {e}")

        # ========== Print detailed evidence list ==========
        print("\n" + "="*70)
        print(f"📊 Summary Agent collected {evidence_count} evidence materials + {len(svg_images)} molecular structure diagrams")
        print("="*70)
        
        print("\n📄 [Text Evidence Details]:")
        print("-"*70)
        for ev in evidence_list:
            print(f"  [{ev['index']:02d}] {ev['source']}")
            print(f"       Path: {ev['file_path']}")
            print(f"       Content Length: {ev['content_length']} characters")
        print("-"*70)
        
        print(f"\n🖼️  [Molecular Structure Diagram (SVG) Details]:")
        print("-"*70)
        for i, img in enumerate(svg_images, 1):
            print(f"  [{i:02d}] {img['file_name']}")
            print(f"       Path: {img['file_path']}")
            if 'source_info' in img:
                # Extract SMILES information
                lines = img['source_info'].split('\n')
                for line in lines:
                    if 'SMILES' in line or 'Bit' in line:
                        print(f"       {line.strip()}")
            has_png = '✅ PNG Converted' if 'png_base64' in img else '❌ Conversion Failed'
            print(f"       Status: {has_png}")
        print("-"*70)
        print("="*70 + "\n")

        # Construct Prompt - First add user original requirement, ensuring no deviation from the target
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # CRITICAL: Emphasize the user's original design requirement at the very beginning
        if initial_query:
            messages.append({
                "role": "user",
                "content": f"[🎯🚨 User Original Design Requirement (CRITICAL - MUST NOT DEVIATE)]\nPlease always keep in mind and strictly focus all analyses and recommendations around the following user requirements:\n\n{initial_query}\n\n⚠️ WARNING: All your analyses and design recommendations must directly serve the user requirements above, do not deviate!"
            })
        
        messages.append({"role": "user", "content": f"This is the original planning task of Planner Agent:\n{json.dumps(planner_json, ensure_ascii=False)}"})

        # Inject Critic Feedback if available (Iterative Refinement)
        if critic_feedback:
             messages.append({
                "role": "user", 
                "content": f"[⚠️ Why the previous scheme failed the review (Critical Feedback)]\nPlease pay great attention to the following review feedback and correct it accordingly in this design:\n{critic_feedback}"
            })

        # Inject Global SHAP Knowledge
        shap_knowledge = self.config_loader.prompts.get('shap_knowledge')
        if shap_knowledge:
             messages.append({
                "role": "user", 
                "content": f"[Global Feature Importance Knowledge Base (Global SHAP Knowledge)]\nPlease refer to the following global rules, which are the highest priority rules obtained from global data mining:\n{shap_knowledge}"
            })
        
        # Inject Evidence (Text)
        if evidence_text:
            messages.append({
                "role": "user", 
                "content": f"The following is specific evidence mined by Deep Analysis:\n{evidence_text}"
            })
        else:
             messages.append({
                "role": "user", 
                "content": "No new quantitative analysis evidence was mined in this round. Please reason mainly based on the knowledge base and planning intent."
            })
        
        # Inject verified Bit chemical meaning knowledge base (from Scout's KNOWN_BIT_SMARTS)
        # These are human-verified correct chemical meanings of Morgan Fingerprint Bits,
        # which can compensate for the limitation of Deep Analysis being unable to generate structure diagrams due to Hashed Fingerprint conflicts.
        KNOWN_BIT_MEANINGS = {
            "Bit_456":  "Aromatic anhydride (Anhydride, C(=O)OC(=O)) and/or aromatic carboxylic acid (Carboxylic Acid, cC(=O)O). This Bit matches both anhydride and carboxylic acid structures.",
            "Bit_1925": "Diaryl ketone structures (Diaryl Ketone, c(c)C(=O)c(c)). That is, a carbonyl C=O bridging two aromatic rings.",
            "Bit_1984": "Fused aromatic hydrocarbon / aromatic carbon (Fused Aromatic / Aromatic Carbon, c1ccc2ccccc2c1 or cc(c)c). Represents the carbon skeleton in polycyclic aromatic hydrocarbons.",
            "Bit_1039": "Aromatic carbon backbone (Aromatic Ring System). Typically represents the conjugated carbon skeleton structure of multiple aromatic rings.",
            "Bit_352":  "Oxygen heterocycle or ether linkage (Oxygen Heterocycle / Ether, C-O-C). May represent a furan ring, pyran ring, or simple aromatic ether structure.",
            "Bit_831":  "Boron-containing aromatic structure (Aryl Boron, Bc(c)ccc). Represents the connection of boric acid or boronic ester with an aromatic ring.",
            "Bit_1917": "Carbonyl group (Carbonyl, C=O). Represents the most basic C=O double bond structure.",
        }
        
        # Build text injection
        known_bits_text = "[🔬 Pre-decoded Bit Knowledge]\n"
        known_bits_text += "The following are human-verified correct chemical meanings of Morgan Fingerprint Bits. In your analysis, please use these definitions directly instead of guessing:\n\n"
        for bit_name, meaning in KNOWN_BIT_MEANINGS.items():
            known_bits_text += f"- **{bit_name}**: {meaning}\n"
        known_bits_text += "\n⚠️ If the structure diagram provided by Deep Analysis conflicts with the knowledge base above, please refer to the knowledge base."
        
        messages.append({
            "role": "user",
            "content": known_bits_text
        })
        
        # Inject Structure Evidence (Images OR Text)
        if svg_images:
            print(f"\n🖼️ Sending {len(svg_images)} key structural evidence units to LLM...")
            
            for img_info in svg_images:
                img_desc = f"[Key Substructure Evidence]\nFile Name: {img_info['file_name']}"
                if 'source_info' in img_info:
                    img_desc += f"\n{img_info['source_info']}"
                
                # Option A: With PNG image, use Vision capability
                if 'png_base64' in img_info:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"{img_desc}\n\n"
                                    f"⚠️ **Important Notes**:\n"
                                    f"1. **'Substructure SMILES'** is the core substructure represented by this Bit fingerprint. Please refer to this.\n"
                                    f"2. **'Source Molecule SMILES'** is merely an example parent molecule containing this Bit, providing context.\n"
                                    f"3. The highlighted (red/colored) part in the image corresponds to the Substructure, while the gray background is the parent molecule.\n"
                                    f"4. Please focus on analyzing the chemical characteristics of the Substructure (highlighted part) and its contribution to performance, do not mistake the whole parent molecule as the definition of the feature.\n"
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_info['png_base64']}"
                                }
                            }
                        ]
                    })
                # Option B: Fallback - No image, send text only
                else:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"{img_desc}\n\n"
                            f"⚠️ (Note: Due to environmental constraints, preview images could not be generated)\n"
                            f"Please note the distinction:\n"
                            f"- **Substructure SMILES**: The core structure of this Bit (analysis target)\n"
                            f"- **Source Molecule SMILES**: For context reference only\n"
                            f"Please analyze the potential impact of this feature on performance based primarily on the Substructure SMILES."
                        )
                    })
        
        messages.append({
            "role": "user", 
            "content": "Evidence presentation complete. Please synthesize all structural information, molecular structure diagrams (if any), PDP trends, and the global feature importance knowledge base above to output the final JSON summary report.\n\n⚠️ Special attention: Please describe each Bit feature based on the actual chemical structure observed in the molecular structure diagrams, do not guess out of thin air. If you see specific functional groups, heterocycles, or other structural features, describe them accurately."
        })

        print("The first 1000 characters of the summary messages prompt are:")
        try:
            print(str(messages)[0:1000])
        except: pass

        print("\n" + "="*60)
        print("🤖 Requesting LLM to perform comprehensive reasoning...")
        print("="*60 + "\n")
        
        try:
            from src.llm_client import retry_with_backoff
            
            response = retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                ),
                max_retries=5,
                initial_wait=30,
                description=f"SummaryAgent:{self.model}",
                input_messages=messages
            )
            
            # Save Readable Log
            self._save_readable_log(messages, response)
            
            if not response.choices or not response.choices[0].message:
                raise ValueError("LLM returned empty response")
            
            result_content = response.choices[0].message.content
            
            if not result_content or result_content.strip() == "":
                raise ValueError("LLM returned empty content")
            
            print(f"📝 [Debug] Raw LLM Response (first 500 chars): {result_content[:500]}...")
            
            # Parse JSON
            try:
                result_json = json.loads(result_content)
            except json.JSONDecodeError:
                # Fallback: Try to extract JSON from markdown code block
                import re
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result_content)
                if json_match:
                    result_json = json.loads(json_match.group(1))
                else:
                    cleaned = result_content.strip().strip("`").strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    result_json = json.loads(cleaned)

            # Print complete JSON report
            print("\n" + "="*60)
            print("📋 [SummaryAgent] Complete Summary Report (JSON):")
            print("="*60)
            print(json.dumps(result_json, indent=4, ensure_ascii=False))
            print("="*60 + "\n")
            
            # Save report
            report_path = os.path.join(get_run_dir(), "final_summary_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            print(f"✅ Summary report saved to: {report_path}")

            return result_json

        except Exception as e:
            print(f"❌ Failed to generate summary: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}