"""
Test Script: Verify SVG to PNG Conversion Functionality (including Browser Fallback)
"""

import os
import sys
import glob
import base64
import subprocess

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("="*60)
print("🧪 SVG to PNG Conversion Functionality Test (Browser Fallback)")
print("="*60)

# 1. Simulate Environment: Force disable Python graphical library to force using Browser
print("\n📂 [Step 1] Configuring simulation environment...")
# Do not import svglib, directly test browser logic

# Import SummaryAgent class (but do not instantiate fully to avoid LLM initialization)
from src.llm_agents.summary import SummaryAgent

# Create a temporary class to test the method
class TestSummaryAgent(SummaryAgent):
    def __init__(self):
        # Bypass parent class initialization, no LLM connection
        pass

agent = TestSummaryAgent()

# 2. Find or create SVG files for testing
print("\n📂 [Step 2] Finding or creating test SVG files...")
experiments_dir = os.path.join(project_root, "experiments")
svg_pattern = os.path.join(experiments_dir, "**", "*.svg")
svg_files = glob.glob(svg_pattern, recursive=True)

temp_svg_created = False
if len(svg_files) == 0:
    print("   ℹ️ No existing SVG files found, creating a temporary mock_molecule.svg...")
    temp_dir = os.path.join(project_root, "test_temp")
    os.makedirs(temp_dir, exist_ok=True)
    test_svg = os.path.join(temp_dir, "mock_molecule.svg")
    # Write a simple SVG (a pretty red circle representing atomic core)
    mock_svg_content = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FFFFFF"/>
  <circle cx="200" cy="200" r="100" stroke="#FF0000" stroke-width="8" fill="#FFEAEA"/>
  <text x="200" y="215" font-family="Arial" font-size="40" font-weight="bold" text-anchor="middle" fill="#FF0000">C-Core</text>
</svg>
"""
    with open(test_svg, "w", encoding="utf-8") as f:
        f.write(mock_svg_content)
    temp_svg_created = True
    print(f"   ✅ Temporary SVG created: {os.path.relpath(test_svg, project_root)}")
else:
    test_svg = svg_files[0]
    print(f"   Selected test file: {os.path.relpath(test_svg, project_root)}")

# 3. Test locating browser
print("\n🔎 [Step 3] Testing system browser location...")
browser_path = agent._get_browser_executable()
if browser_path:
    print(f"   ✅ Browser found at: {browser_path}")
else:
    print(f"   ❌ Edge or Chrome browser not found")
    print("   ⚠️ Test might fail")

# 4. Test browser screenshot function
print("\n📸 [Step 4] Testing browser screenshot conversion (headless)...")
test_png = test_svg.replace('.svg', '_browser_test.png')

# Delete old PNG if it exists for test cleanliness
if os.path.exists(test_png):
    os.remove(test_png)

try:
    print(f"   Converting: {os.path.basename(test_svg)}")
    success = agent._convert_svg_using_browser(test_svg, test_png)
    
    if success and os.path.exists(test_png):
        size = os.path.getsize(test_png)
        print(f"   ✅ Conversion successful!")
        print(f"   Generated file: {os.path.relpath(test_png, project_root)}")
        print(f"   File size: {size} bytes")
        
        if size < 1000:
            print("   ⚠️ Warning: File size is extremely small, could be blank")
        else:
            print("   ✨ Image seems valid")
            
        # Clean up
        os.remove(test_png)
        print("   🧹 Test file cleaned up")
    else:
        print("   ❌ Conversion failed")

except Exception as e:
    print(f"   ❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()

# 5. Test comprehensive _svg_to_png method
print("\n📸 [Step 5] Testing comprehensive _svg_to_png method...")
# Force HAS_SVG_CONVERTER to False to test browser fallback path
import src.llm_agents.summary as summary_module
summary_module.HAS_SVG_CONVERTER = False
print(f"   Forced HAS_SVG_CONVERTER = False")

base64_str = agent._svg_to_png(test_svg)

if base64_str:
    print(f"   ✅ Comprehensive method successfully called")
    print(f"   Returned Base64 string length: {len(base64_str)}")
else:
    print(f"   ❌ Comprehensive method call failed")

if temp_svg_created:
    print("\n🧹 [Clean-up] Cleaning up temporary SVG and directories...")
    if os.path.exists(test_svg):
        os.remove(test_svg)
    temp_dir = os.path.dirname(test_svg)
    # Remove any temp png files
    for ext in ["_browser_test.png", ".png"]:
        tmp_png = test_svg.replace(".svg", ext)
        if os.path.exists(tmp_png):
            os.remove(tmp_png)
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
        os.rmdir(temp_dir)
    print("   ✅ Temporary files and directory cleanup completed")

print("\n" + "="*60)
print("🎉 Test Complete!")
print("="*60)
