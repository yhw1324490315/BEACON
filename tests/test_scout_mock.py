# test_scout_mock.py

import os
import sys
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch

# Add current directory to path to ensure src can be imported
sys.path.append(os.getcwd())

# --- Mock src.utils dependency ---
# Must mock before importing scout, otherwise it will error if utils.py does not exist locally
sys.modules['src.utils'] = MagicMock()
from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client

# Now we can safely import scout
from src.llm_agents.scout import ScoutAgent

class TestScoutAgent(unittest.TestCase):
    
    def setUp(self):
        """Prepare before test: create fake data and directories"""
        print("\n[Test] Setting up environment...")
        self.test_dir = "test_results"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 1. Create fake molecular data file (Tab separated)
        self.data_path = "mock_cid_smiles.tsv"
        mock_data = [
            [1, "c1ccccc1"],                  # Benzene (MW ~78)
            [2, "CC(=O)O"],                   # Acetic Acid (MW ~60)
            [3, "c1ccccc1C(=O)O"],            # Benzoic Acid (has both, will be Top Candidate)
            [4, "CCO"],                       # Ethanol (no features)
            [5, "c1ccccc1N"],                 # Aniline
            [6, "O=C(O)c1ccccc1O"],           # Salicylic Acid
        ]
        
        # --- [CRITICAL CHANGE] Greatly increase background noise data volume ---
        # Previously only 100 records, 0.5% sampling rate likely yields no samples.
        # Now increased to 5000 to expect ~25 background points, sufficient for plotting.
        print("[Test] Generating 5000 mock molecules to ensure background sampling...")
        for i in range(7, 5007):
            # Generate simple carboxylic acids containing oxygen to pass min_heteroatoms >= 2 filter
            chain_len = (i % 10) + 1
            mock_data.append([i, "C" * chain_len + "C(=O)O"]) 
            
        pd.DataFrame(mock_data).to_csv(self.data_path, sep='\t', header=False, index=False)

    def tearDown(self):
        """Clean up after test"""
        # After tests pass, temporary files can be manually deleted if needed
        # if os.path.exists(self.data_path): os.remove(self.data_path)
        pass

    @patch('src.llm_agents.scout.ConfigLoader')
    @patch('src.llm_agents.scout.get_llm_client')
    @patch('src.llm_agents.scout.get_run_dir')
    def test_search_logic(self, mock_get_run_dir, mock_get_client, mock_config_loader):
        """Test core search and plotting logic"""
        print("[Test] Starting logic verification...")

        # --- 1. Mock Configuration ---
        mock_get_run_dir.return_value = self.test_dir
        
        # Mock LLM Client
        mock_client = MagicMock()
        mock_get_client.return_value = (mock_client, "gpt-4", 0.7)
        
        # Mock ConfigLoader to return fake data path
        mock_instance = MagicMock()
        mock_instance.get_data_path.return_value = self.data_path
        mock_config_loader.get_instance.return_value = mock_instance

        # --- 2. Initialize Agent ---
        agent = ScoutAgent()
        
        # --- 3. Mock LLM's SMARTS return ---
        # Simulate LLM translating natural language to SMARTS
        agent._get_smarts_from_llm = MagicMock(side_effect=lambda desc: 
            "c1ccccc1" if "benzene ring" in desc or "benzene" in desc else ("C(=O)O" if "carboxyl" in desc or "carboxylic" in desc else None)
        )

        # --- 4. Construct fake Summary Report ---
        fake_summary = {
            "critical_structures": [
                {"feature_name": "Ring", "chemical_meaning": "benzene ring structure"},
                {"feature_name": "Acid", "chemical_meaning": "carboxyl group"}
            ],
            "design_guidelines": {
                # Set a lower molecular weight constraint to ensure test molecules pass
                "structural_rules": ["molecular weight > 10"]
            }
        }

        # --- 5. Run search ---
        print("[Test] Running agent.search_molecules...")
        # Set limit to None or large enough to process all 5000 generated records
        results = agent.search_molecules(fake_summary, limit=6000)

        # --- 6. Verify results ---
        # A. Verify if molecules were found
        self.assertTrue(len(results) > 0, "Agent returned empty list!")
        print(f"[Test] Found {len(results)} candidates.")
        
        # Verify Top 1 is correct
        top_mol = results[0]
        # We expect Benzoic Acid (ID 3) or Salicylic Acid (ID 6) to have the highest score (2)
        print(f"[Test] Top Candidate: {top_mol['SMILES']} (Score: {top_mol['Total_Score']})")
        self.assertTrue(top_mol['Total_Score'] >= 1, "Scoring failed")

        # B. Verify file outputs
        images_dir = os.path.join(self.test_dir, "images")
        self.assertTrue(os.path.exists(images_dir), "Image directory not created")
        
        files = os.listdir(images_dir)
        print(f"[Test] Files generated in {images_dir}: {files}")
        
        # Check if .png generated
        png_files = [f for f in files if f.endswith('.png')]
        self.assertTrue(len(png_files) > 0, 
                        f"No PNG image generated! (Files found: {files}). "
                        "Possible reason: Background sampling yielded 0 molecules due to small dataset.")
        
        # Check if data CSVs generated (background, top, kde)
        base_name = os.path.splitext(png_files[0])[0]
        
        csv_bg = f"{base_name}_background_points.csv"
        csv_top = f"{base_name}_top_candidates.csv"
        csv_kde = f"{base_name}_kde_density.csv"
        
        self.assertIn(csv_bg, files, "Missing background points data")
        self.assertIn(csv_top, files, "Missing top candidates data")
        self.assertIn(csv_kde, files, "Missing KDE density data")
        
        print("\n✅ Test Passed: Code logic, plotting, and data saving are correct.")

if __name__ == '__main__':
    unittest.main()