import os
import sys
import shutil
import json
import joblib
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure src module can be imported
sys.path.append(os.getcwd())

from src.llm_agents.optimizer import OptimizerAgent
from src.utils import ConfigLoader

# ==========================================
# 1. Define Mock Model and Scaler (Mock Objects)
# ==========================================
class DummyModel:
    """A dummy regression model that returns pseudo-random predictions based on input features"""
    def predict(self, X):
        # To make the 3D surface plot look meaningful and attractive, we introduce some pattern
        # Assume X is a numpy array or DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Simple simulation: Score depends on temperature (col -5) and time (col -4)
        # and added random noise
        n_samples = X.shape[0]
        feature_a = X[:, -5] # Temperature
        feature_b = X[:, -4] # Time
        
        # Generate peaky surface data
        base = np.sin(feature_a / 100.0) * np.cos(feature_b / 10.0)
        score = 500 + 200 * base + np.random.normal(0, 10, n_samples)
        return np.abs(score)

class DummyScaler:
    """Dummy scaler that performs no modifications"""
    def transform(self, X):
        return X

# ==========================================
# 2. Test Environment Setup Functions
# ==========================================
TEST_DIR = "test_env_optimizer"
DATA_DIR = os.path.join(TEST_DIR, "data")
RUN_DIR = os.path.join(TEST_DIR, "run")

def setup_test_environment():
    """Create a temporary directory and save dummy .pkl model files"""
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(DATA_DIR)
    os.makedirs(RUN_DIR)

    print(f"🛠️  Constructing test environment: {TEST_DIR}")

    # 1. Save fake Feature Names (2199 dimensions, matching the actual setup)
    feature_names = [f"Bit_{i}" for i in range(2048)]
    feature_names.extend([
        "Step_1_Temperature", "Step_1_Time", "Step1_Carbon_Dots_Dosage", 
        "Step_2_Temperature", "Step 2_Time", "Ratio", 
        "Preparation_Method_Code_1", "Preparation_Method_Code_2",
        "Step_1_Reaction_Code_1", "Step_1_Reaction_Code_2",
        "Pre1_C", "Pre1_H", "Pre1_O", "Pre1_N", 
        "Test_Temperature", "Ex"
    ])
    
    # Write feature names to match agent expectations (em_xxx, life_xxx)
    with open(os.path.join(DATA_DIR, "em_feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    
    with open(os.path.join(DATA_DIR, "life_feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # 2. Save dummy Model and Scaler (.pkl)
    dummy_model = DummyModel()
    dummy_scaler = DummyScaler()

    joblib.dump(dummy_model, os.path.join(DATA_DIR, "trained_em_model.pkl"))
    joblib.dump(dummy_scaler, os.path.join(DATA_DIR, "em_scaler.pkl"))
    
    joblib.dump(dummy_model, os.path.join(DATA_DIR, "trained_life_model.pkl"))
    joblib.dump(dummy_scaler, os.path.join(DATA_DIR, "life_scaler.pkl"))

    print("✅ Dummy model files successfully generated (Mock Models Saved)")

# ==========================================
# 3. Main Test Logic
# ==========================================
def run_test():
    setup_test_environment()

    # --- Patch ConfigLoader & get_run_dir ---
    # Intercept path retriever functions in utils to point to our test folder
    
    with patch('src.utils.ConfigLoader.get_model_path') as mock_get_path, \
         patch('src.utils.get_run_dir') as mock_get_run_dir, \
         patch('src.llm_agents.optimizer.get_run_dir') as mock_agent_run_dir:

        # 1. Setup Mock return values
        mock_get_path.side_effect = lambda x: os.path.join(DATA_DIR, "trained_em_model.pkl")
        mock_get_run_dir.return_value = RUN_DIR
        mock_agent_run_dir.return_value = RUN_DIR

        # 2. Initialize Agent
        print("\n🤖 Initializing OptimizerAgent...")
        agent = OptimizerAgent()
        
        # Override local data directory (double assurance)
        agent.data_dir = DATA_DIR

        # 3. Prepare test input data
        summary_report = {
            "target_property": "emission wavelength and lifetime",
            "critical_features": ["Amide", "C=O"]
        }

        candidates = [
            {"Name": "Test_Mol_A", "SMILES": "CC(=O)O"},      # Acetic Acid
            {"Name": "Test_Mol_B", "SMILES": "c1ccccc1N"},    # Aniline
            {"Name": "Test_Mol_C", "SMILES": "NCC(=O)O"}      # Glycine
        ]

        # 4. Run optimization
        print("🚀 Running optimize()...")
        # In mock tests, this runs relatively fast while calculating descriptors
        result = agent.optimize(summary_report, candidates)

        # 5. Verify results
        print("\n" + "="*40)
        print("🧪 Test Result Verification")
        print("="*40)

        # Verify CSV is generated
        csv_path = os.path.join(RUN_DIR, "logs", "Total_Model_Input_Features.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ CSV file generated successfully: {csv_path}")
            print(f"   -> Data shape: {df.shape}")
        else:
            print(f"❌ CSV file not found!")

        # Verify 3D surface images generated (Nature style)
        img_dir = os.path.join(RUN_DIR, "images")
        images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        if images:
            print(f"✅ Generated {len(images)} 3D surface images successfully:")
            for img in images:
                print(f"   -> {img}")
            print("🎨 Please open test_env_optimizer/run/images to check the rendered styles!")
        else:
            print("⚠️ Warning: No images generated. This could happen if random feature variance is too small and filtered out by std checks in _batch_plot_all_surfaces.")
            print("   (This is possible with DummyModel; re-running or adjusting variance solves it.)")

        # Print top recommendation result
        print("\n📋 Final output recommendation (Top Recommendation):")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n❌ Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()