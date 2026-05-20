# src/llm_agents/scout.py

import os
import pandas as pd
import json
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from openai import OpenAI
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Crippen, Lipinski
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import random

from src.utils import ConfigLoader, get_run_dir, get_prompt, get_llm_client

try:
    from rdkit.Chem import RDConfig
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORES = True
except BaseException:
    HAS_SASCORES = False

# ==============================================================================
# --- I. Visual Configuration (Nature/Science Ultimate Edition) ---
# ==============================================================================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.weight'] = 'normal'

# Resolution
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600

# Font sizes significantly increased
mpl.rcParams['axes.labelsize'] = 48    # Axis labels (Huge)
mpl.rcParams['xtick.labelsize'] = 40   # Tick labels
mpl.rcParams['ytick.labelsize'] = 40
mpl.rcParams['legend.fontsize'] = 34   # Legend
mpl.rcParams['font.size'] = 34         # Global default

# Line bolding
mpl.rcParams['axes.linewidth'] = 4.0      # Borders thicker
mpl.rcParams['xtick.major.width'] = 4.0   # Tick lines thicker
mpl.rcParams['ytick.major.width'] = 4.0
mpl.rcParams['xtick.major.size'] = 14     # Tick lines longer
mpl.rcParams['ytick.major.size'] = 14


def _worker_process_chunk(chunk_data):
    """
    Independent process work unit: processes one data chunk.
    [Performance Optimized] Using list iteration instead of iterrows(), performance improved by 3-5x.
    """
    df_chunk, constraints, smarts_tuples = chunk_data

    candidates = []
    bg_samples = []
    
    # Pre-compile SMARTS
    patterns = []
    for s, bid, d in smarts_tuples:
        p = Chem.MolFromSmarts(s)
        if p: patterns.append((p, bid, d))
    
    min_mw = constraints.get('min_mw', 0)
    max_mw = constraints.get('max_mw', 9999)
    min_heteroatoms = constraints.get('min_heteroatoms', 0)
    max_logp = constraints.get('max_logp', 99)
    forbidden_atoms = constraints.get('forbidden_atoms', []) # e.g., [7] represents nitrogen
    required_atoms = constraints.get('required_atoms', [])   # e.g., [8] represents oxygen
    max_fraction_csp3 = constraints.get('max_fraction_csp3', 1.0)
    max_sascore = constraints.get('max_sascore', 10.0)       # Add synthetic feasibility filtering
    allowed_elements = constraints.get('allowed_elements', []) # e.g., [6,1,8] = C/H/O only
    
    # Extract to native lists, avoiding pandas iterrows() Series overhead
    id_list = df_chunk['ID'].tolist()
    smi_list = df_chunk['SMILES'].tolist()
    
    # Batch generate random numbers for background sampling
    bg_rand = np.random.random(len(smi_list))
    
    for idx in range(len(smi_list)):
        try:
            smi = smi_list[idx]
            if not isinstance(smi, str): continue

            # A. Basic molecular parsing and molecular weight filtering
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue

            mw = Descriptors.MolWt(mol)
            if not (min_mw <= mw <= max_mw):
                continue

            # Atom type filtering (hard interception)
            actual_atom_nums = set(atom.GetAtomicNum() for atom in mol.GetAtoms())
            
            # Whitelist priority: if allowed_elements specified, keep only molecules containing these elements
            if allowed_elements and not actual_atom_nums.issubset(set(allowed_elements)):
                continue
            
            # Blacklist
            if forbidden_atoms and any(fa in actual_atom_nums for fa in forbidden_atoms):
                continue
            # Required atoms
            if required_atoms and not all(ra in actual_atom_nums for ra in required_atoms):
                continue
            
            # B. Reactivity filtering: heteroatom counting (oxygen and nitrogen)
            if min_heteroatoms > 0:
                num_N_O = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in (7, 8))
                if num_N_O < min_heteroatoms:
                    continue  # Insufficient heteroatoms, discard directly!

            # C. Background sampling (0.5%)
            if bg_rand[idx] < 0.005:
                bg_samples.append({'SMILES': smi})

            # D. Structural feature scoring
            score = 0
            details = []
            
            for pat, bit_id, desc in patterns:
                if mol.HasSubstructMatch(pat):
                    score += 1
                    details.append(f"[{desc} (Bit_{bit_id})]")
            
            # E. Save and advanced descriptor calculation
            if score > 0:
                # Calculate advanced physicochemical descriptors
                logp = Crippen.MolLogP(mol)
                if logp > max_logp:
                    continue  # Lipophilicity too high, discard directly!
                    
                num_aromatic_rings = Descriptors.NumAromaticRings(mol)
                fraction_csp3 = Descriptors.FractionCSP3(mol)
                
                # Added: sp3 hybridized carbon ratio hard filter
                if fraction_csp3 > max_fraction_csp3:
                    continue
                    
                tpsa = Descriptors.TPSA(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                
                # Added: synthetic accessibility score constraint (SA_Score)
                sa_score = sascorer.calculateScore(mol) if HAS_SASCORES else 0.0
                if sa_score > max_sascore:
                    continue
                
                candidates.append({
                    'ID': id_list[idx],
                    'SMILES': smi,
                    'MW': mw,
                    'Num_Aromatic_Rings': num_aromatic_rings,
                    'FractionCSP3': fraction_csp3,
                    'LogP': logp,
                    'TPSA': tpsa,
                    'HBD': hbd,
                    'HBA': hba,
                    'SA_Score': sa_score,
                    'Total_Score': score,
                    'Matched_Details': "; ".join(details)
                })
        except Exception:
            continue
            
    return candidates, bg_samples


# ==========================================
#  Main Class: ScoutAgent
# ==========================================
class ScoutAgent:
    def __init__(self):
        self.config_loader = ConfigLoader.get_instance()
        self.client, self.model, self.temperature = get_llm_client()
        self.run_dir = get_run_dir()
        self.img_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

    def _design_search_strategy(self, summary_report, initial_query):
        """Invoke LLM to act as a parameter designer, dynamically deciding the hard boundaries and sorting strategy for molecular retrieval"""
        import json
        from src.llm_client import retry_with_backoff
        from src.utils import get_prompt
        
        template = get_prompt('scout_strategy_designer_prompt')
        if not template:
            print("⚠️ Failed to find scout_strategy_designer_prompt in prompts.yaml.")
            template = ""
            
        summary_report_core = json.dumps(summary_report.get('design_guidelines', summary_report), ensure_ascii=False)
        prompt = template.format(
            initial_query=initial_query,
            summary_report_core=summary_report_core
        )
        
        try:
            msgs = [{"role": "user", "content": prompt}]
            resp = retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=0.3
                ),
                max_retries=3,
                initial_wait=10,
                description=f"Scout-Strategy:{self.model}",
                input_messages=msgs
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```[\s\S]*", "", content)
            strategy = json.loads(content)
            print(f"🧠 [Scout] 🔍 LLM 'Parameter Designer' deeply analyzed and set a custom retrieval strategy!")
            print(f"   -> 💡 Rationale: {strategy.get('rationale', '')}")
            print(f"   -> ⚖️ MW Range: {strategy.get('min_mw', 50)} - {strategy.get('max_mw', 500)} Da")
            print(f"   -> ⚛️ Min Heteroatoms (O/N): {strategy.get('min_heteroatoms', 0)}")
            print(f"   -> 💧 Max LogP: {strategy.get('max_logp', 99)}")
            print(f"   -> 🚫 Forbidden Atoms: {strategy.get('forbidden_atoms', [])} | ✅ Required Atoms: {strategy.get('required_atoms', [])}")
            print(f"   -> 🧩 Allowed Elements: {strategy.get('allowed_elements', [])} (Empty=No Limit)")
            print(f"   -> 🧬 Max sp3: {strategy.get('max_fraction_csp3', 1.0)} | 🧪 Max SA_Score: {strategy.get('max_sascore', 10.0)}")
            print(f"   -> 🔀 Sorting Weights: {strategy.get('sort_columns')} (Ascending: {strategy.get('sort_ascending')})")
            return strategy
        except Exception as e:
            print(f"⚠️ [Scout] Dynamic strategy design callback failed or parsing error: {e}. Falling back to default configuration.")
            return {
                "rationale": "Default Fallback",
                "min_mw": 0, "max_mw": 350, "min_heteroatoms": 2, "max_logp": 4.0,
                "forbidden_atoms": [], "required_atoms": [], "allowed_elements": [6, 1, 8, 7],
                "max_fraction_csp3": 0.5, "max_sascore": 3.5,
                "sort_columns": ["Total_Score", "LogP", "MW"],
                "sort_ascending": [False, True, True]
            }

    def _get_smarts_from_llm(self, desc):
        """Invoke LLM to get SMARTS"""
        from src.llm_client import retry_with_backoff
        
        prompt_template = get_prompt('scout_smarts_prompt')
        prompt = prompt_template.format(desc=desc) if prompt_template else f"What is the simplest RDKit SMARTS string for the chemical group '{desc}'? Return only the string."

        try:
            msgs = [{"role": "user", "content": prompt}]
            resp = retry_with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs
                ),
                max_retries=5,
                initial_wait=15,
                description=f"Scout-SMARTS:{self.model}",
                input_messages=msgs
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            content = re.sub(r"`.*?`", "", content)
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            candidate = lines[-1].split(' ')[0] 
            return candidate
        except Exception as e:
            print(f"⚠️ [Scout] Failed to get SMARTS: {e}")
            return None

    def _compute_chemical_space(self, df_bg, df_top):
        """Calculate PCA"""
        print("🧪 [Vis] Calculating chemical space coordinates (PCA)...")
        sample_size = min(2000, len(df_bg))
        bg_sample = df_bg.sample(n=sample_size, random_state=42).copy() if len(df_bg) > 0 else pd.DataFrame()
        
        if bg_sample.empty and df_top.empty:
            return None, None

        combined_smiles = bg_sample['SMILES'].tolist() + df_top['SMILES'].tolist()
        
        fps = []
        valid_indices = []
        
        for i, smi in enumerate(combined_smiles):
            m = Chem.MolFromSmiles(smi)
            if m:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                fps.append(np.array(fp))
                valid_indices.append(i)
                
        if not fps: return None, None

        X = np.array(fps)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        n_bg_total = len(bg_sample)
        n_bg_valid = sum(1 for i in valid_indices if i < n_bg_total)
        
        bg_pca = X_pca[:n_bg_valid]
        top_pca = X_pca[n_bg_valid:]
        
        return bg_pca, top_pca

    def _plot_chemical_space_trajectory(self, bg_pca, top_pca, save_name="chem_space.png"):
        """Plot and save chemical space map (no title, ultra large font)"""
        if bg_pca is None or top_pca is None: return

        print("🎨 [Vis] Generating chemical space map (Nature Style - Big Fonts)...")
        
        base_name = os.path.splitext(save_name)[0]
        data_prefix = os.path.join(self.img_dir, base_name)
        
        if len(bg_pca) > 0:
            df_bg_save = pd.DataFrame(bg_pca, columns=['PC1', 'PC2'])
            df_bg_save.to_csv(f"{data_prefix}_background_points.csv", index=False)

        if len(top_pca) > 0:
            df_top_save = pd.DataFrame(top_pca, columns=['PC1', 'PC2'])
            df_top_save['Step'] = range(1, len(top_pca) + 1)
            df_top_save.to_csv(f"{data_prefix}_top_candidates.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        if len(bg_pca) > 5:
            x_bg, y_bg = bg_pca[:, 0], bg_pca[:, 1]
            try:
                xy = np.vstack([x_bg, y_bg])
                kde = gaussian_kde(xy)
                
                xmin, xmax = x_bg.min() - 1, x_bg.max() + 1
                ymin, ymax = y_bg.min() - 1, y_bg.max() + 1
                X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions).T, X.shape)
                
                df_kde_save = pd.DataFrame({'Grid_X': X.ravel(), 'Grid_Y': Y.ravel(), 'Density_Z': Z.ravel()})
                df_kde_save.to_csv(f"{data_prefix}_kde_density.csv", index=False)
                
                ax.contourf(X, Y, Z, levels=10, cmap='Blues', alpha=0.6, zorder=1)
            except Exception as e:
                print(f"⚠️ KDE Error: {e}")
                ax.scatter(x_bg, y_bg, color='#B0C4DE', alpha=0.5, s=80) 
        
        x_top, y_top = None, None
        
        if len(top_pca) > 0:
            x_top, y_top = top_pca[:, 0], top_pca[:, 1]
            ax.plot(x_top, y_top, color='#d48806', linewidth=4.5, linestyle='--', alpha=0.8, zorder=2, label='Search Path')
            ax.scatter(x_top, y_top, marker='*', s=500, color='gold', edgecolors='#d48806', linewidth=2.5, zorder=3, label='Top Candidates')

        ax.set_xlabel('Principal Component 1', family='Arial', weight='bold', fontsize=48)
        ax.set_ylabel('Principal Component 2', family='Arial', weight='bold', fontsize=48)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4.0)
            
        ax.tick_params(width=4.0, length=14, labelsize=40)
        
        ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, framealpha=0.9,
                  facecolor='white', edgecolor='black', fontsize=34,
                  handlelength=2.5, borderpad=0.8, labelspacing=0.8)
        
        ax.grid(False)
        
        plt.tight_layout()
        base_name_no_ext = os.path.splitext(save_name)[0]
        no_anno_path = os.path.join(self.img_dir, f"{base_name_no_ext}_no_annotation.png")
        plt.savefig(no_anno_path, bbox_inches='tight')
        print(f"✅ [Vis] No annotation image saved: {no_anno_path}")

        if x_top is not None and y_top is not None:
            ax.annotate('Best Candidate',
                xy=(x_top[0], y_top[0]), 
                xycoords='data',
                xytext=(x_top[0], y_top[0] - 1.5), 
                textcoords='data',
                ha='center', 
                arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=12), 
                fontsize=38, 
                family='Arial',
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=3, alpha=0.9))
        
        full_save_path = os.path.join(self.img_dir, save_name)
        plt.savefig(full_save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ [Vis] Image saved: {full_save_path}")

    # ==========================================
    #  Core Entry: search_molecules
    # ==========================================
    def _fetch_molecule_names(self, df):
        """Call PubChem API to batch fetch commercial/common names of molecules based on CID"""
        import requests
        print("\n🌐 [Scout] Calling PubChem REST API to fetch commercial and common names for candidates...")
        cids = df['ID'].dropna().astype(int).tolist()
        
        cids = cids[:100]
        if not cids: return df
        
        cid_str = ",".join(map(str, cids))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/Title/JSON"
        
        name_dict = {cid: "Synthetic Exclusive Molecule" for cid in cids}
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("PropertyTable", {}).get("Properties", []):
                    cid = item.get("CID")
                    title = item.get("Title", "Common Synthetic Fragment")
                    if cid: name_dict[int(cid)] = title
        except Exception as e:
            print(f"⚠️ [Scout] Failed to fetch molecular names: {e}")
            
        df['Name'] = df['ID'].map(name_dict).fillna("Special Synthetic Block")
        
        cols = df.columns.tolist()
        if 'Name' in cols:
            cols.insert(1, cols.pop(cols.index('Name')))
            df = df[cols]
        return df

    def search_molecules(self, summary_report, limit=None, max_mw=None, initial_query=""):
        print(f"🚀 [Scout] Launching high-performance parallel search engine (multi-core)...")
        target_path = self.config_loader.get_data_path('cid_smiles')
        if not target_path or not os.path.exists(target_path):
            print(f"❌ [Scout] Data file not found: {target_path}")
            return []

        # Acting as 'Parameter Designer'
        strategy = self._design_search_strategy(summary_report, initial_query)
        
        constraints = {
            'min_mw': strategy.get('min_mw', 0),
            'max_mw': strategy.get('max_mw', 500),
            'min_heteroatoms': strategy.get('min_heteroatoms', 0),  
            'max_logp': strategy.get('max_logp', 99),
            'forbidden_atoms': strategy.get('forbidden_atoms', []),
            'required_atoms': strategy.get('required_atoms', []),
            'allowed_elements': strategy.get('allowed_elements', []),
            'max_fraction_csp3': strategy.get('max_fraction_csp3', 1.0),
            'max_sascore': strategy.get('max_sascore', 10.0)
        }
        
        if max_mw is not None:
            constraints['max_mw'] = min(constraints['max_mw'], max_mw)

        # Filter non-structural features
        print("⚙️  [Pre-compile] Parsing feature SMARTS (only real Bit features)...")
        critical_structs = summary_report.get("critical_structures", [])
        if not critical_structs:
            critical_structs = summary_report.get("critical_features_analysis", [])

        # Verified Bit -> SMARTS correction table
        # Planner often decodes Morgan Fingerprint Bits incorrectly (e.g. decoding anhydride as carboxylic acid),
        # so we overwrite the LLM's incorrect decoding with a human-verified SMARTS list here.
        # Each Bit can have multiple SMARTS variants (e.g., anhydride + carboxylic acid) to maximize recall.
        KNOWN_BIT_SMARTS = {
            "Bit_456": [
                ("C(=O)OC(=O)", "Aromatic Anhydride"),
                ("cC(=O)O",     "Aromatic Carboxylic Acid"),
            ],
            "Bit_1925": [
                ("c(c)C(=O)c(c)", "Diaryl Ketone"),
            ],
            "Bit_1984": [
                ("c1ccc2ccccc2c1", "Fused Aromatic"),
                ("cc(c)c",         "Aromatic Carbon"),
            ],
            "Bit_1039": [
                ("c1cccc2ccccc12", "Aromatic Carbon Backbone"),
            ],
            "Bit_352": [
                ("COC",    "Ether Linkage"),
                ("c1ccoc1", "Furan Ring"),
            ],
            "Bit_831": [
                ("Bc(c)c", "Boron-containing Aromatic"),
            ],
            "Bit_1917": [
                ("C=O",    "Carbonyl"),
            ],
        }

        smarts_tuples = []
        skipped_features = []
        
        for struct in critical_structs:
            bit_id = struct.get('feature_name') or struct.get('bit_id', 'Unknown')
            desc = struct.get('chemical_meaning') or struct.get('chemical_desc', '')
            feat_type = struct.get('type', '')
            
            if not desc or "undecoded" in desc.lower() or "n/a" in desc.lower() or desc.strip() == "*":
                continue
            
            bit_id_str = str(bit_id)
            is_real_bit = bit_id_str.startswith("Bit_") and bit_id_str.replace("Bit_", "").isdigit()
            
            non_structural_keywords = [
                "molecular weight", "mw",
                "test_n", "test_o", "test_c", "test_h",
                "temperature", "time",
                "pre1_", "pre2_", "step_", "ex",
                "melting point", "boiling point",
                "tpsa", "log p",
            ]
            
            desc_lower = desc.lower()
            bit_id_lower = bit_id_str.lower()
            is_non_structural = any(kw in desc_lower or kw in bit_id_lower for kw in non_structural_keywords)
            
            if feat_type.lower() in ('condition', 'process'):
                is_non_structural = True
            
            if is_non_structural and not is_real_bit:
                skipped_features.append(f"{bit_id}: {desc}")
                continue
            
            # Prioritize using correction table
            if bit_id_str in KNOWN_BIT_SMARTS:
                for smarts_str, smarts_desc in KNOWN_BIT_SMARTS[bit_id_str]:
                    print(f"  -> {bit_id_str}: {smarts_desc} => {smarts_str} [Correction Table]")
                    smarts_tuples.append((smarts_str, bit_id_str, smarts_desc))
            else:
                smarts = self._get_smarts_from_llm(desc)
                if smarts and smarts != '*' and smarts != '[*]':
                    print(f"  -> {bit_id}: {desc} => {smarts}")
                    smarts_tuples.append((smarts, bit_id, desc))
        
        if skipped_features:
            print(f"  ⏭️ Skipped {len(skipped_features)} non-structural features (not applicable for substructure search):")
            for sf in skipped_features[:5]:
                print(f"     - {sf}")
            if len(skipped_features) > 5:
                print(f"     ... and another {len(skipped_features) - 5} others")

        if not smarts_tuples:
            print("  ⚠️ No valid structural features available for search.")
            return []

        CHUNK_SIZE = 100000 
        max_workers = min(24, max(1, multiprocessing.cpu_count() - 4))
        print(f"🔥 [Parallel] Launching {max_workers} parallel worker processes...")
        print(f"📋 [Task Details] All worker processes will perform the same screening tasks in parallel:")
        print(f"   1. Molecular weight limits: {constraints.get('min_mw', 0)} - {constraints.get('max_mw', 'Inf')}")
        print(f"   2. Heteroatom (O/N) lower bound: {constraints.get('min_heteroatoms', 0)}")
        print(f"   3. LogP upper bound (lipophilicity interception): {constraints.get('max_logp', 'Inf')}")
        print(f"   4. Element blacklist: {constraints.get('forbidden_atoms', [])} | Required elements: {constraints.get('required_atoms', [])}")
        print(f"   4b. 💎 Element whitelist (only allowed): {constraints.get('allowed_elements', [])} (Empty=No Limit)")
        print(f"   5. Advanced interception: SA_Score (synthesis difficulty) <= {constraints.get('max_sascore', 'Inf')} | sp3 ratio <= {constraints.get('max_fraction_csp3', 'None')}")
        
        feat_list = '\n'.join([f"      -> {t[2]}" for t in smarts_tuples]) if smarts_tuples else "      -> (None)"
        print(f"   6. Target structural features ({len(smarts_tuples)}):\n{feat_list}")
        print(f"📄 [Data] Streaming file read: {target_path}")

        global_candidates = []
        global_bg_samples = [] 

        estimated_total = limit if limit else 100_000_000 
        pbar = tqdm(total=estimated_total, unit="mol", desc="Mining")

        from concurrent.futures import FIRST_COMPLETED, wait
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            reader = pd.read_csv(target_path, sep='\t', header=None, names=['ID', 'SMILES'], chunksize=CHUNK_SIZE, iterator=True)
            lines_read = 0
            
            try:
                for chunk in reader:
                    if limit and lines_read >= limit: break
                    
                    if len(futures) >= max_workers * 3:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in done:
                            try:
                                res_cand, res_bg = f.result()
                                global_candidates.extend(res_cand)
                                global_bg_samples.extend(res_bg)
                                pbar.update(CHUNK_SIZE)
                            except Exception as e:
                                print(f"Worker Error: {e}")
                                
                        # Prune in time to reduce memory usage, applying LLM dynamic strategy
                        if len(global_candidates) > 5000:
                            temp_df = pd.DataFrame(global_candidates)
                            try:
                                temp_df = temp_df.sort_values(
                                    by=strategy.get('sort_columns', ['Total_Score', 'LogP', 'MW']), 
                                    ascending=strategy.get('sort_ascending', [False, True, True])
                                ).head(2000)
                            except:
                                temp_df = temp_df.sort_values(by=['Total_Score', 'MW'], ascending=[False, True]).head(2000)
                            global_candidates = temp_df.to_dict('records')

                    future = executor.submit(_worker_process_chunk, (chunk, constraints, smarts_tuples))
                    futures.add(future)
                    lines_read += len(chunk)
                    
                # Wait for all remaining tasks
                for f in as_completed(futures):
                    try:
                        res_cand, res_bg = f.result()
                        global_candidates.extend(res_cand)
                        global_bg_samples.extend(res_bg)
                        pbar.update(CHUNK_SIZE)
                    except Exception as e: 
                        print(f"Worker Error: {e}")
                        
                    if len(global_candidates) > 5000:
                        temp_df = pd.DataFrame(global_candidates)
                        try:
                            temp_df = temp_df.sort_values(
                                by=strategy.get('sort_columns', ['Total_Score', 'LogP', 'MW']), 
                                ascending=strategy.get('sort_ascending', [False, True, True])
                            ).head(2000)
                        except:
                            temp_df = temp_df.sort_values(by=['Total_Score', 'MW'], ascending=[False, True]).head(2000)
                        global_candidates = temp_df.to_dict('records')

            except StopIteration: pass
            except Exception as e: print(f"❌ Streaming file read interrupted: {e}")
            finally: pbar.close()

        print(f"\n✅ Scan complete. Found a total of {len(global_candidates)} potential candidate molecules.")
        if not global_candidates: return []

        df_final = pd.DataFrame(global_candidates)
        
        # Final sorting scheme: custom strategy generated by LLM
        try:
            df_final = df_final.sort_values(
                by=strategy.get('sort_columns', ['Total_Score', 'LogP', 'MW']), 
                ascending=strategy.get('sort_ascending', [False, True, True])
            )
        except Exception as e:
            print(f"⚠️ Dynamic sorting failed: {e}, enabling default sorting")
            df_final = df_final.sort_values(by=['Total_Score', 'MW'], ascending=[False, True])
            
        df_save = df_final.head(100).copy()
        
        # Fetch commercial names
        df_save = self._fetch_molecule_names(df_save)
        
        # Update selection reason, showing polar and water solubility parameters, including Name and SA_Score
        df_save['Selection_Reason'] = df_save.apply(
            lambda row: f"Name: {row.get('Name','Unknown')} | Score: {row['Total_Score']} | SA_Score: {row.get('SA_Score', 0):.1f} | MW: {row['MW']:.1f} | LogP: {row['LogP']:.2f} | TPSA: {row['TPSA']:.1f}", 
            axis=1
        )

        try:
            df_bg = pd.DataFrame(global_bg_samples)
            if not df_bg.empty:
                bg_pca, top_pca = self._compute_chemical_space(df_bg, df_save.head(20))
                timestamp = int(time.time())
                self._plot_chemical_space_trajectory(bg_pca, top_pca, save_name=f"chem_space_{timestamp}.png")
            else:
                print("⚠️ [Vis] Skip plotting: No background molecules sampled.")
        except Exception as e:
            print(f"⚠️ Plotting error: {e}")

        timestamp = int(time.time())
        save_path = os.path.join(self.run_dir, f"scout_candidates_{timestamp}.csv")
        df_save.to_csv(save_path, index=False)
        print(f"💾 Results saved: {save_path}")
        if not df_save.empty:
            print(f"🏆 Highest score molecule: {df_save.iloc[0]['SMILES']} (Score: {df_save.iloc[0]['Total_Score']})")

        return df_save.to_dict(orient='records')