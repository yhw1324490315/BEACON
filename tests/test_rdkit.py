import sys
import traceback
import os

print("="*60)
print("🧪 RDKit Core & Chemical Drawing Diagnostics")
print("="*60)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    
    # 1. Verify RDKit basic structure parsing and fingerprint computation
    print("   [Step 1] Verifying molecular structure parsing and Morgan fingerprint computation...")
    mol = Chem.MolFromSmiles('CCO')
    if mol is None:
        raise ValueError("Unable to parse SMILES 'CCO'")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    print(f"      ✅ RDKit parsed successfully. Morgan fingerprint length: {len(fp)} bits")
    
    # 2. Verify rdMolDraw2D (BEACON core SVG rendering engine)
    print("\n   [Step 2] Verifying C++ vector rendering engine (rdMolDraw2D)...")
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_content = drawer.GetDrawingText()
    
    if svg_content and "<svg" in svg_content:
        print("      ✅ Vector graphic (SVG) rendered successfully!")
        print(f"      SVG character length: {len(svg_content)} characters")
        
        # Write test SVG to the current directory
        with open("rdkit_test.svg", "w", encoding="utf-8") as f:
            f.write(svg_content)
        print("      💾 Example file saved as: rdkit_test.svg")
    else:
        raise ValueError("Generated SVG text is invalid")
        
    # 3. Check Cairo-supported legacy drawing library status and notify
    print("\n   [Step 3] Checking Cairo legacy MolToImage dependency status...")
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol)
        img.save('rdkit_test.png')
        print("      ✅ Legacy Pillow+Cairo MolToImage is available.")
    except Exception as e:
        print("      ℹ️ Legacy Cairo MolToImage is unavailable (this is normal).")
        print("      💡 Note: The BEACON multi-agent system fully relies on rdMolDraw2D to generate SVGs + browser headless screenshot to generate PNGs,")
        print("         therefore, this system DOES NOT require complex low-level dependencies like Cairo. You can ignore this warning.")

    print("\n" + "="*60)
    print("🎉 RDKit test successful! Core dependencies are ready.")
    print("="*60)

except Exception as e:
    print('\n❌ RDKit core test failed:', e)
    traceback.print_exc()
    sys.exit(1)
