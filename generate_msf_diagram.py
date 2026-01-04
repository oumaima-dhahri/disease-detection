#!/usr/bin/env python3
"""
Generate MSF Module Architecture Diagram using PlantUML
Requires: plantuml (install via: pip install plantuml or use online service)
"""

import subprocess
import os

def generate_diagram(puml_file, output_format='png'):
    """
    Generate diagram from PlantUML file
    
    Args:
        puml_file: Path to .puml file
        output_format: 'png', 'svg', 'pdf', or 'eps'
    """
    try:
        # Try using plantuml command line tool
        cmd = ['plantuml', f'-t{output_format}', puml_file]
        subprocess.run(cmd, check=True)
        print(f"✅ Diagram generated: {puml_file.replace('.puml', f'.{output_format}')}")
    except FileNotFoundError:
        print("⚠️  PlantUML command not found. Using online service...")
        print(f"\nTo generate the diagram:")
        print(f"1. Go to: http://www.plantuml.com/plantuml/uml/")
        print(f"2. Copy the content of: {puml_file}")
        print(f"3. Paste and generate")
        print(f"\nOr install PlantUML:")
        print(f"  - Windows: Download from http://plantuml.com/download")
        print(f"  - Linux: sudo apt-get install plantuml")
        print(f"  - Mac: brew install plantuml")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating diagram: {e}")

if __name__ == "__main__":
    # Generate all three versions
    files = [
        "msf_module_architecture.puml",
        "msf_module_architecture_horizontal.puml",
        "msf_module_architecture_simple.puml"
    ]
    
    print("Generating MSF Module Architecture Diagrams...\n")
    
    for puml_file in files:
        if os.path.exists(puml_file):
            print(f"Processing: {puml_file}")
            generate_diagram(puml_file, 'png')
            print()
        else:
            print(f"⚠️  File not found: {puml_file}\n")
    
    print("✅ Done! Check the generated PNG files.")

