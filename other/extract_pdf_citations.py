#!/usr/bin/env python3
"""
Extract author names and years from PDF files in the article folder.
"""
import os
import re
import json
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Installing pypdf...")
    import subprocess
    subprocess.check_call(["pip", "install", "pypdf", "-q"])
    from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
ARTICLE_DIR = ROOT / "article"

# Map reference numbers to PDF filename patterns
CITATION_MAP = {
    1: ["Conmedia", "PotatoLeafPest"],
    2: ["agriculture-15-01526", "agriculture-15"],
    3: ["ennadifi2020", "ennadifi"],
    4: ["jimaging-09-00140", "jimaging"],
    5: ["Plant Pathology", "Long", "wheat diseases deep learning"],
    6: ["s11042-022-12160-3", "s11042"],
    7: ["s41598-024-83636-5", "s41598"],
    8: ["s44447-025-00007-w", "s44447"],
    9: ["TSP_CMC_61995", "TSP_CMC"],
    10: ["Wheat_disease_recognition", "Multi-Model_Analysis", "LIME", "Grad-CAM"],
    11: ["Wheat_Diseases_Recognition", "Optimal_Features", "Soft_Attention"],
    12: ["Data-Efficient_Wheat", "Shifted_Window_Transformer"],
    13: ["data-10-00025", "data-10"],
    14: ["Disease_Detection_and_Identification_of_Rice", "Rice_Leaf"]
}

def extract_year_from_filename(filename):
    """Extract year from filename."""
    year_match = re.search(r'(20\d{2}|19\d{2})', filename)
    return year_match.group() if year_match else None

def extract_author_from_metadata(pdf_path):
    """Extract author from PDF metadata."""
    try:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        author = meta.get('/Author', meta.get('Author', ''))
        if author:
            # Clean up author string
            author = author.strip()
            # If multiple authors, take first one and add "et al."
            if ';' in author or ',' in author:
                first_author = author.split(';')[0].split(',')[0].strip()
                return f"{first_author} et al."
            return author
    except Exception as e:
        pass
    return None

def find_pdf_for_citation(num):
    """Find PDF file matching citation number."""
    patterns = CITATION_MAP.get(num, [])
    if not patterns:
        return None
    
    for pdf_file in ARTICLE_DIR.glob("*.pdf"):
        filename_lower = pdf_file.name.lower()
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return pdf_file
    return None

def extract_citation_info():
    """Extract citation info for all references."""
    citations = {}
    
    for num in sorted(CITATION_MAP.keys()):
        pdf_path = find_pdf_for_citation(num)
        if not pdf_path:
            print(f"[{num}] No PDF found")
            continue
        
        author = extract_author_from_metadata(pdf_path)
        year = extract_year_from_filename(pdf_path.name)
        
        if author:
            if year:
                citations[num] = f"{author}, {year}"
            else:
                citations[num] = author
            print(f"[{num}] {pdf_path.name}")
            print(f"     -> {citations[num]}")
        else:
            # Try to extract from first page text
            try:
                reader = PdfReader(str(pdf_path))
                if len(reader.pages) > 0:
                    first_page_text = reader.pages[0].extract_text()
                    # Look for common author patterns
                    import re
                    # Pattern: "Author1, Author2, Author3" or "Author1 et al."
                    author_patterns = [
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+et\s+al\.',
                        r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*',
                    ]
                    for pattern in author_patterns:
                        match = re.search(pattern, first_page_text[:2000])
                        if match:
                            author = match.group(1)
                            year = extract_year_from_filename(pdf_path.name)
                            if year:
                                citations[num] = f"{author} et al., {year}"
                            else:
                                citations[num] = f"{author} et al."
                            print(f"[{num}] {pdf_path.name}")
                            print(f"     -> {citations[num]} (extracted from text)")
                            break
            except:
                pass
            
            if num not in citations:
                # Fallback: use filename-based guess
                year = extract_year_from_filename(pdf_path.name)
                if year:
                    citations[num] = f"Unknown et al., {year}"
                else:
                    citations[num] = "Unknown et al."
                print(f"[{num}] {pdf_path.name} (metadata not found)")
                print(f"     -> {citations[num]}")
    
    return citations

if __name__ == "__main__":
    print("Extracting citations from PDFs...\n")
    citations = extract_citation_info()
    
    # Save to JSON for reference
    output_file = ROOT / "docs" / "citation_map.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(citations, f, indent=2, ensure_ascii=False)
    
    print(f"\nCitations saved to: {output_file}")

