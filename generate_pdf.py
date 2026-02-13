#!/usr/bin/env python3
"""Generate a proper PDF from guitar tabs."""

from fpdf import FPDF
import sys
import re

def create_tab_pdf(input_file: str, output_file: str, title: str = "Guitar Tablature"):
    """Create a nicely formatted PDF from tab text."""
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, title, ln=True, align="C")
    pdf.ln(5)
    
    # Use monospace font for tabs
    pdf.set_font("Courier", "", 10)
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Parse and format the content
    lines = content.split('\n')
    
    # Track if we're in a tab section
    tab_lines = []
    
    for line in lines:
        # Skip markdown code blocks
        if line.strip() == '```':
            continue
            
        # Check if it's a header line (starts with #)
        if line.startswith('#'):
            if tab_lines:
                # Output accumulated tab lines
                for tl in tab_lines:
                    pdf.cell(0, 4, tl, ln=True)
                tab_lines = []
                pdf.ln(3)
            
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, line.replace('#', '').strip(), ln=True)
            pdf.set_font("Courier", "", 10)
            continue
        
        # Check if it's a tab line (starts with string name)
        if re.match(r'^[EADGBe]\|', line):
            tab_lines.append(line)
            if line.startswith('E|') and len(tab_lines) == 6:
                # Complete tab block - output it
                for tl in tab_lines:
                    pdf.cell(0, 4, tl, ln=True)
                tab_lines = []
                pdf.ln(2)
        elif line.strip():
            # Other content (chords, etc.)
            if tab_lines:
                for tl in tab_lines:
                    pdf.cell(0, 4, tl, ln=True)
                tab_lines = []
                pdf.ln(2)
            
            # Check for chord line
            if any(chord in line for chord in ['maj', 'min', 'sus', 'dim', 'aug', 'A5', 'E5', 'G5']):
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(0, 5, line.strip(), ln=True)
                pdf.set_font("Courier", "", 10)
            else:
                pdf.cell(0, 5, line, ln=True)
    
    # Output any remaining tab lines
    if tab_lines:
        for tl in tab_lines:
            pdf.cell(0, 4, tl, ln=True)
    
    pdf.output(output_file)
    print(f"✅ PDF saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_pdf.py input.txt output.pdf [title]")
        sys.exit(1)
    
    title = sys.argv[3] if len(sys.argv) > 3 else "Guitar Tablature"
    create_tab_pdf(sys.argv[1], sys.argv[2], title)
