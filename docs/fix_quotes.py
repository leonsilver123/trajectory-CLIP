# -*- coding: utf-8 -*-
"""Fix Chinese quotation marks in generate_report.py"""
import re

filepath = r'h:\trajectory CLIP\docs\generate_report.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Chinese left/right double quotes with simple angle quotes
content = content.replace('\u201c', '\u300a')  # " -> 《
content = content.replace('\u201d', '\u300b')  # " -> 》

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed Chinese quotes in generate_report.py")
