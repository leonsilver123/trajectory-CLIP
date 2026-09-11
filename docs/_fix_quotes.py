# -*- coding: utf-8 -*-
"""Fix unbalanced double quotes in generate_report.py"""
import re

path = r'h:\trajectory CLIP\docs\generate_report.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace problematic patterns: inner " used as Chinese quotes
# Pattern: "text" inside a double-quoted string -> use \u201c \u201d
replacements = [
    # Line 148 area
    ('\u5b8c\u6210\u201c\u67e5\u627e\u2192\u786e\u8ba4\u2192\u56de\u6eaf\u2192\u53d6\u8bc1\u2192\u62a5\u544a\u2192\u7559\u75d5\u201d',
     '\u5b8c\u6210\u300c\u67e5\u627e\u2192\u786e\u8ba4\u2192\u56de\u6eaf\u2192\u53d6\u8bc1\u2192\u62a5\u544a\u2192\u7559\u75d5\u300d'),
    # General pattern: replace "X" inside strings with 「X」
    # Find lines with unbalanced quotes and fix them
]

# Simple approach: replace all remaining " that look like Chinese quotes
# These are ASCII " used inside strings as Chinese-style quotes
# Strategy: for any line, if it has more than 2 " and they're unbalanced,
# convert inner ones to 「」

lines = content.split('\n')
fixed_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('#'):
        fixed_lines.append(line)
        continue
    
    # Count quotes
    quote_count = stripped.count('"')
    if quote_count <= 2:
        fixed_lines.append(line)
        continue
    
    # Check if quotes are balanced (even number)
    if quote_count % 2 == 0:
        # Could be multiple strings on same line (like in list/tuple)
        # Check if it's a simple function call with one string arg containing inner quotes
        # Pattern: func("...text"inner"text...")
        # We need to find the string boundaries
        
        # Try to parse: find the first and last " which are the real string delimiters
        first_q = line.index('"')
        last_q = line.rindex('"')
        
        if first_q != last_q:
            # Everything between first and last quote that is also a quote is an inner quote
            prefix = line[:first_q+1]
            suffix = line[last_q:]
            middle = line[first_q+1:last_q]
            
            # Replace inner " with Unicode LEFT/RIGHT DOUBLE QUOTATION MARK
            # Simple approach: alternate between left and right
            new_middle = []
            is_left = True
            for c in middle:
                if c == '"':
                    new_middle.append('\u300c' if is_left else '\u300d')
                    is_left = not is_left
                else:
                    new_middle.append(c)
            
            fixed_lines.append(prefix + ''.join(new_middle) + suffix)
        else:
            fixed_lines.append(line)
    else:
        # Odd number of quotes - more complex issue
        # Just replace inner quotes
        first_q = line.index('"')
        last_q = line.rindex('"')
        if first_q != last_q:
            prefix = line[:first_q+1]
            suffix = line[last_q:]
            middle = line[first_q+1:last_q]
            new_middle = []
            is_left = True
            for c in middle:
                if c == '"':
                    new_middle.append('\u300c' if is_left else '\u300d')
                    is_left = not is_left
                else:
                    new_middle.append(c)
            fixed_lines.append(prefix + ''.join(new_middle) + suffix)
        else:
            fixed_lines.append(line)

content = '\n'.join(fixed_lines)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed quotes in generate_report.py")
