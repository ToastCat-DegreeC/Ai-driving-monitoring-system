import re

file_path = 'THESIS_DRAFT.md'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Separate the references section and appendix
ref_marker = '\n## REFERENCES\n'
appendix_marker = '\n---\n\n\n## [APPENDIX'

parts = content.split(ref_marker)
if len(parts) < 2:
    print("Could not find REFERENCES section")
    exit(1)

body = parts[0]
ref_and_appendix = parts[1]

appendix_parts = ref_and_appendix.split(appendix_marker)
ref_section = appendix_parts[0]
appendix = appendix_marker + appendix_parts[1] if len(appendix_parts) > 1 else ""

# 1. Discover the order of appearance in the body
# We look for [n] where n is a number
all_citations = re.findall(r'\[(\d+)\]', body)
seen = set()
order = []
for c in all_citations:
    if c not in seen:
        order.append(c)
        seen.add(c)

# Create mapping: old_index -> new_index
mapping = {old: str(i + 1) for i, old in enumerate(order)}

# 2. Update the body text
# Use a placeholder to avoid double replacement (e.g., replacing [1] with [2] then [2] with [3])
def replace_with_placeholder(match):
    num = match.group(1)
    if num in mapping:
        return f'@@@{mapping[num]}@@@'
    return match.group(0)

body_with_placeholders = re.sub(r'\[(\d+)\]', replace_with_placeholder, body)
final_body = re.sub(r'@@@(\d+)@@@', r'[\1]', body_with_placeholders)

# 3. Parse and rebuild the References section
# Each ref is like [n] Text...\n
refs_found = re.findall(r'\[(\d+)\] (.*?)(?=\n\[\d+\] |\n---|\Z)', ref_section, re.DOTALL)
ref_dict = {m[0]: m[1].strip() for m in refs_found}

new_ref_list = []
for old_num in order:
    if old_num in ref_dict:
        new_num = mapping[old_num]
        new_ref_list.append(f'[{new_num}] {ref_dict[old_num]}')
    else:
        print(f"Warning: Reference [{old_num}] found in text but missing in bibliography")

# Handle references that might be in the bibliography but NOT in the text
# (Though per the user's request, we likely only care about used ones)
unused_refs = set(ref_dict.keys()) - set(order)
if unused_refs:
    print(f"Note: The following references were in the bibliography but not in the text: {unused_refs}")
    # Optional: Append them at the end? Or just leave them out? 
    # Usually, IEEE only lists cited works. I'll leave them out for now.

final_ref_section = '\n'.join(new_ref_list)

# 4. Final assembly
final_content = final_body + ref_marker + final_ref_section + '\n' + appendix

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(final_content)

print(f"Successfully reordered {len(order)} references.")
