
import re

def extract_boxed(text: str) -> str:
    """
    Extract the text from the last \\boxed{...} in the text.
    Uses regex to find the position, then extracts the content handling nested braces.
    """
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return ""
    
    last_match = matches[-1]
    start_pos = last_match.end()
    
    # Count braces to find the matching closing brace
    brace_count = 1
    pos = start_pos
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count != 0:
        return ""
    
    return text[start_pos:pos-1]

def extract_last_enclosed_answer(text: str) -> str:
    """
    Extract the last text enclosed in <answer>...</answer> tags.
    Ignores incomplete or unmatched opening <answer> tags.
    """
    # Find all closing </answer> tags
    closing_tags = list(re.finditer(r'</answer>', text))
    if not closing_tags:
        return ""
    
    # find the last opening tag before the last closing tag
    last_closing = closing_tags[-1]
    closing_pos = last_closing.start()
    opening_tags = list(re.finditer(r'<answer>', text[:closing_pos]))
    if not opening_tags:
        return ""
    
    # Use the last opening tag before the closing tag
    last_opening = opening_tags[-1]
    return text[last_opening.end():last_closing.start()]
