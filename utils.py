import re


def remove_thinking_tokens(text: str) -> tuple[str, bool]:
    """
    Extract final script from LLM response by finding content within <final_script> tags.
    
    The LLM returns the final cleaned output wrapped in <final_script></final_script> tags.
    This function extracts only the content between the LAST opening and LAST closing tag,
    discarding all reasoning, thinking tokens, and other artifacts.

    Args:
        text: Raw LLM response text containing <final_script> tags

    Returns:
        Tuple of (cleaned_text, success_bool) where success_bool indicates if tags were found
    """
    if not text:
        print("[DEBUG] remove_thinking_tokens: Empty text received")
        return text, False

    print(f"[DEBUG] remove_thinking_tokens: Processing {len(text)} characters")
    original_length = len(text)

    # Find the last occurrence of opening and closing tags
    opening_tag = '<final_script>'
    closing_tag = '</final_script>'
    
    last_open_index = text.lower().rfind(opening_tag.lower())
    last_close_index = text.lower().rfind(closing_tag.lower())
    
    if last_open_index != -1 and last_close_index != -1 and last_close_index > last_open_index:
        # Extract content between last opening tag and last closing tag
        start_pos = last_open_index + len(opening_tag)
        final_content = text[start_pos:last_close_index].strip()
        
        print(f"[INFO] Found <final_script> tags, extracting content from last occurrence")
        print(f"[CLEANUP] Extracted {len(final_content)} chars from final_script tag (removed {original_length - len(final_content)} chars)")
        return final_content, True
    else:
        print("[WARNING] No valid <final_script> tags found, thinking tokens not properly removed")
        print(f"[DEBUG] Original text content:\n{text}")
        return text.strip(), False
