from langchain_text_splitters import RecursiveCharacterTextSplitter

from system_prompts import *
from utils import remove_thinking_tokens

def condense_content(content: str, current_model):
    print(f"[INFO] condense_content: Starting condensation for {len(content)} chars")
    chunks = split_content(content)
    print(f"[INFO] Content split into {len(chunks)} chunks")
    processed_chunks = []
    map_prompt = map_reduce_custom_prompts["map_prompt"]
    reduce_prompt = map_reduce_custom_prompts["reduce_prompt"]

    #Map phase
    print("[INFO] Starting MAP phase")
    for idx, chunk in enumerate(chunks, 1):
        print(f"[DEBUG] Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
        map_input = f"""
            System:
            {yt_transcript_shortener_system_message}
            Input:
            {map_prompt.replace('{chunk_text}', chunk)}
        """

        # Stream the response and accumulate all chunks
        print(f"[DEBUG] Streaming MAP chunk {idx}...")
        chunk_response_text = ""
        for streamed_chunk in current_model.stream(map_input):
            # Extract content from AIMessageChunk
            if hasattr(streamed_chunk, 'content'):
                chunk_response_text += streamed_chunk.content
            else:
                chunk_response_text += str(streamed_chunk)
        
        print(f"[DEBUG] MAP chunk {idx} streaming complete: {len(chunk_response_text)} chars")
        cleaned_chunk_response, success = remove_thinking_tokens(chunk_response_text)
        if not success:
            error_msg = f"Failed to remove thinking tokens from MAP chunk {idx}/{len(chunks)}"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        processed_chunks.append(cleaned_chunk_response)
        print(f"[DEBUG] Cleaned chunk {idx} processed: {len(cleaned_chunk_response)} chars")

    print(f"[INFO] MAP phase complete. {len(processed_chunks)} chunks processed")
    #Reduce phase
    print("[INFO] Starting REDUCE phase")
    # Combine all processed chunks into a single string
    combined_chunks = "\n\n---\n\n".join(processed_chunks)
    print(f"[DEBUG] Combined chunks: {len(combined_chunks)} chars")
    
    reduce_input = f"""
            System:
            {yt_transcript_shortener_system_message}
            Input:
            {reduce_prompt.replace('{combined_map_results}', combined_chunks)}
        """
    
    # Stream the reduce response and accumulate all chunks
    print(f"[DEBUG] Streaming REDUCE phase...")
    reduce_response_text = ""
    for streamed_chunk in current_model.stream(reduce_input):
        # Extract content from AIMessageChunk
        if hasattr(streamed_chunk, 'content'):
            reduce_response_text += streamed_chunk.content
        else:
            reduce_response_text += str(streamed_chunk)
    
    print(f"[DEBUG] REDUCE streaming complete: {len(reduce_response_text)} chars")
    cleaned_reduce_response, success = remove_thinking_tokens(reduce_response_text)
    if not success:
        error_msg = "Failed to remove thinking tokens from REDUCE phase"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    print(f"[INFO] REDUCE phase complete: {len(cleaned_reduce_response)} chars (cleaned response)")
    print(f"[SUCCESS] Condensation complete. Original: {len(content)} -> Final: {len(cleaned_reduce_response)} chars")
    return cleaned_reduce_response

def split_content(content: str):
    chunk_size = 10000
    chunk_overlap = 200
    print(f"[DEBUG] split_content: Splitting {len(content)} chars with chunk_size={chunk_size}, overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_text(content)
    print(f"[DEBUG] split_content: Created {len(chunks)} chunks")
    return chunks