import json
import os
import fitz

def chunk_text(text, max_chunk_size):
    chunks = []
    current_chunk = ""

    # Split text by newlines to preserve paragraph structure
    paragraphs = text.split('\n\n')

    for paragraph in paragraphs:
        # Check if this paragraph contains JSON-like content
        json_start = paragraph.find('{')
        json_end = paragraph.rfind('}')
        contains_json = json_start != -1 and json_end != -1

        # Calculate size with the new paragraph
        new_chunk_size = len((current_chunk + "\n\n" + paragraph if current_chunk else paragraph).encode('utf-8'))

        # If adding this paragraph would exceed max size and it's not JSON
        if new_chunk_size > max_chunk_size and not contains_json:
            # If we have content, add the current chunk to our list
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # If a single paragraph is too big, we still need to add it
                chunks.append(paragraph.strip())
                current_chunk = ""
        else:
            # If it contains JSON or is small enough, add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

            # If we just added JSON content and have reached a good size,
            # finalize this chunk to avoid splitting related content
            if contains_json and new_chunk_size > max_chunk_size * 0.8:
                chunks.append(current_chunk.strip())
                current_chunk = ""

    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def extract_text_from_pdf(input_file_path, output_file_path, output_jsonl_file_path, title):
    try:
        doc = fitz.open(input_file_path)
        text = ""

        if len(doc) > 0:
            first_page = doc[0]
            page_rect = first_page.rect

            # Define header/footer regions to exclude (adjust these values as needed)
            header_height = 72  # ~1 inch from top
            footer_height = 72  # ~1 inch from bottom

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            content_rect = fitz.Rect(
                page_rect.x0,
                page_rect.y0 + header_height,
                page_rect.x1,
                page_rect.y1 - footer_height
            )
            text += page.get_text("text", clip=content_rect) + "\n\n"


        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text)

        titleFragments = title.split("_")
        id = titleFragments[0]
        apiName = " ".join(titleFragments[1:])
        #
        #
        # json_object = {"id": id, "apiName": apiName, "title": title, "documentType": "Tmforum api user guide", "text": text}
        #
        # # Write to the JSONL file (each line is a valid JSON object)
        # with open(output_jsonl_file_path, 'a', encoding='utf-8') as jsonl_file:
        #     jsonl_file.write(json.dumps(json_object) + '\n')
        #
        # print(f"Info: Text extracted from {input_file_path} and saved to {output_file_path}")
        # Break text into chunks
        chunks = chunk_text(text, max_chunk_size=7000)  # 7KB in bytes

        # Create JSON object for each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{id}-chunk-{i+1}"
            json_object = {
                "tmforumApiId": id,
                "id": chunk_id,
                "chunkIndex": i + 1,
                "totalChunks": len(chunks),
                "apiName": apiName,
                "title": f"{title} (Part {i+1}/{len(chunks)})",
                "documentType": "Tmforum api user guide",
                "text": chunk
            }

            # Write to the JSONL file (each line is a valid JSON object)
            with open(output_jsonl_file_path, 'a', encoding='utf-8') as jsonl_file:
                jsonl_file.write(json.dumps(json_object) + '\n')

        print(f"Info: Text extracted from {input_file_path}, saved to {output_file_path}, and chunked into {len(chunks)} parts")

    except Exception as e:
        print(f"Error: extracting text from {input_file_path}: {e}")
        print(f"Error: details: {type(e).__name__}")
        return None

def process_directory(input_dir):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "texts")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_jsonl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "jsonls")
    if os.path.exists(output_jsonl_dir):
        import shutil
        shutil.rmtree(output_jsonl_dir)
    os.makedirs(output_jsonl_dir, exist_ok=True)

    for root, dir, files in os.walk(input_dir):
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]

        for pdf_file in pdf_files:
            input_file_path = os.path.join(root, pdf_file)
            filename = os.path.splitext(pdf_file)[0]
            base_filename = filename + ".txt"
            base_jsonl_filename = filename + ".jsonl"
            output_file_path = os.path.join(output_dir, base_filename)
            output_jsonl_file_path = os.path.join(output_jsonl_dir, base_jsonl_filename)
            print(f"Info: Processing: input_dir: {input_file_path}..")
            extract_text_from_pdf(input_file_path, output_file_path, output_jsonl_file_path, filename)


if __name__ == "__main__":
    input_dir = "/Users/AkashKumar/D0cuments/docs/tmforum"  # Replace with your PDF file path
    process_directory(input_dir)