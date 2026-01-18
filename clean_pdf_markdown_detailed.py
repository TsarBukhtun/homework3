import re

def clean_markdown_file(input_file, output_file):
    """
    Clean up a markdown file by removing trash information.
    
    Args:
        input_file (str): Path to the input markdown file
        output_file (str): Path to the output cleaned markdown file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Store original length for comparison
    original_length = len(content)
    
    # Remove image placeholders
    content = re.sub(r'<!-- image -->', '', content)
    
    # Remove standalone page numbers (numbers on their own lines)
    content = re.sub(r'^\d+\s*$', '', content, flags=re.MULTILINE)
    
    # Remove page numbers that appear at the beginning or end of lines
    content = re.sub(r'(^\s*\d+\s*)|(\s*\d+\s*$)', '', content, flags=re.MULTILINE)
    
    # Remove extra whitespace and empty lines created by cleaning
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Replace multiple empty lines with single
    
    # Remove trailing whitespaces
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Remove leading/trailing whitespace from the entire document
    content = content.strip()
    
    # Fix multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove bullet points that look like trash (e.g., ". " at the beginning of lines)
    content = re.sub(r'^\.\s+', '', content, flags=re.MULTILINE)
    
    # Remove isolated dots that might be remnants of formatting
    content = re.sub(r'^\.\s*$', '', content, flags=re.MULTILINE)
    
    # Clean up any remaining excessive whitespace
    content = re.sub(r'[ \t]+\n', '\n', content)  # Remove spaces/tabs before newlines
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Original length: {original_length}")
    print(f"Cleaned length: {len(content)}")
    print(f"Characters removed: {original_length - len(content)}")
    print(f"Cleaned markdown saved to {output_file}")

def compare_files(original_file, cleaned_file):
    """
    Compare the original and cleaned files to show differences.
    """
    with open(original_file, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        cleaned_lines = f.readlines()
    
    print(f"\nOriginal file lines: {len(original_lines)}")
    print(f"Cleaned file lines: {len(cleaned_lines)}")
    print(f"Lines removed: {len(original_lines) - len(cleaned_lines)}")

if __name__ == "__main__":
    input_path = "parsed_pdf_output/BOOK_KZ_HISTORY.pdf.md"
    output_path = "parsed_pdf_output/BOOK_KZ_HISTORY_CLEANED.pdf.md"
    
    clean_markdown_file(input_path, output_path)
    compare_files(input_path, output_path)