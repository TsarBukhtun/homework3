import json
import re
import argparse
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# ----------------------------
# Helper: Load system prompt
# ----------------------------
def load_system_prompt(prompt_file: str = "system_sft_prompt.txt") -> str:
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"System prompt file '{prompt_file}' not found.")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()

# ----------------------------
# Helper: Split Markdown by headings
# ----------------------------
def split_markdown_by_headings(markdown_text: str) -> List[str]:
    # Split by level 1 or 2 headings (e.g., # or ##)
    sections = re.split(r'\n(?=#+\s)', markdown_text)
    cleaned = []
    for sec in sections:
        sec = sec.strip()
        if sec and not sec.startswith("#"):
            # Reattach heading if lost
            pass
        if sec:
            cleaned.append(sec)
    return cleaned

# ----------------------------
# Main SFT generation function
# ----------------------------
def create_sft_dataset_with_langchain(input_path: str, output_path: str, system_prompt: str, target_count: int = 300):
    # Initialize Qwen via DashScope (requires QWEN_API_KEY env var)
    llm = ChatOpenAI(
        model="qwen-plus",  # or "qwen-turbo", "qwen-max"
        temperature=0.1,
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        max_tokens=5000,
    )

    # Define prompt template
    sft_prompt = PromptTemplate.from_template("""
{system_prompt}

### Input Markdown:
{text}

### Output JSON SFT Samples:
""")

    # Use JsonOutputParser — but wrap in retry/error handling
    parser = JsonOutputParser()

    # Build chain
    chain = (
        {"system_prompt": lambda x: system_prompt, "text": RunnablePassthrough()}
        | sft_prompt
        | llm
        | parser
    )

    # Read input Markdown
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Split into logical sections (by headings)
    sections = split_markdown_by_headings(full_text)

    all_samples = []
    attempts = 0
    max_attempts = 50 

    print(f"Generating {target_count}+ SFT samples from {len(sections)} sections...")

    while len(all_samples) < target_count and attempts < max_attempts:
        # Rotate through sections to avoid repetition
        section = sections[attempts % len(sections)]
        
        try:
            print(f"Attempt {attempts + 1}: Generating from section snippet...")
            result = chain.invoke(section)  # Truncate very long sections

            if isinstance(result, list):
                # Filter valid samples
                valid_samples = [
                    s for s in result
                    if isinstance(s, dict) and "question" in s and "output" in s
                ]
                all_samples.extend(valid_samples)
                print(f"  → Got {len(valid_samples)} valid samples (total: {len(all_samples)})")
                with open(output_path + '_ATTEMPT_' + str(attempts) + ".json", "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                print(f"  → Unexpected output type: {type(result)}")

        except Exception as e:
            print(f"  ❌ Error on attempt {attempts + 1}: {e}")

        attempts += 1
        time.sleep(0.5)  # Rate limiting (DashScope allows ~10–20 RPM for qwen-plus)

        if len(all_samples) >= target_count:
            break

    # Trim to exact target if needed (or keep extras)
    final_samples = all_samples[:target_count] if target_count else all_samples

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Successfully saved {len(final_samples)} SFT samples to {output_path}")

# ----------------------------
# CLI Entry Point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert markdown file to history-focused SFT training data using LangChain + Qwen"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="parsed_pdf_output/BOOK_KZ_HISTORY_CLEANED.pdf.md",
        help="Input markdown file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="sft_output/langchain_ready_kz_history_sft_data.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="system_sft_prompt.txt",
        help="System prompt SFT file path"
    )
    
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.prompt)
    create_sft_dataset_with_langchain(args.input, args.output, system_prompt)

if __name__ == "__main__":
    load_dotenv()
    # Ensure DashScope API key is set
    if not os.getenv("QWEN_API_KEY"):
        raise EnvironmentError("Please set QWEN_API_KEY environment variable for Qwen access.")
    main()