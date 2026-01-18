import gc
import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    OcrMacOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def parse_pdf_with_page_range(pdf_path: Path, start_page: int, end_page: int, output_dir: Path):
    """
    Parses a PDF document and extracts content from a specified page range,
    saving each page's content to a separate Markdown file.

    Args:
        pdf_path: The path to the PDF file.
        start_page: The starting page number (inclusive, 1-based).
        end_page: The ending page number (inclusive, 1-based).
        output_dir: The directory to save the output files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # Docling Parse with ocrmac (macOS only)
    # --------------------------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    pipeline_options.ocr_options = OcrMacOptions()

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    try:
        start_time = time.time()
        document_result = doc_converter.convert(pdf_path, page_range=(start_page, end_page))
        end_time = time.time() - start_time

        _log.info(f"Document converted in {end_time:.2f} seconds.")
        print(f"Output will be saved to: {output_dir}")

        if not document_result:
            print("Error: Docling did not return any content.")
            return

        print(f"Processing extracted content for pages {start_page} to {end_page}...")

        # Export Markdown format:
        with (output_dir / f"{pdf_path}.md").open("w", encoding="utf-8") as fp:
            fp.write(document_result.document.export_to_markdown())
    finally:
        # Explicitly clean up resources to prevent memory leaks
        # Delete the converter and force garbage collection
        del doc_converter
        gc.collect()


if __name__ == "__main__":
    # IMPORTANT: Replace 'your_document.pdf' with the actual path to your PDF file.
    # Ensure this PDF file exists in the same directory or provide an absolute path.
    pdf_file_path = Path("BOOK_KZ_HISTORY.pdf")
    
    start_page_num = 3
    end_page_num = 23

    # Define an output directory
    output_directory = Path("parsed_pdf_output")
    
    parse_pdf_with_page_range(pdf_file_path, start_page_num, end_page_num, output_directory)

    print("\nScript execution complete.")
    print("Please ensure 'your_document.pdf' is in the correct location and adjust the script if the 'docling' result structure differs from the assumed one.")