import os
import sys
import json
import logging
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    PictureDescriptionApiOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import ConversionStatus

# Configure logging for docling
logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s: %(message)s")
_log = logging.getLogger(__name__)


def convert_pdf_to_markdown(input_pdf_path, output_dir):
    """
    PDF to Markdown converter using the Docling Python library.
    """
    print("🚀 Starting PDF to Markdown conversion using Docling library...")
    print(f"Input: {input_pdf_path}")
    print(f"Output: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Configure pipeline options for GPU, enrichment, and image handling
        accelerator_opts = AcceleratorOptions(device=AcceleratorDevice.CUDA)

        # Configure to use a local Ollama model for picture description
        ollama_options = PictureDescriptionApiOptions(
            url="https://ai.bachatgroup.com/api/generate",  # Default Ollama API endpoint
            params={"model": "llava:latest"},  # Use the LLaVA model for picture description
            prompt="Describe this image in detail.",
        )
  
        pipeline_opts = PdfPipelineOptions(
            accelerator_options=accelerator_opts,
            do_formula_enrichment=True,         # Equivalent to --enrich-formula
            do_picture_description=True,        # Equivalent to --enrich-picture-description
            picture_description_options=ollama_options,
            generate_page_images=True,          # Required for image processing (and picture description)
            generate_picture_images=True,       # Required for picture description
            images_scale=2  ,
            enable_remote_services=True       # Allow remote API calls for enrichment
        )

        # 2. Create a format option for PDF conversion
        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_opts)

        # 3. Initialize the DocumentConverter
        converter = DocumentConverter(
            format_options={
                "pdf": pdf_format_option,
                "image": pdf_format_option # Also apply to images within PDFs
            }
        )

        # 4. Run the conversion
        print("🔥 Running conversion... This may take a while for large documents.")
        conv_result = converter.convert(source=input_pdf_path)

        # 5. Check the result and save the output
        if conv_result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
            print("✅ Conversion process completed successfully!")

            input_file_stem = Path(input_pdf_path).stem
            output_markdown_file = Path(output_dir) / f"{input_file_stem}.md"

            # Save the document as Markdown with embedded images
            conv_result.document.save_as_markdown(
                filename=output_markdown_file,
                image_mode=ImageRefMode.EMBEDDED  # Equivalent to --image-export-mode embedded
            )

            print(f"📁 Output file saved to: {output_markdown_file}")
            return str(output_markdown_file)

        else:
            print(f"❌ Conversion failed with status: {conv_result.status}")
            for error in conv_result.errors:
                print(f"  - Error: {error.error_message}")
            return None

    except Exception as e:
        print(f"❌ An unexpected error occurred during conversion for {input_pdf_path}: {e}")
        _log.exception("Conversion failed")
        return None

if __name__ == "__main__":
    print("📚 PDF to Multimodal Converter Starting...")
    print("=" * 60)

    # Set environment variables to fix Windows symlink issues
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    print("🔧 Applied Windows compatibility fixes for model downloads...")

    # Load configuration from config.json (at project root)
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    config_path = _PROJECT_ROOT / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Define the input folder containing PDF files
    input_folder = Path(config["input_dir"])
    # Define the output directory for converted markdown files (same as PDF names with .md extension)
    output_directory = Path(config["output_base_dir"])

    # Ensure output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    pdf_files_found = []
    for file_path in input_folder.glob("*.pdf"):
        pdf_files_found.append(file_path)

    if not pdf_files_found:
        print(f"⚠️ No PDF files found in the input folder: {input_folder}")
    else:
        print(f"🔍 Found {len(pdf_files_found)} PDF files to convert.")
        for pdf_file in pdf_files_found:
            print(f"\n--- Processing {pdf_file.name} ---")
            converted_file_path = convert_pdf_to_markdown(pdf_file, str(output_directory))
            if converted_file_path:
                print(f"✨ Successfully converted {pdf_file.name} to {Path(converted_file_path).name}")
            else:
                print(f"💔 Failed to convert {pdf_file.name}")
                break

    print("\n=" * 60)
    print("🏁 PDF to Multimodal Converter Finished.")
