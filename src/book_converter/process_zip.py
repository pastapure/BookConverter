import os
import sys
import json
import logging
import zipfile
import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling_core.types.doc import ImageRefMode
from docling.datamodel.base_models import ConversionStatus

# Configure logging for docling
logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s: %(message)s")
_log = logging.getLogger(__name__)


def convert_pdf_to_markdown(converter, input_pdf_path, output_dir):
    """
    PDF to Markdown converter using the Docling Python library.
    """
    print(f"🚀 Starting PDF to Markdown conversion for: {Path(input_pdf_path).name}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if output markdown file already exists
    input_file_stem = Path(input_pdf_path).stem
    output_markdown_file = Path(output_dir) / f"{input_file_stem}.md"
    
    if output_markdown_file.exists():
        print(f"⏭️  Skipping {Path(input_pdf_path).name} - output file already exists: {output_markdown_file.name}")
        return str(output_markdown_file)

    try:
        # The DocumentConverter is now passed in, so we can skip initialization.
        # 1. Run the conversion
        print("🔥 Running conversion... This may take a while.")
        conv_result = converter.convert(source=str(input_pdf_path))

        # 2. Check the result and save the output
        if conv_result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS]:
            conv_result.document.save_as_markdown(
                filename=output_markdown_file,
                image_mode=ImageRefMode.EMBEDDED
            )
            print(f"✨ Successfully converted {Path(input_pdf_path).name} to {output_markdown_file.name}")
            return str(output_markdown_file)
        else:
            print(f"❌ Conversion failed for {Path(input_pdf_path).name} with status: {conv_result.status}")
            for error in conv_result.errors:
                print(f"  - Error: {error.error_message}")
            return None

    except Exception as e:
        print(f"❌ An unexpected error occurred during conversion for {Path(input_pdf_path).name}: {e}")
        _log.exception("Conversion failed")
        return None

if __name__ == "__main__":
    print("📚 ZIP Archive to Multimodal Converter Starting...")
    print("=" * 60)

    # Set environment variables to fix Windows symlink issues with Hugging Face models
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print("🔧 Applied Windows compatibility fixes for model downloads.")

    # Load configuration from config.json (at project root)
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    config_path = _PROJECT_ROOT / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    input_folder = Path(config["input_dir"])
    output_directory = Path(config["output_base_dir"])

    # --- Validate directories exist ---
    print(f"📁 Checking input directory: {input_folder}")
    if not input_folder.exists():
        print(f"❌ Input directory does not exist: {input_folder}")
        print("Please update config.json with a valid input directory path.")
        sys.exit(1)
    
    print(f"📁 Checking output directory: {output_directory}")
    # Try to create output directory to verify it's writable
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = output_directory / "write_test.tmp"
        test_file.touch()
        test_file.unlink()
        print(f"✅ Output directory is accessible: {output_directory}")
    except PermissionError:
        print(f"❌ Permission denied: Cannot write to output directory: {output_directory}")
        print("Please update config.json with a writable output directory path.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot access output directory: {output_directory}. Error: {e}")
        print("Please update config.json with a valid output directory path.")
        sys.exit(1)

    # --- Initialize DocumentConverter once ---
    print("🛠️ Initializing DocumentConverter... This might take a moment.")
    # 1. Configure pipeline options for GPU, enrichment, and image handling
    accelerator_opts = AcceleratorOptions(device=AcceleratorDevice.CUDA)

    pipeline_opts = PdfPipelineOptions(
        accelerator_options=accelerator_opts,
        do_formula_enrichment=True,
        do_picture_description=False,
        generate_page_images=True,
        generate_picture_images=True,
        images_scale=2,
    )

    # 2. Create a format option for PDF conversion
    pdf_format_option = PdfFormatOption(pipeline_options=pipeline_opts)

    # 3. Initialize the DocumentConverter
    converter = DocumentConverter(format_options={"pdf": pdf_format_option, "image": pdf_format_option})
    print("✅ DocumentConverter initialized.")

    # Ensure output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)

    # Find all ZIP files in the input directory
    zip_files_found = list(input_folder.glob("*.zip"))

    if not zip_files_found:
        print(f"⚠️ No ZIP files found in the input folder: {input_folder}")
    else:
        print(f"🔍 Found {len(zip_files_found)} ZIP files to process.")
        for zip_file_path in zip_files_found:
            print(f"\n--- Processing ZIP file: {zip_file_path.name} ---")
            # Use a temporary directory to extract files
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    print(f"📦 Extracting {zip_file_path.name}...")
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find all PDF files in the extracted content (including subdirectories)
                    extracted_pdfs = list(Path(temp_dir).rglob("*.pdf"))
                    if not extracted_pdfs:
                        print(f"🤷 No PDF files found inside {zip_file_path.name}.")
                        continue

                    print(f"Found {len(extracted_pdfs)} PDF(s) in the archive. Starting conversion...")
                    for pdf_file in extracted_pdfs:
                        convert_pdf_to_markdown(converter, pdf_file, str(output_directory))

                except zipfile.BadZipFile:
                    print(f"💔 Error: {zip_file_path.name} is not a valid ZIP file or is corrupted.")
                except Exception as e:
                    print(f"❌ An unexpected error occurred while processing {zip_file_path.name}: {e}")

    print("\n=" * 60)
    print("🏁 ZIP Archive to Multimodal Converter Finished.")
