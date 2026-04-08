"""Convert all raw/*.pdf files to raw/*.md using docling."""

import glob
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

def main():
    pdfs = sorted(Path("raw").glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in raw/")
        sys.exit(1)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })

    for pdf in pdfs:
        out_path = pdf.with_suffix(".md")
        if out_path.exists():
            print(f"Skip {pdf.name} (markdown already exists)")
            continue
        print(f"Converting {pdf.name} ...")
        result = converter.convert(str(pdf))
        out_path.write_text(result.document.export_to_markdown())
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
