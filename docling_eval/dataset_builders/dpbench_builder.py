import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    ImageRef,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)
from docling_core.types.io import DocumentStream
from PIL.Image import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
from tqdm import tqdm

from docling_eval.datamodels.dataset_record import DatasetRecord
from docling_eval.datamodels.types import BenchMarkColumns
from docling_eval.dataset_builders.dataset_builder import (
    BaseEvaluationDatasetBuilder,
    HFSource,
)
from docling_eval.utils.utils import (
    add_pages_to_true_doc,
    convert_html_table_into_docling_tabledata,
    crop_bounding_box,
    extract_images,
    from_pil_to_base64uri,
    get_binary,
    get_binhash,
)

# Get logger
_log = logging.getLogger(__name__)

# Labels to export in HTML visualization
TRUE_HTML_EXPORT_LABELS: Set[DocItemLabel] = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

PRED_HTML_EXPORT_LABELS: Set[DocItemLabel] = {
    DocItemLabel.TITLE,
    DocItemLabel.DOCUMENT_INDEX,
    DocItemLabel.SECTION_HEADER,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TABLE,
    DocItemLabel.PICTURE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CODE,
    DocItemLabel.REFERENCE,
    # Additional
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}


# Revision of the ``upstage/dp-bench`` HF dataset this builder is validated
# against. The coordinate convention of ``reference.json`` changed between
# revisions (see ``reference_coords_to_page_bbox``), so the revision is pinned to
# keep future upstream changes from silently corrupting the ground truth.
DPBENCH_HF_REVISION = "24702c61a2fb13325534be664653bc6e60250d13"

# Resolution at which upstage/dp-bench rendered the page images that the
# absolute pixel coordinates of ``reference.json`` refer to (revision 24702c6).
DPBENCH_REFERENCE_DPI = 200.0

# Documents whose reference image was not rendered at DPBENCH_REFERENCE_DPI.
# Maps the file name to the (width, height) in pixels of the reference image,
# derived by comparing the absolute coordinates of revision 24702c6 against the
# normalized coordinates of the previous revision (b29fd1c).
DPBENCH_REFERENCE_IMAGE_SIZES: Dict[str, Tuple[float, float]] = {
    # 1728 x 2592 pt page (24 x 36 in), reference image is 1205 x 1810 px.
    "01030000000141.pdf": (1205.0, 1810.0),
}


class PdfPageGeometry(BaseModel):
    """PDF page boxes in PDF user space (points, bottom-left origin)."""

    media_box: Tuple[float, float, float, float]  # (left, bottom, right, top)
    crop_box: Tuple[float, float, float, float]  # (left, bottom, right, top)

    @classmethod
    def from_pdf(cls, pdf_path: Path, page_index: int = 0) -> "PdfPageGeometry":
        page = PdfReader(str(pdf_path)).pages[page_index]

        def _box(rect) -> Tuple[float, float, float, float]:
            return (
                float(rect.left),
                float(rect.bottom),
                float(rect.right),
                float(rect.top),
            )

        return cls(media_box=_box(page.mediabox), crop_box=_box(page.cropbox))


def reference_coords_to_page_bbox(
    coordinates: List[Mapping[str, float]],
    page_width: float,
    page_height: float,
    geometry: Optional[PdfPageGeometry] = None,
    reference_image_size: Optional[Tuple[float, float]] = None,
) -> BoundingBox:
    """
    Convert DP-Bench ``reference.json`` polygon coordinates into a page bbox.

    Two conventions exist across revisions of ``upstage/dp-bench``:

    * up to b29fd1c: coordinates are normalized to [0, 1] relative to the page
      (the PDF CropBox, which is what docling reports as the page size).
    * since 24702c6: coordinates are absolute pixels of the page image rendered
      at ``DPBENCH_REFERENCE_DPI`` from the full PDF MediaBox (top-left
      origin). For pages whose CropBox is smaller than their MediaBox the
      CropBox offset has to be removed. A few documents were rendered at a
      different size, given by ``reference_image_size`` (width, height) in
      pixels of the MediaBox rendering.

    Returns a TOPLEFT bbox in page coordinates (points of the CropBox).
    """
    xs = [float(c["x"]) for c in coordinates]
    ys = [float(c["y"]) for c in coordinates]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    if max_x <= 1.0 and max_y <= 1.0:
        # Normalized [0, 1] page coordinates (older dataset revisions).
        return BoundingBox(
            l=min_x * page_width,
            r=max_x * page_width,
            t=min_y * page_height,
            b=max_y * page_height,
            coord_origin=CoordOrigin.TOPLEFT,
        )

    if geometry is not None:
        media_l, media_b, media_r, media_t = geometry.media_box
        crop_l, _, _, crop_t = geometry.crop_box
        media_width = media_r - media_l
        media_height = media_t - media_b
        offset_x = crop_l - media_l
        offset_y = media_t - crop_t
    else:
        media_width, media_height = page_width, page_height
        offset_x = offset_y = 0.0

    if reference_image_size is not None:
        scale_x = reference_image_size[0] / media_width
        scale_y = reference_image_size[1] / media_height
    else:
        scale_x = scale_y = DPBENCH_REFERENCE_DPI / 72.0

    return BoundingBox(
        l=min_x / scale_x - offset_x,
        r=max_x / scale_x - offset_x,
        t=min_y / scale_y - offset_y,
        b=max_y / scale_y - offset_y,
        coord_origin=CoordOrigin.TOPLEFT,
    )


class DPBenchDatasetBuilder(BaseEvaluationDatasetBuilder):
    """
    DPBench dataset builder implementing the base dataset builder interface.

    This builder processes the DPBench dataset, which contains document
    understanding benchmarks for various document types.
    """

    def __init__(
        self,
        target: Path,
        split: str = "test",
        begin_index: int = 0,
        end_index: int = -1,
        revision: str = DPBENCH_HF_REVISION,
    ):
        """
        Initialize the DPBench dataset builder.

        Args:
            target: Path where processed dataset will be saved
            split: Dataset split to use
            begin_index: Start index for processing (inclusive)
            end_index: End index for processing (exclusive), -1 means process all
            revision: Revision of the upstage/dp-bench HF dataset
        """
        super().__init__(
            name="DPBench",
            dataset_source=HFSource(repo_id="upstage/dp-bench", revision=revision),
            target=target,
            split=split,
            begin_index=begin_index,
            end_index=end_index,
        )

        self.must_retrieve = True

    def _update_gt_doc(
        self,
        doc: DoclingDocument,
        annots: Dict,
        page,
        page_image: Image,
        page_width: float,
        page_height: float,
        geometry: Optional[PdfPageGeometry] = None,
        reference_image_size: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Update ground truth document with annotations.

        Args:
            doc: DoclingDocument to update
            annots: Annotation data
            page: Page object
            page_image: Page image
            page_width: Page width
            page_height: Page height
            geometry: PDF page boxes, needed for absolute reference coordinates
            reference_image_size: Size of the reference image, if not at
                DPBENCH_REFERENCE_DPI
        """
        label = annots["category"]

        text = annots["content"]["text"].replace("\n", " ")
        html = annots["content"]["html"]

        bbox = reference_coords_to_page_bbox(
            annots["coordinates"],
            page_width=page_width,
            page_height=page_height,
            geometry=geometry,
            reference_image_size=reference_image_size,
        )

        # Create provenance
        prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=(0, len(text)))

        # Crop image element
        img = crop_bounding_box(page_image=page_image, page=page, bbox=bbox)

        # Add element to document based on label
        if label == "Header":
            doc.add_text(
                label=DocItemLabel.PAGE_HEADER, text=text, orig=text, prov=prov
            )

        elif label == "Footer":
            doc.add_text(
                label=DocItemLabel.PAGE_FOOTER, text=text, orig=text, prov=prov
            )

        elif label == "Paragraph":
            doc.add_text(label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)

        elif label == "Index":
            # FIXME: ultra approximate solution
            text = annots["content"]["text"]
            rows = text.split("\n")

            num_rows = len(rows)
            num_cols = 2

            row_span = 1
            col_span = 1

            cells = []
            for row_idx, row in enumerate(rows):
                parts = row.split(" ")

                col_idx = 0
                cell = TableCell(
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    text=" ".join(parts[:-1]),
                )
                cells.append(cell)

                col_idx = 1
                cell = TableCell(
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + row_span,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + col_span,
                    text=parts[-1],
                )
                cells.append(cell)

            table_data = TableData(
                num_rows=num_rows, num_cols=num_cols, table_cells=cells
            )
            doc.add_table(
                data=table_data,
                caption=None,
                prov=prov,
                label=DocItemLabel.DOCUMENT_INDEX,
            )

        elif label == "List":
            doc.add_list_item(text=text, orig=text, prov=prov)

        elif label == "Caption":
            doc.add_text(label=DocItemLabel.CAPTION, text=text, orig=text, prov=prov)

        elif label == "Equation":
            doc.add_text(label=DocItemLabel.FORMULA, text=text, orig=text, prov=prov)

        elif label == "Figure":
            uri = from_pil_to_base64uri(img)
            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img.width, height=img.height),
                uri=uri,
            )
            doc.add_picture(prov=prov, image=imgref)

        elif label == "Table":
            table_data = convert_html_table_into_docling_tabledata(table_html=html)
            doc.add_table(data=table_data, caption=None, prov=prov)

        elif label == "Chart":
            uri = from_pil_to_base64uri(img)
            imgref = ImageRef(
                mimetype="image/png",
                dpi=72,
                size=Size(width=img.width, height=img.height),
                uri=uri,
            )
            doc.add_picture(prov=prov, image=imgref)

        elif label == "Footnote":
            doc.add_text(label=DocItemLabel.FOOTNOTE, text=text, orig=text, prov=prov)

        elif label == "Heading1":
            doc.add_heading(text=text, orig=text, level=1, prov=prov)

    def iterate(self) -> Iterable[DatasetRecord]:
        """
        Iterate through the dataset and yield DatasetRecord objects.

        Yields:
            DatasetRecord objects
        """
        if not self.retrieved and self.must_retrieve:
            raise RuntimeError(
                "You must first retrieve the source dataset. Call retrieve_input_dataset()."
            )

        assert self.dataset_local_path is not None

        # Load the ground truth
        reference_path = self.dataset_local_path / "dataset/reference.json"
        with open(reference_path, "r") as fr:
            gt = json.load(fr)

        # Sort the filenames for deterministic ordering
        sorted_filenames = sorted(gt.keys())
        total_files = len(sorted_filenames)

        # Apply index range
        begin, end = self.get_effective_indices(total_files)
        selected_filenames = sorted_filenames[begin:end]

        # Log stats
        self.log_dataset_stats(total_files, len(selected_filenames))
        _log.info(f"Processing DP-Bench dataset with {len(selected_filenames)} files")

        for filename in tqdm(
            selected_filenames,
            desc="Processing files for DP-Bench",
            ncols=128,
        ):
            # Get annotations for this file
            annots = gt[filename]
            pdf_path = self.dataset_local_path / f"dataset/pdfs/{filename}"

            # Create the ground truth Document
            true_doc = DoclingDocument(
                name=f"ground-truth {os.path.basename(pdf_path)}"
            )
            true_doc, true_page_images = add_pages_to_true_doc(
                pdf_path=pdf_path, true_doc=true_doc, image_scale=2.0
            )

            assert len(true_page_images) == 1, "len(true_page_images)==1"

            # Get page dimensions
            page_width = true_doc.pages[1].size.width
            page_height = true_doc.pages[1].size.height
            geometry = PdfPageGeometry.from_pdf(pdf_path)
            reference_image_size = DPBENCH_REFERENCE_IMAGE_SIZES.get(filename)

            # Process each element in the annotation
            for elem in annots["elements"]:
                self._update_gt_doc(
                    true_doc,
                    elem,
                    page=true_doc.pages[1],
                    page_image=true_page_images[0],
                    page_width=page_width,
                    page_height=page_height,
                    geometry=geometry,
                    reference_image_size=reference_image_size,
                )

            # Extract images from the ground truth document
            true_doc, true_pictures, true_page_images = extract_images(
                document=true_doc,
                pictures_column=BenchMarkColumns.GROUNDTRUTH_PICTURES.value,
                page_images_column=BenchMarkColumns.GROUNDTRUTH_PAGE_IMAGES.value,
            )

            # Get PDF as binary data
            pdf_bytes = get_binary(pdf_path)
            pdf_stream = DocumentStream(name=pdf_path.name, stream=BytesIO(pdf_bytes))

            # Create dataset record
            record = DatasetRecord(
                doc_id=str(filename),
                doc_hash=get_binhash(pdf_bytes),
                ground_truth_doc=true_doc,
                ground_truth_pictures=true_pictures,
                ground_truth_page_images=true_page_images,
                original=pdf_stream,
                mime_type="application/pdf",
            )

            yield record
