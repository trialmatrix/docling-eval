from pathlib import Path

import pytest
from docling_core.types.doc import CoordOrigin
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import RectangleObject
from reportlab.pdfgen import canvas

from docling_eval.dataset_builders.dataset_builder import HFSource
from docling_eval.dataset_builders.dpbench_builder import (
    DPBENCH_HF_REVISION,
    DPBENCH_REFERENCE_IMAGE_SIZES,
    DPBenchDatasetBuilder,
    PdfPageGeometry,
    reference_coords_to_page_bbox,
)


def _rect(l: float, t: float, r: float, b: float):
    return [
        {"x": l, "y": t},
        {"x": r, "y": t},
        {"x": r, "y": b},
        {"x": l, "y": b},
    ]


# Excerpt of upstage/dp-bench@24702c6 dataset/reference.json for
# 01030000000001.pdf (page size 439.37 x 666.142 pt, MediaBox == CropBox).
# The same elements in revision b29fd1c were normalized to [0, 1].
PAGE_001 = (439.37, 666.142)
GEOMETRY_001 = PdfPageGeometry(
    media_box=(0.0, 0.0, 439.37, 666.142), crop_box=(0.0, 0.0, 439.37, 666.142)
)
YARROW_HEADER = _rect(
    983.318030733286, 102.3493458064781, 1079.872431445213, 120.6598699131856
)
PARAGRAPH = _rect(
    170.18537282269165, 169.39723781817833, 1080.6978511750278, 503.0565269474403
)


def test_absolute_pixel_coords_are_200dpi_page_pixels():
    bbox = reference_coords_to_page_bbox(
        PARAGRAPH,
        page_width=PAGE_001[0],
        page_height=PAGE_001[1],
        geometry=GEOMETRY_001,
    )
    scale = 200.0 / 72.0
    assert bbox.coord_origin == CoordOrigin.TOPLEFT
    assert bbox.l == pytest.approx(170.18537282269165 / scale)
    assert bbox.t == pytest.approx(169.39723781817833 / scale)
    assert bbox.r == pytest.approx(1080.6978511750278 / scale)
    assert bbox.b == pytest.approx(503.0565269474403 / scale)
    # Roughly where docling's layout model finds the same text block.
    assert bbox.l == pytest.approx(61.3, abs=1.0)
    assert bbox.r == pytest.approx(389.1, abs=2.0)


@pytest.mark.parametrize("coords", [YARROW_HEADER, PARAGRAPH])
def test_absolute_pixel_coords_land_inside_page(coords):
    bbox = reference_coords_to_page_bbox(
        coords, page_width=PAGE_001[0], page_height=PAGE_001[1], geometry=GEOMETRY_001
    )
    assert 0.0 <= bbox.l < bbox.r <= PAGE_001[0]
    assert 0.0 <= bbox.t < bbox.b <= PAGE_001[1]


def test_normalized_coords_are_scaled_by_page_size():
    bbox = reference_coords_to_page_bbox(
        _rect(0.1, 0.2, 0.5, 0.25),
        page_width=400.0,
        page_height=800.0,
        geometry=GEOMETRY_001,
    )
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx(
        (40.0, 160.0, 200.0, 200.0)
    )


def test_crop_box_offset_is_removed():
    # 01030000000032.pdf: 612 x 792 MediaBox, 431.928 x 648.03 CropBox.
    geometry = PdfPageGeometry(
        media_box=(0.0, 0.0, 612.0, 792.0),
        crop_box=(83.16, 96.264, 515.088, 744.294),
    )
    scale = 200.0 / 72.0
    # A box whose top-left corner is exactly at the CropBox top-left corner.
    coords = _rect(
        83.16 * scale, (792.0 - 744.294) * scale, 183.16 * scale, 147.706 * scale
    )
    bbox = reference_coords_to_page_bbox(
        coords, page_width=431.928, page_height=648.03, geometry=geometry
    )
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx((0.0, 0.0, 100.0, 100.0))


def test_reference_image_size_override():
    assert "01030000000141.pdf" in DPBENCH_REFERENCE_IMAGE_SIZES
    geometry = PdfPageGeometry(
        media_box=(0.0, 0.0, 1728.0, 2592.0), crop_box=(0.0, 0.0, 1728.0, 2592.0)
    )
    bbox = reference_coords_to_page_bbox(
        _rect(602.5, 905.0, 1205.0, 1810.0),
        page_width=1728.0,
        page_height=2592.0,
        geometry=geometry,
        reference_image_size=(1205.0, 1810.0),
    )
    assert (bbox.l, bbox.t, bbox.r, bbox.b) == pytest.approx(
        (864.0, 1296.0, 1728.0, 2592.0)
    )


def test_page_geometry_from_pdf(tmp_path: Path):
    src = tmp_path / "src.pdf"
    c = canvas.Canvas(str(src), pagesize=(612, 792))
    c.drawString(100, 700, "hello")
    c.save()

    writer = PdfWriter()
    page = PdfReader(str(src)).pages[0]
    page.cropbox = RectangleObject((83.16, 96.264, 515.088, 744.294))
    writer.add_page(page)
    cropped = tmp_path / "cropped.pdf"
    with open(cropped, "wb") as fw:
        writer.write(fw)

    geometry = PdfPageGeometry.from_pdf(cropped)
    assert geometry.media_box == pytest.approx((0.0, 0.0, 612.0, 792.0))
    assert geometry.crop_box == pytest.approx((83.16, 96.264, 515.088, 744.294))


def test_builder_pins_dataset_revision(tmp_path: Path):
    builder = DPBenchDatasetBuilder(target=tmp_path)
    assert isinstance(builder.dataset_source, HFSource)
    assert builder.dataset_source.revision == DPBENCH_HF_REVISION
