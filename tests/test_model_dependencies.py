r"""Offline checks for the model code docling-eval takes from docling / docling-ibm-models.

docling-ibm-models 4.0 dropped ``layoutmodel`` and ``reading_order``. These tests guard
that the evaluators keep importing and working with both the old and the new layout
without downloading any dataset or model.
"""

from docling_core.types.doc.base import CoordOrigin, Size
from docling_core.types.doc.document import RefItem
from docling_core.types.doc.labels import DocItemLabel

from docling_eval.evaluators.pixel.layout_labels import LayoutLabels
from docling_eval.evaluators.pixel_layout_evaluator import PixelLayoutEvaluator
from docling_eval.evaluators.readingorder_evaluator import (
    ReadingOrderEvaluator,
    ReadingOrderPageElement,
    ReadingOrderPredictor,
)


def test_layout_labels():
    layout_labels = LayoutLabels()

    canonical = layout_labels.canonical_categories()
    assert len(canonical) == 17
    assert canonical[0] == "Caption"
    assert canonical[16] == "Key-Value Region"

    shifted = layout_labels.shifted_canonical_categories()
    assert shifted[0] == "Background"
    assert all(shifted[k + 1] == v for k, v in canonical.items())

    assert layout_labels.canonical_to_int()["Table"] == 8
    assert layout_labels.shifted_canonical_to_int()["Table"] == 9

    # Every canonical name must map onto a DocItemLabel
    for name in canonical.values():
        DocItemLabel(name.lower().replace(" ", "_").replace("-", "_"))


def test_pixel_layout_evaluator_label_matrices():
    evaluator = PixelLayoutEvaluator()
    id_to_name = evaluator._matrix_id_to_name
    label_to_id = evaluator._matrix_doclabelitem_to_id

    assert id_to_name[0] == "Background"
    assert len(id_to_name) == 18
    assert label_to_id[DocItemLabel.CAPTION] == 1
    assert label_to_id[DocItemLabel.TABLE] == 9
    assert label_to_id[DocItemLabel.KEY_VALUE_REGION] == 17


def test_reading_order_predictor():
    page_size = Size(width=100, height=100)

    def element(cid: int, label: DocItemLabel, t: float, b: float):
        return ReadingOrderPageElement(
            cid=cid,
            ref=RefItem(cref=f"#/texts/{cid}"),
            text="dummy",
            page_no=1,
            page_size=page_size,
            label=label,
            l=10,
            r=90,
            t=t,
            b=b,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )

    elements = [
        element(1, DocItemLabel.TEXT, t=60, b=40),
        element(0, DocItemLabel.TITLE, t=90, b=80),
        element(2, DocItemLabel.TEXT, t=30, b=10),
    ]
    sorted_elements = ReadingOrderPredictor().predict_reading_order(
        page_elements=elements
    )
    assert [el.cid for el in sorted_elements] == [0, 1, 2]

    assert ReadingOrderEvaluator() is not None
