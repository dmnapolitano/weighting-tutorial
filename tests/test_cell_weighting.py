import pandas
from pandas.testing import assert_frame_equal

from mrp.cell_weighting import CellReweighter, RakeReweighter

# examples from
# Kalton, G., & Flores-Cervantes, I. (2003). Weighting Methods. Journal of Official Statistics, 19(2), 81–97.


COLS = ["B1", "B2", "B3"]
INDEX = ["A1", "A2", "A3", "A4"]
CT_DF = pandas.DataFrame([[20, 40, 40], [50, 140, 310], [100, 50, 50], [30, 100, 70]],
                         columns=COLS, index=INDEX)
SAMPLE_DF = pandas.DataFrame([[80, 40, 55], [60, 150, 340], [170, 60, 200], [55, 165, 125]],
                             index=INDEX, columns=COLS)


def test_basic_cell_weighting():
    expected_df = pandas.DataFrame([[4, 1, 1.38], [1.2, 1.07, 1.1], [1.7, 1.2, 4], [1.83, 1.65, 1.79]],
                                   columns=COLS, index=INDEX)
    
    cr = CellReweighter(CT_DF, SAMPLE_DF, SAMPLE_DF, COLS)
    current_df = cr.reweight(return_weights=True)

    assert_frame_equal(expected_df, current_df, rtol=1e-2)


def test_basic_raking():
    expected_df = pandas.DataFrame([[1.81, 1.45, 2.02], [1.08, 0.87, 1.21], [2.2, 1.76, 2.45], [1.83, 1.47, 2.04]],
                                   columns=COLS, index=INDEX)

    rr = RakeReweighter(CT_DF, SAMPLE_DF, SAMPLE_DF, COLS)
    current_df = rr.reweight(return_weights=True)

    assert_frame_equal(expected_df, current_df, rtol=1e-2)
