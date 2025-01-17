import pandas
from pandas.testing import assert_frame_equal
from samplics.weighting import SampleWeight

from src.cell_weighting import CellReweighter, RakeReweighter

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


def test_GREG_weighting():
    # reformat our crosstabs
    ct_df = CT_DF.copy().reset_index().melt(id_vars="index")
    ct_df["A"] = ct_df["index"].str[1].astype(float)
    ct_df["B"] = ct_df["variable"].str[1].astype(float)

    # get the population totals we want to weight by
    sample_df = SAMPLE_DF.copy().reset_index().melt(id_vars="index")
    sample_df["A"] = sample_df["index"].str[1].astype(float)
    sample_df["B"] = sample_df["variable"].str[1].astype(float)
    totals = {"A" : (sample_df["A"] * sample_df["value"]).sum(),
              "B" : (sample_df["B"] * sample_df["value"]).sum()}

    # apply GREG calibration and divide to get the weights
    ct_df["calibration"] = SampleWeight().calibrate(ct_df["value"], ct_df[["A", "B"]], totals)
    ct_df["weights"] = ct_df["calibration"] / ct_df["value"]

    # column x row of Table 2.D in Kalton et al. (2003)
    ct_df["expected"] = [1.21, 1.43, 1.66, 1.88,
                         1.17, 1.40, 1.62, 1.85,
                         1.14, 1.36, 1.59, 1.81]

    assert (ct_df["expected"] - ct_df["weights"].round(2)).mean() < 0.01
