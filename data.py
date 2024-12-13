import pandas


CROSSTAB_COLS = ["CD1", "CD2", "CD3", "CD4", "VOTED_2020_TRUMP", "VOTED_2020_BIDEN", "VOTED_2020_DIDNT",
                 "CHILD_LT_18", "EDU_NO_COLL", "EDU_COLL"]
CROSSTAB_COL_GROUPS = [["CD1", "CD2", "CD3", "CD4"], ["VOTED_2020_TRUMP", "VOTED_2020_BIDEN", "VOTED_2020_DIDNT"],
                       ["CHILD_LT_18"], ["EDU_NO_COLL", "EDU_COLL"]]
CONST_COLS = ["CANDIDATE", "LV"]


# source: https://sos.iowa.gov/elections/pdf/VRStatsArchive/2024/CongOct24.pdf
CD_REG_DF = pandas.DataFrame(
    [[129541 + 48380, 127321 + 42982, 134188 + 44986, 85143 + 38710],
     [149869 + 34385, 149972 + 30288, 151254 + 33128, 191956 + 45281],
     [2394 + 584 + 1670 + 549, 2434 + 534 + 1509 + 528,
      2831 + 535 + 1674 + 514, 2388 + 493 + 1754 + 572]],
    index=["DEM", "REP", "OTHER"], columns=["CD1", "CD2", "CD3", "CD4"])


# source: AP Elections API for votes,
# https://sos.iowa.gov/elections/pdf/VRStatsArchive/2020/CongNov20.pdf for reg total
PAST_VOTE = {"Trump" : 897672,
             "Biden" : 759061,
             "Jorgensen" : 19637,
             "West" : 3210,
             "Hawkins" : 3075,
             "Blankenship" : 1707,
             "De La Fuente" : 1082,
             "King" : 546,
             "Pierce" : 544,
             }
PAST_VOTE["Didn't"] = 2245097 - sum(PAST_VOTE.values())


def load_crosstab_data():
    crosstab_df = pandas.read_csv("data/iowa_2024/iowa_2024_crosstab.csv")
    crosstab_df = crosstab_df[CONST_COLS + CROSSTAB_COLS].copy().fillna(0).set_index("CANDIDATE")

    # total_wgt = crosstab_df["LV"].loc["WGT"]
    # total_wgt - (crosstab_df["LV"].loc["HARRIS"] + crosstab_df["LV"].loc["TRUMP"])
    other_df = crosstab_df[~crosstab_df.index.isin(["HARRIS", "TRUMP", "Total Unweighted Respondents", "WGT"])].copy()
    other = other_df.sum(axis=0)
    other.name = "OTHER"
    other["LV"] = other["LV"] + 2

    crosstab_df = pandas.concat(
        [crosstab_df[crosstab_df.index.isin(["HARRIS", "TRUMP"])], pandas.DataFrame([other])]).reset_index(names=["CANDIDATE"])

    temp = None

    for group_cols in CROSSTAB_COL_GROUPS:
        df = crosstab_df[CONST_COLS + group_cols].copy()
        if group_cols[0] != "CD1":
            unk_col = group_cols[0].split("_")[0] + "_" + "UNK"
            df[unk_col] = df["LV"] - df[group_cols].sum(axis=1)
        if temp is None:
            temp = df.copy()
        else:
            temp = temp.merge(df, on=CONST_COLS, how="left")

    crosstab_df = temp.copy()

    return crosstab_df
