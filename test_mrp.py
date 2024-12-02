import pandas


STATE_FIPS = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
              "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
              "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
              "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
              "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
              "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
              "47": "TN", "48": "TX", "49": "UT", "51": "VA", "50": "VT", "53": "WA", "54": "WV",
              "55": "WI", "56": "WY"}

ETH = {1 : "White", 2 : "Black", 3 : "Hispanic", 4 : "Asian", 5 : "Native American", 6 : "Mixed", 7 : "Other", 8 : "Middle Eastern"}

EDU = {1 : "No HS", 2 : "HS", 3 : "Some college", 4 : "Associates", 5 : "4-Year College", 6 : "Post-grad"}


def clean_cces(csv_path, seed=1010):
    # port of
    # https://bookdown.org/jl5522/MRP-case-studies/introduction-to-mister-p.html#appendix-downloading-and-processing-data
    # section 1.6.1
    df = pandas.read_csv(csv_path, low_memory=False, dtype={"inputstate" : str})
    df = df.set_index("caseid")
    df = df[["CC18_321d", "inputstate", "gender", "race", "birthyr", "educ"]].copy()

    ## Abortion -- dichotomous (0 - Oppose / 1 - Support)
    df["abortion"] = (df["CC18_321d"] - 2).abs()
    del df["CC18_321d"]
    
    df["inputstate"] = df["inputstate"].str.zfill(2)
    df["state"] = df["inputstate"].apply(lambda x : STATE_FIPS[x])
    del df["inputstate"]

    ## Gender -- dichotomous (coded as -0.5 Female, +0.5 Male)
    df["male"] = (df["gender"] - 2).abs() - 0.5
    del df["gender"]

    df["eth"] = df["race"].apply(lambda x : ETH[x])
    del df["race"]
    df["eth"] = df["eth"].apply(lambda x : "Other" if x in
                                ["Asian", "Other", "Middle Eastern",
                                 "Mixed", "Native American"] else x)

    # sticking with 2018 for reproducability
    df["age"] = 2018 - df["birthyr"]
    df["age"] = pandas.cut(df["age"], [0, 29, 39, 49, 59, 69, 120],
                           labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
                           ordered=True)
    del df["birthyr"]

    df["educ"] = df["educ"].apply(lambda x : EDU[x])
    df["educ"] = df["educ"].apply(lambda x : "Some college" if x in ["Some college", "Associates"] else x)

    df = df.dropna(axis=0).dropna(axis=1)
    
    return df.sample(n=5000, random_state=seed)


if __name__ == "__main__":
    cces_df = clean_cces("data/cces18_common_vv.csv.gz")
    poststrat_df = pandas.read_csv("data/poststrat_df.csv")
    statelevel_predictors_df = pandas.read_csv("data/statelevel_predictors.csv")

    print(cces_df)
    print(poststrat_df)
    print(statelevel_predictors_df)
