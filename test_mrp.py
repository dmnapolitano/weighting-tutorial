import pandas
import bambi as bmb
import arviz as az


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

SEED = 1010


def clean_cces(csv_path, statelevel_predictors_df, sample=False):
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
    if sample:
        df = df.sample(n=5000, random_state=SEED)
    df = df.reset_index().rename(columns={"index" : "caseid"})

    # grouping based on https://bambinos.github.io/bambi/notebooks/mister_p.html
    df = (df.groupby(["state", "eth", "male", "age", "educ"], observed=False)
          .agg({"caseid": "nunique", "abortion": "sum"})
          .reset_index()
          .sort_values("abortion", ascending=False)
          .rename({"caseid": "n"}, axis=1)
          .merge(statelevel_predictors_df, on=["state"], how="left")
          )
    # TODO: about 240 region and repvote values are missing
    df = df.dropna(axis=0).dropna(axis=1)

    return df


def fit_multilevel_regression(cces_df):
    # original formula from the MRP Case Studies:
    # abortion ~ (1 | state) + (1 | eth) + (1 | educ) + male +
    #    (1 | male:eth) + (1 | educ:age) + (1 | educ:eth) +
    #    repvote + factor(region)
    
    formula = "p(abortion, n) ~ (1 | state) + (1 | eth) + (1 | educ) + (1 | male:eth) + (1 | educ:age) + (1 | educ:eth) + male + repvote + C(region)"
    
    fit = bmb.Model(
        formula,
        family="binomial",
        link="logit",
        data=cces_df,
    )
    result = fit.fit(random_seed=SEED, target_accept=0.99,
                     chains=2,
                     idata_kwargs={"log_likelihood": True})

    print(fit)
    print()
    print(az.summary(result, var_names=["Intercept", "1|state", "male", "1|educ", "1|eth", "repvote", "1|educ:age", "1|educ:eth"])
          .sort_values(by=["mean"], ascending=False))


if __name__ == "__main__":
    statelevel_predictors_df = pandas.read_csv("data/statelevel_predictors.csv")
    cces_df = clean_cces("data/cces18_common_vv.csv.gz", statelevel_predictors_df, sample=True)
    poststrat_df = pandas.read_csv("data/poststrat_df.csv")

    fit_multilevel_regression(cces_df)
