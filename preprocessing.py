# Column definitions and the small cleaning step shared by training and the app.

TARGET = "Revenue"

NUM_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Weekend",
]

# OperatingSystems, Browser, Region and TrafficType are stored as integers in the
# csv but they are just category codes, so they are one-hot encoded along with
# Month and VisitorType.
CAT_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]

FEATURES = NUM_COLS + CAT_COLS


def to_binary(series):
    # Weekend / Revenue can come in as True/False, "TRUE"/"FALSE" or 1/0
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def clean(df):
    df = df.copy()
    df["Weekend"] = to_binary(df["Weekend"])
    if TARGET in df.columns:
        df[TARGET] = to_binary(df[TARGET])
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
    return df
