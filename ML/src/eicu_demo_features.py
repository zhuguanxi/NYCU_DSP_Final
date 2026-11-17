"""Map eICU Demo observations to the original P12 feature extractor."""

from pathlib import Path

import numpy as np
import pandas as pd

from prepare import ROOT, STATIC, UNITS, clean, extract_record


DEMO = ROOT / "ML/dataset/EICU/eicu-crd-demo-2.0.1"
STAY = "patientunitstayid"
WINDOW = 2880
MODES = ("hourly", "native")
EVENT = [STAY, "minute", "variable", "value"]
UNMAPPED = {"Mg", "MechVent", "Urine"}
PERIODIC = {
    "temperature": "Temp",
    "heartrate": "HR",
    "respiration": "RespRate",
    "systemicsystolic": "SysABP",
    "systemicdiastolic": "DiasABP",
    "systemicmean": "MAP",
}
APERIODIC = {
    "noninvasivesystolic": "NISysABP",
    "noninvasivediastolic": "NIDiasABP",
    "noninvasivemean": "NIMAP",
}
LAB = {
    ("albumin", 1, "g/dl"): "Albumin",
    ("alkaline phos.", 1, "units/l"): "ALP",
    ("alt (sgpt)", 1, "units/l"): "ALT",
    ("ast (sgot)", 1, "units/l"): "AST",
    ("total bilirubin", 1, "mg/dl"): "Bilirubin",
    ("bun", 1, "mg/dl"): "BUN",
    ("total cholesterol", 1, "mg/dl"): "Cholesterol",
    ("creatinine", 1, "mg/dl"): "Creatinine",
    ("glucose", 1, "mg/dl"): "Glucose",
    ("bicarbonate", 1, "mmol/l"): "HCO3",
    ("hct", 3, "%"): "HCT",
    ("potassium", 1, "mmol/l"): "K",
    ("lactate", 1, "mmol/l"): "Lactate",
    ("sodium", 1, "mmol/l"): "Na",
    ("platelets x 1000", 3, "k/mcl"): "Platelets",
    ("wbc x 1000", 3, "k/mcl"): "WBC",
    ("paco2", 7, "mm hg"): "PaCO2",
    ("pao2", 7, "mm hg"): "PaO2",
    ("ph", 7, ""): "pH",
    ("o2 sat (%)", 7, "%"): "SaO2",
    ("troponin - i", 1, "ng/ml"): "TroponinI",
    ("troponin - t", 1, "ng/ml"): "TroponinT",
    ("fio2", 7, "%"): "FiO2",
}
INPUT_TABLES = (
    "patient",
    "lab",
    "vitalPeriodic",
    "vitalAperiodic",
    "nurseCharting",
    "respiratoryCharting",
    "intakeOutput",
)


def read_table(directory, table, columns):
    return pd.read_csv(directory / f"{table}.csv.gz", usecols=columns, low_memory=False)


def normalize(values):
    return values.astype("string").str.strip().str.casefold()


def select_cohort(directory, *, end_minute=WINDOW):
    fields = [
        STAY, "uniquepid", "hospitalid", "hospitaldischargestatus",
        "hospitaldischargeoffset", "unitdischargeoffset", "unitvisitnumber",
        "age", "gender", "admissionheight", "admissionweight", "unittype",
    ]
    patients = read_table(directory, "patient", fields).set_index(STAY).sort_index()
    if patients.index.hasnans or not patients.index.is_unique:
        raise ValueError("eICU stay keys must be complete and unique")

    age = pd.to_numeric(patients.age.astype("string").str.strip().replace({"> 89": "90", ">89": "90"}), errors="coerce")
    label = normalize(patients.hospitaldischargestatus).map({"alive": 0, "expired": 1})
    filters = {
        "adult": age.ge(18),
        "first_icu_in_hospitalization": patients.unitvisitnumber.eq(1),
        f"icu_stay_at_least_{end_minute // 60}h": patients.unitdischargeoffset.ge(end_minute),
        "known_hospital_outcome": label.notna(),
        f"hospital_exit_after_{end_minute // 60}h": patients.hospitaldischargeoffset.gt(end_minute),
    }
    counts = {"all_stays": len(patients)}
    eligible = pd.Series(True, index=patients.index)
    for name, condition in filters.items():
        eligible &= condition.fillna(False)
        counts[name] = int(eligible.sum())

    cohort = patients.loc[eligible, ["uniquepid", "hospitalid"]].copy()
    cohort["y"] = label.loc[eligible].astype("int8")
    if cohort.empty or cohort.uniquepid.isna().any() or cohort.y.nunique() != 2:
        raise ValueError("Eligible stays need patient keys and both outcome classes")

    static = pd.DataFrame(index=cohort.index)
    static["Age"] = age.loc[eligible].astype(float)
    static["Gender"] = normalize(patients.loc[eligible, "gender"]).map({"female": 0.0, "male": 1.0})
    static["ICUType"] = normalize(patients.loc[eligible, "unittype"]).map({"csicu": 2.0, "micu": 3.0, "sicu": 4.0})
    for variable, field in (("Height", "admissionheight"), ("Weight", "admissionweight")):
        value = pd.to_numeric(patients.loc[eligible, field], errors="coerce")
        static[variable] = value.where(np.isfinite(value) & value.gt(0))
    return cohort, static.loc[:, list(STATIC)], counts


def windowed(directory, table, cohort, offset, values, entry=None, *, end_minute=WINDOW):
    columns = [STAY, offset, *values]
    if entry:
        columns.append(entry)
    raw = read_table(directory, table, columns)
    eligible = raw[STAY].isin(cohort) & raw[offset].between(0, end_minute)
    late = raw[entry].gt(end_minute) if entry else pd.Series(False, index=raw.index)
    if entry and end_minute < WINDOW:
        late |= raw[entry].isna()
    excluded = int((eligible & late).sum())
    frame = raw.loc[eligible & ~late].rename(columns={offset: "minute"})
    frame["minute"] = frame.minute.astype("int32")
    return frame, excluded


def numeric_events(frame):
    frame = frame.copy()
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    frame = frame.loc[np.isfinite(frame.value)]
    groups = frame.groupby("variable", sort=False)
    valid = []
    for name, rows in groups:
        allowed = [value for value in rows.value.unique() if clean(name, value)[1] == "valid"]
        valid.append(rows.loc[rows.value.isin(allowed), EVENT])
    return pd.concat(valid, ignore_index=True) if valid else frame.loc[:, EVENT]


def chart_events(frame, selected, variable, field):
    return frame.loc[selected, [STAY, "minute", field]].rename(columns={field: "value"}).assign(variable=variable)


def monitor_events(directory, table, cohort, mapping, *, end_minute=WINDOW):
    frame, _ = windowed(directory, table, cohort, "observationoffset", mapping, end_minute=end_minute)
    frame = frame.rename(columns=mapping).melt(
        id_vars=[STAY, "minute"], var_name="variable", value_name="value",
    )
    return merge_minutes(numeric_events(frame))


def merge_minutes(frame, method="median"):
    return frame.groupby([STAY, "minute", "variable"], sort=False, as_index=False).value.agg(method)


def lab_events(directory, cohort, *, end_minute=WINDOW):
    frame, late = windowed(
        directory, "lab", cohort, "labresultoffset",
        ("labname", "labtypeid", "labmeasurenamesystem", "labresult"),
        "labresultrevisedoffset",
        end_minute=end_minute,
    )
    keys = zip(normalize(frame.labname), frame.labtypeid, normalize(frame.labmeasurenamesystem).fillna(""))
    frame["variable"] = [LAB.get(key) for key in keys]
    frame = frame.loc[frame.variable.notna()].rename(columns={"labresult": "value"})
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    oxygen = frame.variable.eq("FiO2")
    frame = frame.loc[~oxygen | frame.value.between(21, 100)].copy()
    frame.loc[frame.variable.eq("FiO2"), "value"] /= 100
    return merge_minutes(numeric_events(frame)), late


def nurse_events(directory, cohort, *, end_minute=WINDOW):
    frame, late = windowed(
        directory, "nurseCharting", cohort, "nursingchartoffset",
        ("nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"),
        "nursingchartentryoffset",
        end_minute=end_minute,
    )
    label = normalize(frame.nursingchartcelltypevallabel)
    name = normalize(frame.nursingchartcelltypevalname)
    gcs_mask = (label.eq("glasgow coma score") & name.eq("gcs total")) | (
        label.eq("score (glasgow coma scale)") & name.eq("value")
    )
    gcs = numeric_events(chart_events(frame, gcs_mask, "GCS", "nursingchartvalue"))
    gcs = merge_minutes(gcs.loc[gcs.value.mod(1).eq(0)], method="min")
    temp = chart_events(frame, label.eq("temperature") & name.eq("temperature (c)"), "Temp", "nursingchartvalue")
    return gcs, merge_minutes(numeric_events(temp)), late


def respiratory_events(directory, cohort, *, end_minute=WINDOW):
    frame, late = windowed(
        directory, "respiratoryCharting", cohort, "respchartoffset",
        ("respchartvaluelabel", "respchartvalue"), "respchartentryoffset",
        end_minute=end_minute,
    )
    label = normalize(frame.respchartvaluelabel)
    allowed = label.isin({"fio2", "fio2 (%)", "o2 percentage", "set fraction of inspired oxygen (fio2)"})
    frame = frame.loc[allowed].copy()
    text = frame.respchartvalue.astype("string").str.strip()
    raw = pd.to_numeric(text.str.removesuffix("%").str.strip(), errors="coerce")
    explicit_percent = text.str.endswith("%", na=False) | label.loc[allowed].isin({"fio2 (%)", "o2 percentage"})
    fractional = raw.between(0.21, 1) & ~explicit_percent
    frame["value"] = raw.div(100).where(raw.between(21, 100)).where(~fractional, raw)
    return merge_minutes(numeric_events(frame.assign(variable="FiO2"))), late


def weight_events(directory, cohort, *, end_minute=WINDOW):
    frame, late = windowed(
        directory, "intakeOutput", cohort, "intakeoutputoffset",
        ("celllabel", "cellvaluenumeric"), "intakeoutputentryoffset",
        end_minute=end_minute,
    )
    weight = chart_events(frame, normalize(frame.celllabel).eq("bodyweight (kg)"), "Weight", "cellvaluenumeric")
    weight = numeric_events(weight)
    return merge_minutes(weight.loc[weight.value.gt(0)]), late


def hourly_medians(periodic):
    return (
        periodic.assign(hour=periodic.minute // 60)
        .groupby([STAY, "variable", "hour"], as_index=False, sort=False)
        .agg(minute=("minute", "max"), value=("value", "median"))
        .loc[:, EVENT]
    )


def p12_features(events, static):
    rows = {}
    ordered = events.sort_values(["minute", "variable"], kind="stable")
    groups = ordered.groupby(STAY, sort=False).indices
    for stay, admission in static.iterrows():
        lines = ["Time,Parameter,Value", f"00:00,RecordID,{stay}"]
        lines.extend(f"00:00,{name},{value}" for name, value in admission.items() if pd.notna(value))
        if stay in groups:
            observations = ordered.iloc[groups[stay]][["minute", "variable", "value"]]
            lines.extend(
                f"{minute // 60:02}:{minute % 60:02},{variable},{value}"
                for minute, variable, value in observations.itertuples(index=False, name=None)
            )
        feature, _, _ = extract_record("\n".join(lines).encode(), stay)
        feature["static_Weight"] = admission["Weight"]
        rows[stay] = feature
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(STAY)


def build_features(directory: Path, feature_extension=None, *, end_minute=WINDOW):
    if end_minute not in (1440, WINDOW):
        raise ValueError("Expected the 24-hour or 48-hour observation window")
    cohort, static, counts = select_cohort(directory, end_minute=end_minute)
    periodic = monitor_events(directory, "vitalPeriodic", cohort.index, PERIODIC, end_minute=end_minute)
    aperiodic = monitor_events(directory, "vitalAperiodic", cohort.index, APERIODIC, end_minute=end_minute)
    lab, lab_late = lab_events(directory, cohort.index, end_minute=end_minute)
    gcs, temp, nurse_late = nurse_events(directory, cohort.index, end_minute=end_minute)
    resp, resp_late = respiratory_events(directory, cohort.index, end_minute=end_minute)
    weight, weight_late = weight_events(directory, cohort.index, end_minute=end_minute)

    periodic = periodic.loc[~(periodic.variable.eq("Temp") & periodic[STAY].isin(temp[STAY]))]
    lab_oxygen = lab.loc[lab.variable.eq("FiO2"), [STAY, "minute"]]
    resp = resp.merge(lab_oxygen.assign(has_lab=True), how="left", on=[STAY, "minute"], validate="one_to_one")
    resp = resp.loc[resp.has_lab.isna(), EVENT]
    fixed = pd.concat([aperiodic, lab, gcs, temp, resp, weight], ignore_index=True)
    modes = {"hourly": hourly_medians(periodic), "native": periodic}
    features = {}
    dropped_late = {"lab": lab_late, "nurseCharting": nurse_late, "respiratoryCharting": resp_late, "intakeOutput": weight_late}
    mapped = set(LAB.values()) | set(PERIODIC.values()) | set(APERIODIC.values()) | {"GCS", "Weight"}
    if mapped != set(UNITS) - UNMAPPED:
        raise ValueError("The eICU mapping does not cover the expected P12 variables")
    for name, measurements in modes.items():
        events = pd.concat([measurements, fixed], ignore_index=True)
        if not events.minute.between(0, end_minute).all():
            raise ValueError("An observation falls outside the requested window")
        result = p12_features(events, static)
        if not result.index.equals(cohort.index) or result.shape[1] != 435:
            raise ValueError("Misaligned P12 input shape or stay order")
        if result[[f"{variable}__count" for variable in UNMAPPED]].to_numpy().any():
            raise ValueError("An unmapped signal unexpectedly has observations")
        if feature_extension is not None:
            extra = feature_extension(events, cohort.index)
            if not extra.index.equals(result.index) or set(extra.columns) & set(result.columns):
                raise ValueError("Extra features differ in stay order or overlap with original features")
            result = pd.concat([result, extra], axis=1)
        features[name] = result
    return cohort, features, counts, dropped_late
