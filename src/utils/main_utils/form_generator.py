import os
import pandas as pd


SECTION_MAPPING = {
    "Policy Details": [
        "months_as_customer",
        "policy_state",
        "policy_csl",
        "policy_deductable",
        "policy_annual_premium",
        "umbrella_limit",
        "policy_bind_year",
        "policy_bind_month",
        "policy_bind_day",
    ],

    "Customer Details": [
        "age",
        "insured_sex",
        "insured_education_level",
        "insured_occupation",
        "insured_hobbies",
        "insured_relationship",
        "capital_gains",
        "capital_loss",
    ],

    "Incident Details": [
        "incident_type",
        "collision_type",
        "incident_severity",
        "authorities_contacted",
        "incident_state",
        "incident_city",
        "incident_hour_of_the_day",
        "incident_month",
        "incident_day",
        "number_of_vehicles_involved",
        "property_damage",
        "bodily_injuries",
        "witnesses",
        "police_report_available",
    ],

    "Claim Details": [
        "injury_claim",
        "property_claim",
        "vehicle_claim",
    ],

    "Vehicle Details": [
        "auto_make",
        "auto_model",
        "auto_year",
    ],
}


def pretty_label(col):
    return col.replace("_", " ").replace("-", " ").title()


def numeric_field(col, series):

    minimum = series.min()
    maximum = series.max()

    return f"""
<div class="col-lg-6 col-md-6 mb-4">

    <label class="form-label fw-semibold">
        {pretty_label(col)}
    </label>

    <input
        type="number"
        class="form-control"
        name="{col}"
        min="{minimum}"
        max="{maximum}"
        step="any"
        placeholder="Enter value ({minimum} - {maximum})"
        required>

    <div class="form-text text-primary">
        Allowed Range : {minimum} — {maximum}
    </div>

</div>
"""


def categorical_field(col, series):

    values = sorted(series.dropna().astype(str).unique())

    options = ""

    for value in values:
        options += f'<option value="{value}">{value}</option>\n'

    return f"""
<div class="col-lg-6 col-md-6 mb-4">

    <label class="form-label fw-semibold">
        {pretty_label(col)}
    </label>

    <select
        class="form-select"
        name="{col}"
        required>

        {options}

    </select>

</div>
"""


def generate_html_form(df, target_column,
                       output_path="templates/form_fields.html"):

    html = ""

    used_columns = set()

    for section, columns in SECTION_MAPPING.items():

        html += f"""

<div class="card shadow-sm mb-4">

    <div class="card-header bg-primary text-white">

        <h5 class="mb-0">{section}</h5>

    </div>

    <div class="card-body">

        <div class="row">

"""

        for col in columns:

            if col not in df.columns:
                continue

            used_columns.add(col)

            if df[col].dtype == "object":

                html += categorical_field(col, df[col])

            else:

                html += numeric_field(col, df[col])

        html += """

        </div>

    </div>

</div>

"""

    remaining = [c for c in df.columns if c not in used_columns and c != target_column]

    if remaining:

        html += """

<div class="card shadow-sm mb-4">

<div class="card-header bg-secondary text-white">

<h5 class="mb-0">Other Features</h5>

</div>

<div class="card-body">

<div class="row">

"""

        for col in remaining:

            if df[col].dtype == "object":
                html += categorical_field(col, df[col])
            else:
                html += numeric_field(col, df[col])

        html += """

</div>

</div>

</div>

"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✔ HTML form generated successfully at {output_path}")