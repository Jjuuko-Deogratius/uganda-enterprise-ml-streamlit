from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE / 'gsec12_2.csv',
    BASE.parent / 'gsec12_2.csv',
    Path('/mnt/data/gsec12_2.csv'),
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError('Could not find gsec12_2.csv. Place it beside train_models.py and rerun.')

MODELS_DIR = BASE / 'models'
MODELS_DIR.mkdir(exist_ok=True)

raw = pd.read_csv(DATA_PATH)

rename_map = {
    'hhid': 'household_id',
    'entID_unps': 'enterprise_id',
    'h12q01_1': 'business_sector',
    'h12q04': 'start_year',
    'h12q16': 'business_acquisition_method',
    'h12q05': 'startup_funding_source',
    'h12q15': 'business_location',
    'h12q06_1': 'used_funding_for_expansion',
    'h12q09': 'months_operated_last_12_months',
    'h12q10': 'monthly_gross_revenue',
    'h12q11A': 'paid_workers',
    'h12q11B': 'unpaid_workers',
    'h12q13': 'monthly_raw_material_cost',
    'h12q14': 'monthly_other_operating_cost',
    'h12q21': 'vat_registration_status',
    'h12q22': 'income_tax_registration_status',
    'h12q17': 'revenue_trend',
}
selected = list(rename_map.keys())
df = raw[selected].copy().rename(columns=rename_map)

# Cleaning rules used in the notebooks.
df.loc[df['months_operated_last_12_months'] > 12, 'months_operated_last_12_months'] = np.nan
df.loc[df['start_year'] > 2020, 'start_year'] = np.nan
df.loc[df['paid_workers'] > 100, 'paid_workers'] = np.nan
df.loc[df['unpaid_workers'] > 100, 'unpaid_workers'] = np.nan
df['business_age'] = 2020 - df['start_year']

# Survey-code mappings.
MAPPINGS = {
    'business_sector': {
        1: 'Manufacturing, mining & quarrying',
        2: 'Trade & Services',
        3: 'Hotels, Bars & Restaurants',
        4: 'Forestry',
        5: 'Agriculture',
    },
    'business_acquisition_method': {
        1: 'Founded',
        2: 'Purchased',
        3: 'Inherited after death of family member',
        4: 'Allocated by family',
        5: 'Gift from non-household member',
        96: 'Other',
        98: "Don't Know",
    },
    'startup_funding_source': {
        1: "Didn't need any money",
        2: "Own/household's savings",
        3: 'Commercial/Development Bank',
        4: 'Deposit Taking Microfinance Institution',
        5: 'SACCO',
        6: 'Local group',
        7: 'NGO',
        96: 'Other',
    },
    'business_location': {
        1: 'Home inside the residence',
        2: 'Home outside the residence',
        3: 'Industrial site',
        4: 'Traditional market',
        5: 'Commercial District shop',
        6: 'Roadside',
        7: 'Other Fixed place',
        8: 'Mobile',
        98: "Don't Know",
    },
    'used_funding_for_expansion': {1: 'Yes', 2: 'No'},
    'vat_registration_status': {1: 'Yes', 2: 'No', 8: 'Refused', 9: "Don't Know"},
    'income_tax_registration_status': {1: 'Yes', 2: 'No', 8: 'Refused', 9: "Don't Know"},
}

for col, mp in MAPPINGS.items():
    df[col] = df[col].map(mp)

# Shared categorical predictors.
categorical_features = [
    'business_sector',
    'business_acquisition_method',
    'startup_funding_source',
    'business_location',
    'used_funding_for_expansion',
    'vat_registration_status',
    'income_tax_registration_status',
]

# ---------------- Regression deployment model ----------------
reg = df.dropna(subset=['monthly_gross_revenue']).copy()
reg['log_monthly_raw_material_cost'] = np.log1p(reg['monthly_raw_material_cost'])
reg['log_monthly_other_operating_cost'] = np.log1p(reg['monthly_other_operating_cost'])
reg['log_monthly_gross_revenue'] = np.log1p(reg['monthly_gross_revenue'])

reg_numerical = [
    'business_age',
    'months_operated_last_12_months',
    'paid_workers',
    'unpaid_workers',
    'log_monthly_raw_material_cost',
    'log_monthly_other_operating_cost',
]
reg_features = reg_numerical + categorical_features

reg_preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ]), reg_numerical),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
    ]), categorical_features),
])

regression_model = Pipeline([
    ('preprocessor', reg_preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )),
])
regression_model.fit(reg[reg_features], reg['log_monthly_gross_revenue'])
joblib.dump(regression_model, MODELS_DIR / 'regression_random_forest.joblib')

# ---------------- Classification deployment model ----------------
clf = df[df['revenue_trend'].isin([1, 2, 3])].copy()
clf['revenue_trend'] = clf['revenue_trend'].map({
    1: 'Increased',
    2: 'Decreased',
    3: 'Remained the same',
})
clf['log_monthly_gross_revenue'] = np.log1p(clf['monthly_gross_revenue'])
clf['log_monthly_raw_material_cost'] = np.log1p(clf['monthly_raw_material_cost'])
clf['log_monthly_other_operating_cost'] = np.log1p(clf['monthly_other_operating_cost'])

clf_numerical = [
    'business_age',
    'months_operated_last_12_months',
    'paid_workers',
    'unpaid_workers',
    'log_monthly_gross_revenue',
    'log_monthly_raw_material_cost',
    'log_monthly_other_operating_cost',
]
clf_features = clf_numerical + categorical_features

clf_preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), clf_numerical),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
    ]), categorical_features),
])

classification_model = Pipeline([
    ('preprocessor', clf_preprocessor),
    ('model', LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        random_state=42,
    )),
])
classification_model.fit(clf[clf_features], clf['revenue_trend'])
joblib.dump(classification_model, MODELS_DIR / 'classification_logistic.joblib')

# Metadata for the UI and presentation explanations.
category_options = {
    col: sorted([str(v) for v in df[col].dropna().unique().tolist()])
    for col in categorical_features
}
metadata = {
    'survey_reference_year': 2020,
    'regression': {
        'algorithm': 'Random Forest Regression',
        'target': 'Monthly gross revenue (UGX)',
        'training_records': int(len(reg)),
        'notebook_test_metrics': {
            'R2': 0.837577,
            'MAE_UGX': 408166.76,
            'RMSE_UGX': 1143812.0,
        },
    },
    'classification': {
        'algorithm': 'Multinomial Logistic Regression',
        'target': 'Revenue trend',
        'training_records': int(len(clf)),
        'classes': classification_model.named_steps['model'].classes_.tolist(),
        'notebook_test_metrics': {
            'Accuracy': 0.545,
            'Balanced_Accuracy': 0.508,
            'Macro_F1': 0.493,
        },
    },
    'category_options': category_options,
}
with open(MODELS_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print('Saved deployment models to:', MODELS_DIR)
print('Regression records:', len(reg))
print('Classification records:', len(clf))
print('Classification classes:', metadata['classification']['classes'])
