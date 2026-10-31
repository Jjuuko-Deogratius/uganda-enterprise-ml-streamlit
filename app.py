from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODELS = BASE / 'models'

st.set_page_config(
    page_title='Uganda Household Enterprise ML App',
    page_icon='📊',
    layout='wide',
)

@st.cache_resource
def load_assets():
    regression_model = joblib.load(MODELS / 'regression_random_forest.joblib')
    classification_model = joblib.load(MODELS / 'classification_logistic.joblib')
    with open(MODELS / 'metadata.json', encoding='utf-8') as f:
        metadata = json.load(f)
    return regression_model, classification_model, metadata

regression_model, classification_model, metadata = load_assets()
options = metadata['category_options']

st.title('Uganda Household Enterprise ML Predictor')
st.header(
    'Group 1 Developing Data Products capstone models '
)
st.subheader(
    'Using Uganda National Panel Survey 2019–2020 household-enterprise data.'
)

st.info(
    'These predictions are for academic demonstration and decision support. '
    'They should not be treated as guaranteed business outcomes or causal conclusions.'
)


def common_business_inputs(prefix: str, include_current_revenue: bool = False):
    """Collect a single enterprise profile from Streamlit widgets."""
    left, right = st.columns(2)

    with left:
        business_age = st.number_input(
            'Business age (years)', min_value=0, max_value=100, value=8, step=1,
            key=f'{prefix}_age'
        )
        months = st.number_input(
            'Months operated in the last 12 months', min_value=0, max_value=12,
            value=12, step=1, key=f'{prefix}_months'
        )
        paid = st.number_input(
            'Paid workers', min_value=0, max_value=100, value=0, step=1,
            key=f'{prefix}_paid'
        )
        unpaid = st.number_input(
            'Unpaid workers', min_value=0, max_value=100, value=0, step=1,
            key=f'{prefix}_unpaid'
        )
        raw_material_cost = st.number_input(
            'Monthly raw-material cost (UGX)', min_value=0.0, value=100000.0,
            step=10000.0, format='%.0f', key=f'{prefix}_raw_cost'
        )
        operating_cost = st.number_input(
            'Other monthly operating cost (UGX)', min_value=0.0, value=10000.0,
            step=5000.0, format='%.0f', key=f'{prefix}_other_cost'
        )
        current_revenue = None
        if include_current_revenue:
            current_revenue = st.number_input(
                'Current monthly gross revenue (UGX)', min_value=0.0,
                value=400000.0, step=50000.0, format='%.0f',
                key=f'{prefix}_revenue'
            )

    with right:
        sector = st.selectbox(
            'Business sector', options['business_sector'],
            key=f'{prefix}_sector'
        )
        acquisition = st.selectbox(
            'How was the business acquired?',
            options['business_acquisition_method'], key=f'{prefix}_acquisition'
        )
        funding = st.selectbox(
            'Main startup funding source', options['startup_funding_source'],
            key=f'{prefix}_funding'
        )
        location = st.selectbox(
            'Business location', options['business_location'],
            key=f'{prefix}_location'
        )
        expansion = st.selectbox(
            'Used funding for expansion?', options['used_funding_for_expansion'],
            key=f'{prefix}_expansion'
        )
        vat = st.selectbox(
            'VAT registration status', options['vat_registration_status'],
            key=f'{prefix}_vat'
        )
        income_tax = st.selectbox(
            'Income-tax registration status', options['income_tax_registration_status'],
            key=f'{prefix}_income_tax'
        )

    row = {
        'business_age': float(business_age),
        'months_operated_last_12_months': float(months),
        'paid_workers': float(paid),
        'unpaid_workers': float(unpaid),
        'log_monthly_raw_material_cost': float(np.log1p(raw_material_cost)),
        'log_monthly_other_operating_cost': float(np.log1p(operating_cost)),
        'business_sector': sector,
        'business_acquisition_method': acquisition,
        'startup_funding_source': funding,
        'business_location': location,
        'used_funding_for_expansion': expansion,
        'vat_registration_status': vat,
        'income_tax_registration_status': income_tax,
    }
    if include_current_revenue:
        row['log_monthly_gross_revenue'] = float(np.log1p(current_revenue))

    return pd.DataFrame([row]), current_revenue


tab_reg, tab_clf, tab_about = st.tabs([
    '💰 Revenue Regression',
    '📈 Revenue Trend Classification',
    'ℹ️ About the Models',
])

with tab_reg:
    st.header('Predict Monthly Gross Revenue')
    st.write(
        'Enter an enterprise profile. The deployed **Random Forest regression** pipeline '
        'predicts average monthly gross revenue.'
    )

    with st.form('regression_form'):
        reg_input, _ = common_business_inputs('reg', include_current_revenue=False)
        reg_submit = st.form_submit_button('Predict Monthly Revenue', type='primary')

    if reg_submit:
        predicted_log = regression_model.predict(reg_input)[0]
        predicted_ugx = max(float(np.expm1(predicted_log)), 0.0)
        st.success(f'Predicted monthly gross revenue: **UGX {predicted_ugx:,.0f}**')

        m1, m2, m3 = st.columns(3)
        metrics = metadata['regression']['notebook_test_metrics']
        m1.metric('Notebook test R²', f"{metrics['R2']:.3f}")
        m2.metric('Notebook test MAE', f"UGX {metrics['MAE_UGX']:,.0f}")
        m3.metric('Notebook test RMSE', f"UGX {metrics['RMSE_UGX']:,.0f}")

        with st.expander('See transformed model input'):
            st.dataframe(reg_input, use_container_width=True)

with tab_clf:
    st.header('Classify Revenue Performance Trend')
    st.write(
        'Enter the current enterprise profile. The deployed **multinomial Logistic '
        'Regression** pipeline classifies the trend as Increased, Decreased, or Remained the same.'
    )

    with st.form('classification_form'):
        clf_input, current_revenue = common_business_inputs(
            'clf', include_current_revenue=True
        )
        clf_submit = st.form_submit_button('Classify Revenue Trend', type='primary')

    if clf_submit:
        predicted_class = classification_model.predict(clf_input)[0]
        probabilities = classification_model.predict_proba(clf_input)[0]
        classes = classification_model.named_steps['model'].classes_

        st.success(f'Predicted revenue trend: **{predicted_class}**')

        probability_df = pd.DataFrame({
            'Revenue trend': classes,
            'Probability': probabilities,
        }).set_index('Revenue trend')
        st.bar_chart(probability_df)

        st.caption(
            'Probabilities express model confidence, not certainty. '
            'The classifier achieved moderate held-out test performance.'
        )

        m1, m2, m3 = st.columns(3)
        metrics = metadata['classification']['notebook_test_metrics']
        m1.metric('Notebook test Accuracy', f"{metrics['Accuracy']:.3f}")
        m2.metric('Balanced Accuracy', f"{metrics['Balanced_Accuracy']:.3f}")
        m3.metric('Macro F1', f"{metrics['Macro_F1']:.3f}")

        with st.expander('See transformed model input'):
            st.dataframe(clf_input, use_container_width=True)

with tab_about:
    st.header('How this deployment works')
    st.markdown(
        f"""
        **Regression model:** {metadata['regression']['algorithm']}  
        **Classification model:** {metadata['classification']['algorithm']}  
        **Survey context:** Uganda National Panel Survey 2019–2020, household enterprises.

        The saved `.joblib` files contain the **full fitted Scikit-learn pipelines**. This is important:
        the app is not only loading an estimator; it is loading the same preprocessing structure
        used in modelling, including missing-value imputation and one-hot encoding.

        For the financial inputs, the app applies the same `log(1 + x)` transformations used
        in the notebooks before passing the row to the fitted pipeline.

        The evaluation metrics displayed in the prediction tabs are the **held-out notebook test
        results**. The deployment models were refitted after model selection so the application can
        learn from all eligible labelled observations.
        """
    )

    st.subheader('Deployment flow')
    st.code(
        'User inputs → validation/transformation → saved preprocessing pipeline '
        '→ fitted model → prediction → human-readable result',
        language=None,
    )
