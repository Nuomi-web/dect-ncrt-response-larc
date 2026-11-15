import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# =========================
# 1. Load RF model
# =========================
@st.cache_resource
def load_model():
    return joblib.load('RF.pkl')

model_rf = load_model()

# =========================
# 2. Configure feature names
# =========================
feature_label = [
    'Zeff_original_glszm_ZoneEntropy_h3',
    'VMI_DL_311',
    'VMI_original_gldm_DependenceVariance_h2',
    'IC_original_glrlm_LongRunLowGrayLevelEmphasis_h2',
    'VMI_DL_91',
    'PEI_DL_11',
    'IC_original_glszm_GrayLevelNonUniformityNormalized_h1',
    'VMI_DL_248',
    'IC_original_glcm_ClusterShade_h1',
    'IC_DL_64',
    'IC_DL_286',
    'VMI_DL_176',
    'VMI_original_firstorder_Skewness_h1',
    'Zeff_DL_132',
    'Persistence_Mean_h1',
    'Zeff_DL_4',
    'IC_DL_197',
    'VMI_original_glszm_HighGrayLevelZoneEmphasis_h1',
    'Zeff_DL_19',
    'Differentiation'
]

# =========================
# 3. Default values
# =========================
default_values = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
]

# =========================
# 4. Streamlit input
# =========================
st.title('Web Predictor for GR to nCRT in Patients with LARC')
st.sidebar.header('Input Features')

inputs = {}

for i, feature in enumerate(feature_label):
    default_val = default_values[i]

    if feature == 'Differentiation':
        options = ['Well/Moderate', 'Poor']
        default_index = 0 if int(default_val) == 0 else 1

        inputs[feature] = st.sidebar.radio(
            label=feature,
            options=options,
            index=default_index
        )

    else:
        inputs[feature] = st.sidebar.number_input(
            label=feature,
            min_value=-20.0,
            max_value=30.0,
            value=round(float(default_val), 8),
            format="%.8f"
        )

# =========================
# 5. Map categorical variable
# =========================
diff_map = {
    'Well/Moderate': 0,
    'Poor': 1
}

inputs['Differentiation'] = diff_map[inputs['Differentiation']]

# =========================
# 6. Convert to DataFrame
# =========================
input_df = pd.DataFrame([inputs], columns=feature_label)

# =========================
# 7. Prediction and SHAP
# =========================
if st.sidebar.button('Predict'):

    try:
        # ---------- 7-1. RF prediction ----------
        if not hasattr(model_rf, "predict_proba"):
            raise TypeError(
                "当前加载的模型不支持 predict_proba，请确认 RF.pkl 是 RandomForestClassifier 分类模型。"
            )

        # 预测类别 1，即 GR to nCRT 的概率
        prediction_prob = model_rf.predict_proba(input_df)[0, 1]

        threshold = 0.5
        prediction_class = int(prediction_prob >= threshold)

        # ---------- 7-2. Display prediction ----------
        st.subheader('Prediction Result')

        st.markdown(
            f"""
            <div style="font-size:24px;">
                Predicted probability of <b>GR to nCRT</b>:
                <span style="color:red; font-size:32px;">
                    {prediction_prob:.8f}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


        # =========================
        # 8. SHAP explanation for RF
        # =========================
        st.subheader('SHAP Force Plot')

        explainer = shap.TreeExplainer(model_rf)
        shap_values_all = explainer.shap_values(input_df)

        # ---------- 8-1. Extract SHAP values of class 1 ----------
        if isinstance(shap_values_all, list):
            shap_values_class1 = shap_values_all[1]

        elif isinstance(shap_values_all, np.ndarray):
            if shap_values_all.ndim == 3:
                shap_values_class1 = shap_values_all[:, :, 1]
            elif shap_values_all.ndim == 2:
                shap_values_class1 = shap_values_all
            else:
                raise ValueError(
                    f"Unexpected SHAP values shape: {shap_values_all.shape}"
                )
        else:
            raise TypeError("Unsupported SHAP values type.")

        # ---------- 8-2. Extract expected value of class 1 ----------
        expected_value = explainer.expected_value

        if isinstance(expected_value, list):
            expected_value_class1 = expected_value[1]

        elif isinstance(expected_value, np.ndarray):
            if expected_value.ndim == 1 and len(expected_value) == 2:
                expected_value_class1 = expected_value[1]
            else:
                expected_value_class1 = expected_value

        else:
            expected_value_class1 = expected_value

        # ---------- 9. Draw SHAP force plot ----------
        plt.figure()

        shap.force_plot(
            expected_value_class1,
            shap_values_class1[0, :],
            input_df.iloc[0, :],
            feature_names=feature_label,
            matplotlib=True,
            contribution_threshold=0.135,
            show=False
        )

        plt.savefig(
            "shap_force_plot_RF.png",
            bbox_inches='tight',
            dpi=150
        )
        plt.close()

        st.image("shap_force_plot_RF.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
