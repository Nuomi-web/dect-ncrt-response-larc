import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


# =========================================================
# 1. Load RF model
# =========================================================
@st.cache_resource
def load_model():
    model_package = joblib.load("RF.pkl")

    # 如果保存的是完整模型字典
    if isinstance(model_package, dict) and "model" in model_package:
        return model_package["model"]

    # 如果直接保存的是随机森林模型
    return model_package


model_rf = load_model()


# =========================================================
# 2. Feature names and order
# =========================================================
feature_label = [
    "Zeff_original_glszm_ZoneEntropy_h3",
    "VMI_DL_311",
    "VMI_original_gldm_DependenceVariance_h2",
    "IC_original_glrlm_LongRunLowGrayLevelEmphasis_h2",
    "VMI_DL_91",
    "PEI_DL_11",
    "IC_original_glszm_GrayLevelNonUniformityNormalized_h1",
    "VMI_DL_248",
    "IC_original_glcm_ClusterShade_h1",
    "IC_DL_64",
    "IC_DL_286",
    "VMI_DL_176",
    "VMI_original_firstorder_Skewness_h1",
    "Zeff_DL_132",
    "Persistence_Mean_h1",
    "Zeff_DL_4",
    "IC_DL_197",
    "VMI_original_glszm_HighGrayLevelZoneEmphasis_h1",
    "Zeff_DL_19",
    "Differentiation"
]


# =========================================================
# 3. Default values
# =========================================================
# 所有特征默认值均设置为 0
default_values = [0] * len(feature_label)


# =========================================================
# 4. Streamlit page
# =========================================================
st.title("Web Predictor for GR to nCRT in Patients with LARC")
st.sidebar.header("Input Features")


# =========================================================
# 5. Input features
# =========================================================
inputs = {}

for i, feature in enumerate(feature_label):

    default_val = default_values[i]

    if feature == "Differentiation":

        options = ["Well/Moderate", "Poor"]

        # 默认值 0 对应 Well/Moderate
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
            value=0.0,
            format="%.8f"
        )


# =========================================================
# 6. Encode categorical variable
# =========================================================
diff_map = {
    "Well/Moderate": 0,
    "Poor": 1
}

inputs["Differentiation"] = diff_map[inputs["Differentiation"]]


# =========================================================
# 7. Convert input to DataFrame
# =========================================================
input_df = pd.DataFrame(
    [inputs],
    columns=feature_label
)

# 确保所有数据均为数值型
input_df = input_df.astype(float)


# 可选：在页面显示输入数据
with st.expander("View Input Data"):
    st.dataframe(input_df)


# =========================================================
# 8. Prediction and SHAP
# =========================================================
if st.sidebar.button("Predict"):

    try:

        # -------------------------------------------------
        # 8-1. Check model
        # -------------------------------------------------
        if not hasattr(model_rf, "predict_proba"):
            raise TypeError(
                "当前加载的模型不支持 predict_proba，"
                "请确认 RF.pkl 是 RandomForestClassifier 分类模型。"
            )

        # 检查模型特征数量
        if hasattr(model_rf, "n_features_in_"):
            if model_rf.n_features_in_ != input_df.shape[1]:
                raise ValueError(
                    f"模型需要 {model_rf.n_features_in_} 个特征，"
                    f"但当前输入了 {input_df.shape[1]} 个特征。"
                )

        # -------------------------------------------------
        # 8-2. RF prediction
        # -------------------------------------------------
        prediction_probabilities = model_rf.predict_proba(input_df)

        # 获取类别 1 的概率
        prediction_prob = prediction_probabilities[0, 1]

        threshold = 0.5
        prediction_class = int(prediction_prob >= threshold)

        # 获取模型类别
        if hasattr(model_rf, "classes_"):
            predicted_label_encoded = model_rf.classes_[prediction_class]
        else:
            predicted_label_encoded = prediction_class

        # -------------------------------------------------
        # 8-3. Display prediction
        # -------------------------------------------------
        st.subheader("Prediction Result")

        if prediction_class == 1:
            prediction_text = "GR to nCRT"
            result_color = "red"
        else:
            prediction_text = "Non-GR to nCRT"
            result_color = "blue"

        st.markdown(
            f"""
            <div style="font-size:24px;">
                Predicted probability of <b>GR to nCRT</b>:
                <span style="color:{result_color}; font-size:32px;">
                    {prediction_prob:.8f}
                </span>
            </div>

            <div style="font-size:24px; margin-top:10px;">
                Predicted class:
                <span style="color:{result_color}; font-size:28px;">
                    {prediction_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 显示两个类别概率
        probability_df = pd.DataFrame({
            "Class": ["Class 0", "Class 1"],
            "Probability": [
                prediction_probabilities[0, 0],
                prediction_probabilities[0, 1]
            ]
        })

        st.dataframe(probability_df)


        # =================================================
        # 9. SHAP explanation for RF
        # =================================================
        st.subheader("SHAP Force Plot")

        explainer = shap.TreeExplainer(model_rf)
        shap_values_all = explainer.shap_values(input_df)

        # -------------------------------------------------
        # 9-1. Extract SHAP values of class 1
        # -------------------------------------------------
        if isinstance(shap_values_all, list):

            # 旧版本 SHAP 通常返回：
            # [class_0_shap_values, class_1_shap_values]
            shap_values_class1 = shap_values_all[1]

        elif isinstance(shap_values_all, np.ndarray):

            if shap_values_all.ndim == 3:

                # 新版本可能返回：
                # (样本数, 特征数, 类别数)
                shap_values_class1 = shap_values_all[:, :, 1]

            elif shap_values_all.ndim == 2:

                shap_values_class1 = shap_values_all

            else:
                raise ValueError(
                    f"Unexpected SHAP values shape: "
                    f"{shap_values_all.shape}"
                )

        else:
            raise TypeError("Unsupported SHAP values type.")

        shap_values_class1 = np.asarray(shap_values_class1)


        # -------------------------------------------------
        # 9-2. Extract expected value of class 1
        # -------------------------------------------------
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


        # -------------------------------------------------
        # 9-3. Draw SHAP force plot
        # -------------------------------------------------
        plt.figure(figsize=(16, 4))

        shap.force_plot(
            expected_value_class1,
            shap_values_class1[0, :],
            input_df.iloc[0, :],
            feature_names=feature_label,
            contribution_threshold=0.135,
            matplotlib=True,
            show=False
        )

        plt.tight_layout()

        shap_image_path = "shap_force_plot_RF.png"

        plt.savefig(
            shap_image_path,
            bbox_inches="tight",
            dpi=150
        )

        plt.close()

        st.image(
            shap_image_path,
            caption="SHAP explanation for Class 1",
            use_container_width=True
        )


    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
