import streamlit as st
from millify import millify
import numpy as np
import pandas as pd
from utils import (
    clean_data_batch,
    load_df_for_plots,
    load_model,
    create_final_input_batch,
)

st.set_page_config(layout="wide")

st.subheader(
    "*Definition* :",
)
st.markdown(
    "**Cases where the model is not very confident in it's prediction i.e. \
            the confidence in the prediction lies near the conversion boundary (0.35 - 0.65).\
            It would be beneficial for business to identify what factors are influntial\
            in determing a conversion.**"
)


# not ideal -> should be caching the results from batch prediction and only running
# prediction on uncached datapoints
threshold = 0.5
df = load_df_for_plots()
xgboost_model, model_input_pipe = load_model()
batch_df = create_final_input_batch(df).dropna()
input_data = model_input_pipe.transform(batch_df.drop("customer_id", axis=1))
predicted_prob = xgboost_model.predict_proba(input_data)[:, 1]
batch_df["pred_prob"] = predicted_prob
batch_df["pred_lapse"] = np.where(predicted_prob < threshold, 0, 1)

borderline_datapoints = batch_df.query("0.35 <= pred_prob <= 0.65")

st.write("----------")
border_choice = st.selectbox(
    "Choose a Borderline Case", borderline_datapoints["customer_id"].unique()
)
original_datapoint = (
    borderline_datapoints.query("customer_id == @border_choice")
    .iloc[0]
    .copy()
    .to_dict()
)
st.write("----------")

_, col1, col2, _ = st.columns([2, 3, 1, 2])
with col1:
    st.metric("Customer Count", len(borderline_datapoints))
with col2:
    st.metric(
        "Total Premium", f"${millify(borderline_datapoints.annual_premium.sum(), 2)}"
    )

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

col_1, col_2, col_3 = st.columns(3)

modified_datapoint = {}

with col_1:
    st.caption("Customer features")
    marital_options = batch_df["marital_status"].unique().tolist()
    smoker_options = batch_df["smoker"].unique().tolist()
    medical_options = batch_df["medical"].unique().tolist()
    education_options = batch_df["education"].unique().tolist()

    modified_datapoint["marital_status"] = st.selectbox(
        "Marital Status",
        marital_options,
        index=marital_options.index(original_datapoint["marital_status"]),
    )
    modified_datapoint["smoker"] = st.selectbox(
        "Smoker",
        smoker_options,
        index=smoker_options.index(original_datapoint["smoker"]),
    )
    modified_datapoint["medical"] = st.selectbox(
        "Underwent Medical Test?",
        medical_options,
        index=medical_options.index(original_datapoint["medical"]),
    )
    modified_datapoint["education"] = st.selectbox(
        "Education",
        education_options,
        index=education_options.index(original_datapoint["education"]),
    )

with col_2:
    st.caption("Policy features")
    payment_freq_options = batch_df["payment_freq"].unique().tolist()

    modified_datapoint["payment_freq"] = st.selectbox(
        "Payment Frequency",
        payment_freq_options,
        index=payment_freq_options.index(original_datapoint["payment_freq"]),
    )
    modified_datapoint["policy_term"] = st.select_slider(
        "Policy Term",
        range(1, batch_df["policy_term"].max()),
        value=original_datapoint["policy_term"],
    )
    modified_datapoint["time_to_issue"] = st.select_slider(
        "Time to issue the policy (months)",
        range(batch_df["time_to_issue"].max()),
        value=original_datapoint["time_to_issue"],
    )

with col_3:
    st.caption("Agent features")
    agent_education_options = batch_df["agent_education"].unique().tolist()
    contacted_options = batch_df["has_contacted_in_last_6_months"].unique().tolist()
    is_reinstated_options = batch_df["is_reinstated"].unique().tolist()
    modified_datapoint["agent_education"] = st.selectbox(
        "Agent Education",
        agent_education_options,
        index=agent_education_options.index(original_datapoint["agent_education"]),
    )
    modified_datapoint["has_contacted_in_last_6_months"] = int(
        st.selectbox(
            "Has contacted in last 6 months",
            contacted_options,
            index=contacted_options.index(
                original_datapoint["has_contacted_in_last_6_months"]
            ),
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    )
    modified_datapoint["is_reinstated"] = int(
        st.selectbox(
            "Is Reinstated?",
            is_reinstated_options,
            index=is_reinstated_options.index(original_datapoint["is_reinstated"]),
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
    )
    modified_datapoint["agent_persistency"] = st.slider(
        "Current Persistency",
        batch_df["agent_persistency"].min().item(),
        batch_df["agent_persistency"].max().item(),
        value=original_datapoint["agent_persistency"],
    )
    modified_datapoint["target_completion_perc"] = st.slider(
        "Target Completion Percentage",
        batch_df["target_completion_perc"].min().item(),
        batch_df["target_completion_perc"].max().item(),
        value=original_datapoint["target_completion_perc"],
    )


modified_datapoint_df = pd.DataFrame(
    {**original_datapoint, **modified_datapoint}, index=[0]
)
customer_datapoint = clean_data_batch(modified_datapoint_df).dropna()
transformed_customer_datapoint = model_input_pipe.transform(
    customer_datapoint.drop("customer_id", axis=1)
)
xgboost_model, model_input_pipe = load_model()

predicted_prob = xgboost_model.predict_proba(transformed_customer_datapoint)[:, 1]
prediction_lapse = (predicted_prob > threshold) * 1

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

col__1, _, col__3 = st.columns(3)
labels = ["Not Lapsed", "Lapsed"]
with col__1:
    st.metric("Original Prediction", labels[original_datapoint["pred_lapse"]])
    st.metric("Original Probability", np.round(original_datapoint["pred_prob"], 4))

with col__3:
    st.metric("New Prediction", labels[int(prediction_lapse)])
    st.metric(
        "New Probability",
        np.round(float(predicted_prob), 4),
        delta=round(float(predicted_prob) - original_datapoint["pred_prob"], 4),
    )
