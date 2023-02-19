from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from utils import clean_data_batch
from utils import (
    load_model,
    create_final_input_batch,
    load_train_data_attribs,
    create_image_df,
    create_shap_plot,
    load_df_for_plots,
)
from millify import millify


st.subheader("Prediction Service for a Batch of Customers")
month_choices = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
month = st.selectbox("Select a month", month_choices)

# threshold = st.number_input('Lapse Probability Threshold', 0.0, 1.0, 0.5, step=0.01)
threshold = 0.5
classes = ["Not Lapsed", "Lapsed"]

df = load_df_for_plots()
df = df.drop(["lapse"], axis=1)


def chart(values, labels, title):
    fig, ax = plt.subplots()
    sns.barplot(x=labels, y=values, ax=ax)
    ax.set_ylabel(title)
    cur_values = ax.get_yticks()
    ax.set_yticklabels([millify(x, 2) for x in cur_values])
    percentages = [f"{int(round(x/sum(values),2)*100)}%" for x in values]
    for x, y, txt in zip(range(len(labels)), np.array(values) / 2, percentages):
        ax.text(x, y, txt)
    return fig


if month is not None:
    pred_month = month_choices[month_choices.index(month) - 1]
    result_df = (
        df.query("@df.policy_issue_date.dt.month_name() == @pred_month")
        .reset_index(drop=True)
        .copy()
    )
    # st.write(f"Policy taken on {pred_month}")
    xgboost_model, model_input_pipe = load_model()
    batch_df = create_final_input_batch(result_df).dropna()
    input_data = model_input_pipe.transform(batch_df.drop("customer_id", axis=1))

    predicted_prob = xgboost_model.predict_proba(input_data)[:, 1]
    batch_df["pred_prob"] = predicted_prob
    batch_df["pred_lapse"] = np.where(predicted_prob < threshold, 0, 1)

    col1, col2 = st.columns(2)
    with col1:
        # 1
        st.metric("Customer Count", len(predicted_prob))
        # 3
        lapse_counts = batch_df["pred_lapse"].value_counts().tolist()
        if len(predicted_prob) > 1:
            st.metric("Lapse Counts", f"{lapse_counts[0]} | {lapse_counts[1]}")
            fig = chart(lapse_counts, classes, "Lapse Count")
            st.pyplot(fig)
    with col2:
        # 2
        st.metric("Total Premium", f"${millify(batch_df.annual_premium.sum(), 2)}")
        # 4
        lapsed_premium = (
            batch_df.groupby("pred_lapse")
            .agg({"annual_premium": np.sum})["annual_premium"]
            .to_list()
        )
        if len(predicted_prob) > 1:
            st.metric(
                "Lapsed Premium",
                f"${millify(lapsed_premium[0],1)} | ${millify(lapsed_premium[1], 1)}",
            )
            fig = chart(lapsed_premium, classes, "Annual Premium ($)")
            st.pyplot(fig)
    # 5
    customer = st.selectbox(
        "Model Prediction Explanation : Select a customer from below",
        batch_df["customer_id"],
    )
    customer_datapoint = batch_df.query("@batch_df.customer_id == @customer")

    if customer_datapoint.pred_lapse.item() == 0:
        st.error("Prediction : Will Lapse")
    else:
        st.success("Prediction : Will Renew")

    feat_df, X_train_trf, explainer = load_train_data_attribs()
    customer_datapoint = clean_data_batch(customer_datapoint).dropna()
    transformed_datapoint = model_input_pipe.transform(
        customer_datapoint.drop("customer_id", axis=1)
    )
    plot_df = create_image_df(model_input_pipe, transformed_datapoint)
    plt.clf()
    fig = create_shap_plot(
        explainer,
        plot_df,
    )

    st.pyplot(fig)
    st.caption(
        "Shapley Waterfall chart for the customer explaining the factors impacting the decision"
    )
