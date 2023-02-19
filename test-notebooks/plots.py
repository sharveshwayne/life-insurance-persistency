import streamlit as st
import pandas as pd
import plotly.express as px
from millify import millify
import numpy as np

INDEX = "policy_number"
DATE_COLS = ["proposal_received_date", "policy_issue_date", "agent_dob", "agent_doj"]
NA_VALUES = ["", "NA", "N/A", "NULL", "null", "?", "*", "#N/A", "#VALUE!", "   "]
DTYPE_DICT = {"zipcode": "str", "agent_code": "str"}
df = pd.read_csv(
    "master_data_final2.csv",
    index_col=INDEX,
    na_values=NA_VALUES,
    parse_dates=DATE_COLS,
    dayfirst=True,
    dtype=DTYPE_DICT,
)

column_mapping = {
    "Education": "education",
    "Occupation": "occupation",
    "Smoker": "smoker",
    "Marital Status": "marital_status",
    "Gender": "owner_gender",
    "Experience": "experience",
    "Income": "income",
    "Age": "owner_age",
    "Lapse": "lapse",
    "Credit Score": "credit_score",
    "Policy Term": "policy_term",
    "Payment Frequency": "payment_freq",
    "Annual Premium": "annual_premium",
    "Sum Insured": "sum_insured",
    "Agent Age": "agent_age",
    "Agent Tenure (days)": "agent_tenure_days",
    "Persistency": "agent_persistency",
    "Last 6 months policies": "last_6_month_submissions",
    "Average Premium": "average_premium",
    "Is Reinstated?": "is_reinstated",
    "Persistency before Reinstated": "prev_persistency",
    "Number Complaints": "num_complaints",
    "Agent Status": "agent_status",
    "Agent Education": "agent_education",
    "Target Completion Percentage": "target_completion_perc",
    "Number of Nominees": "num_nominee",
    "Underwent Medical Test?": "medical",
    "Number of Family Members": "family_member",
    "Number Existing Policies": "existing_num_policy",
    "Has Critical Health History?": "has_critical_health_history",
}

cust_cnt = len(df)
policy_cnt = len(df.index)
lapse_rate = round((df["lapse"].value_counts() / len(df))[1], 3)
num_agents = df["agent_code"].nunique()
tot_annual_prem = df["annual_premium"].sum()
tot_sum_assured = df["sum_insured"].sum()
avg_policy_tenure = round(df["policy_term"].mean())
avg_tkt_size = tot_annual_prem / cust_cnt
revenue_leakage = tot_annual_prem * lapse_rate

agg_df = (
    df.groupby("state")
    .agg(
        {
            "policy_issue_date": pd.Series.count,
            "annual_premium": (lambda x: np.round(pd.Series.mean(x), 1)),
            "lapse": np.sum,
            "agent_code": pd.Series.count,
        }
    )
    .reset_index()
)
agg_df = agg_df.rename(
    {
        "policy_issue_date": "Policy Count",
        "annual_premium": "Annual Premium",
        "lapse": "Lapse",
        "agent_code": "Agent Count",
    },
    axis=1,
)


def plot_map(df, col, color_scale="viridis_r"):
    fig = px.choropleth(
        df,
        locations="state",
        locationmode="USA-states",
        color=col,
        color_continuous_scale=color_scale,
        scope="usa",
    )

    fig.add_scattergeo(
        locations=df["state"], locationmode="USA-states", text=df["state"], mode="text"
    )
    return fig


def descriptive_polts(x_vars, group_vars, column_mapping_dict=column_mapping):
    x_axis = st.selectbox(label="View distribution of", options=x_vars, index=0)
    x_axis = column_mapping_dict[x_axis]

    group_var = st.selectbox(label="Drill-down by", options=group_vars, index=0)
    group_var = column_mapping_dict[group_var]

    IS_X_VAR_CAT = df[x_axis].dtype == "object" or (df[x_axis].nunique() < 20)
    IS_GRP_VAR_CAT = df[group_var].dtype == "object" or (df[group_var].nunique() < 20)

    if IS_X_VAR_CAT and IS_GRP_VAR_CAT:
        fig = px.histogram(
            df,
            x=x_axis,
            color=group_var,
            labels={value: key for key, value in column_mapping.items()},
        )
        #    category_orders={"education":['Lt High School', 'High School', 'Some College', 'Graduate', 'Post Graduate', 'Others']})
        fig.update_xaxes(type="category", categoryorder="category ascending")
        return fig

    elif not IS_X_VAR_CAT and IS_GRP_VAR_CAT:
        fig = px.box(df, x=x_axis, color=group_var)
        fig.update_xaxes(categoryorder="category ascending")
        return fig

    elif not IS_X_VAR_CAT and not IS_GRP_VAR_CAT:
        fig = px.scatter(df, x=x_axis, y=group_var)
        return fig

    else:
        return None


st.set_page_config(layout="wide")

col1, col2, col3, col4 = st.columns([4, 1, 1, 1])

with col1:
    st.subheader("Values across US States")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Policy Count", "Annual Premium", "Lapse Count", "Agent Count"]
    )

    with tab1:
        fig = plot_map(agg_df, "Policy Count")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_map(agg_df, "Annual Premium")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = plot_map(agg_df, "Lapse")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = plot_map(agg_df, "Agent Count")
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Key Indicators")
    st.metric("Customer Count", f"{cust_cnt:,}")
    st.metric("Policy Count", f"{policy_cnt:,}")
    st.metric("Lapse Rate", f"{lapse_rate*100}%")
    st.metric("Number of agents", f"{num_agents:,}")
    st.metric("Average Policy Tenure", f"{avg_policy_tenure} years")

with col4:
    st.subheader("⠀⠀⠀⠀⠀")
    st.metric("Total Annual Premium", f"${millify(tot_annual_prem, precision=1)}")
    st.metric("Total Sum Insured", f"${millify(tot_sum_assured, 1)}")
    st.metric("Average Ticket Size", f"${round(avg_tkt_size):,}")
    st.metric("Revenue Leakage", f"${millify(revenue_leakage,1)}")

st.subheader("Descriptive Plots")

tab1, tab2, tab3 = st.tabs(
    ["Customer Attributes", "Policy Attributes", "Agent Attributes"]
)

with tab1:
    fig = descriptive_polts(
        x_vars=[
            "Occupation",
            "Smoker",
            "Marital Status",
            "Gender",
            "Education",
            "Experience",
            "Income",
            "Age",
            "Credit Score",
            "Number of Nominees",
            "Underwent Medical Test?",
            "Number of Family Members",
            "Number Existing Policies",
            "Has Critical Health History?",
        ],
        group_vars=[
            "Lapse",
            "Occupation",
            "Smoker",
            "Marital Status",
            "Gender",
            "Experience",
            "Income",
            "Age",
            "Credit Score",
            "Number of Nominees",
            "Underwent Medical Test?",
            "Number of Family Members",
            "Number Existing Policies",
            "Has Critical Health History?",
        ],
    )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

with tab2:
    fig = descriptive_polts(
        x_vars=["Policy Term", "Payment Frequency", "Annual Premium", "Sum Insured"],
        group_vars=[
            "Lapse",
            "Policy Term",
            "Payment Frequency",
            "Annual Premium",
            "Sum Insured",
        ],
    )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

with tab3:
    fig = descriptive_polts(
        x_vars=[
            "Agent Age",
            "Agent Tenure (days)",
            "Persistency",
            "Last 6 months policies",
            "Average Premium",
            "Is Reinstated?",
            "Persistency before Reinstated",
            "Number Complaints",
            "Agent Status",
            "Agent Education",
            "Target Completion Percentage",
        ],
        group_vars=[
            "Lapse",
            "Agent Age",
            "Agent Tenure (days)",
            "Persistency",
            "Last 6 months policies",
            "Average Premium",
            "Is Reinstated?",
            "Persistency before Reinstated",
            "Number Complaints",
            "Agent Status",
            "Agent Education",
            "Target Completion Percentage",
        ],
    )

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.write("Invalid Combination")

###
