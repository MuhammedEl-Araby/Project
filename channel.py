import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm
import base64


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dynamic Pricing Simulator",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


img = get_base64_image("gettyimages-1395219224.jpg")


page_bg = f"""
<style>

[data-testid="stAppViewContainer"] {{
background-image:
linear-gradient(rgba(0,0,0,0.55),
rgba(0,0,0,0.55)),
url("data:image/jpg;base64,{img}");

background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
background: rgba(0,0,0,0.35);
}}

h1, h2, h3, h4, h5, h6, p, label, div {{
color: white;
}}

@keyframes pulse {{
0% {{ transform: scale(1); }}
50% {{ transform: scale(1.02); }}
100% {{ transform: scale(1); }}
}}

.stAlert {{
animation: pulse 2s infinite;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "Navigation",
    [
        "Main Simulator",
        "How To Use",
        "AI & Model Details"
    ],
    horizontal=True
)


# =========================================================
# HOW TO USE PAGE
# =========================================================

if page == "How To Use":

    st.title("How To Use The Website")

    st.markdown("""
    ## Website Purpose

    This platform simulates smart electrical dynamic pricing systems.

    The system compares different pricing strategies during high electrical load periods.

    ---

    ## Steps To Use

    1. Enter household information
    2. AI predicts your baseline consumption
    3. Open scenarios tabs
    4. Compare bills and energy usage
    5. Observe fairness and comfort effects

    ---

    ## Hints

    - More ACs increase baseline heavily
    - Large houses consume more energy
    - Heavy machines increase peak load
    - Staying below baseline reduces penalties
    - Automation can reduce electricity bills

    ---

    ## Scenarios

    ### Scenario 1
    Same reduction rule for everyone.

    ### Scenario 2
    Personalized baseline using AI.

    ### Scenario 3
    Pay extra for comfort during peak hours.
    """)

    st.info("Use navigation buttons above to return to simulator.")

    st.stop()


# =========================================================
# AI DETAILS PAGE
# =========================================================

if page == "AI & Model Details":

    st.title("AI Model & Dataset Information")

    st.markdown("""
    ## How The Dataset Was Generated

    The dataset is synthetic.

    Random household features were generated such as:
    - Lamps
    - ACs
    - Washing machine
    - Heavy machines
    - Occupants
    - House size

    Then a mathematical energy formula generated realistic baseline consumption.

    Gaussian noise was added to simulate real-world randomness.

    ---

    ## Machine Learning Model

    ### Random Forest Regressor

    Random Forest:
    - Builds many decision trees
    - Trains every tree on random samples
    - Combines all outputs together

    This improves:
    - Stability
    - Accuracy
    - Noise robustness

    ---

    ## Training Pipeline

    1. Generate synthetic dataset
    2. Split training/testing data
    3. Train Random Forest
    4. Predict unseen data
    5. Calculate MAE and R²

    ---

    ## Why Random Forest?

    Advantages:
    - Handles nonlinear energy behavior
    - Good with noisy datasets
    - Fast and stable
    - Excellent for regression tasks

    ---

    ## Future Improvements

    Future real systems may include:
    - Smart meter integration
    - IoT sensors
    - Deep learning forecasting
    - Reinforcement learning
    - Real-time load balancing
    - Grid optimization AI
    """)

    st.info("Use navigation buttons above to return to simulator.")

    st.stop()


# =========================================================
# CONSTANTS
# =========================================================

BASE_RATE = 0.25
PEAK_RATE = 0.80
PENALTY_RATE = 1.20
DISCOUNT_RATE = 0.15


# =========================================================
# DATA GENERATION
# =========================================================

def generate_training_data(n=1000):

    np.random.seed(42)

    lamps = np.random.randint(1, 15, n)
    acs = np.random.randint(0, 6, n)
    washing = np.random.randint(0, 2, n)
    heavy_machines = np.random.randint(0, 5, n)
    occupants = np.random.randint(1, 8, n)
    house_size = np.random.randint(50, 300, n)

    baseline = (
        0.25 * lamps +
        1.1 * acs +
        1.0 * washing +
        1.4 * heavy_machines +
        0.35 * occupants +
        0.01 * house_size +
        np.random.normal(0, 0.4, n)
    )

    baseline = np.clip(baseline, 0.5, None)

    df = pd.DataFrame({
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy_machines,
        "occupants": occupants,
        "house_size": house_size,
        "historical_baseline_kwh": baseline
    })

    return df


# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_model():

    df = generate_training_data()

    X = df.drop(columns=["historical_baseline_kwh"])
    y = df["historical_baseline_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics, df


# =========================================================
# SCENARIO 1
# =========================================================

def calculate_scenario_1(name, baseline, actual_usage):

    allowed_usage = baseline * 0.30

    if actual_usage <= allowed_usage:

        bill = actual_usage * BASE_RATE
        status = "Reduced enough"

    else:

        normal_part = allowed_usage * BASE_RATE
        penalty_part = (actual_usage - allowed_usage) * PENALTY_RATE

        bill = normal_part + penalty_part

        status = "Penalty applied"

    return {
        "Person": name,
        "Baseline kWh": baseline,
        "Actual Usage kWh": actual_usage,
        "Allowed After 70% Reduction": allowed_usage,
        "Bill": bill,
        "Status": status
    }


# =========================================================
# SCENARIO 2
# =========================================================

def calculate_scenario_2(name, baseline, actual_usage, stayed_below_peak):

    if actual_usage <= baseline:

        bill = actual_usage * BASE_RATE
        status = "Normal rate"

    else:

        normal_part = baseline * BASE_RATE
        extra_part = (actual_usage - baseline) * PENALTY_RATE

        bill = normal_part + extra_part

        status = "Only extra usage penalized"

    if stayed_below_peak:

        discount = bill * DISCOUNT_RATE
        bill_after_discount = bill - discount

    else:

        discount = 0
        bill_after_discount = bill

    return {
        "Person": name,
        "Personal Baseline kWh": baseline,
        "Actual Usage kWh": actual_usage,
        "Bill Before Discount": bill,
        "Discount": discount,
        "Final Bill": bill_after_discount,
        "Status": status
    }


# =========================================================
# SCENARIO 3
# =========================================================

def calculate_scenario_3(baseline, actual_usage, response_mode):

    above_baseline = max(actual_usage - baseline, 0)

    if response_mode == "Notify only":

        final_usage = actual_usage
        comfort = 100

    elif response_mode == "Auto shed non-critical loads":

        final_usage = baseline
        comfort = 75

    else:

        final_usage = actual_usage
        comfort = 100

    normal_usage = min(final_usage, baseline)
    premium_usage = max(final_usage - baseline, 0)

    bill = normal_usage * BASE_RATE + premium_usage * PEAK_RATE

    return final_usage, bill, comfort, premium_usage


# =========================================================
# INPUTS
# =========================================================

def household_input(
    title,
    default_lamps,
    default_acs,
    default_washing,
    default_heavy,
    default_occupants,
    default_size
):

    st.subheader(title)

    lamps = st.number_input(
        f"{title} - Lamps",
        value=default_lamps,
        step=1
    )

    acs = st.number_input(
        f"{title} - ACs",
        value=default_acs,
        step=1
    )

    washing = st.number_input(
        f"{title} - Washing Machine",
        value=default_washing,
        step=1
    )

    heavy = st.number_input(
        f"{title} - Heavy Machines",
        value=default_heavy,
        step=1
    )

    occupants = st.number_input(
        f"{title} - Occupants",
        value=default_occupants,
        step=1
    )

    size = st.number_input(
        f"{title} - House Size m²",
        value=default_size,
        step=1
    )

    return pd.DataFrame([{
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy,
        "occupants": occupants,
        "house_size": size
    }])


# =========================================================
# LOAD MODEL
# =========================================================

model, metrics, training_df = train_model()


# =========================================================
# MAIN TITLE
# =========================================================

st.title("Dynamic Pricing Simulator for Electrical Distribution")

st.warning(
    "⚠ Peak Event Active Now: Electricity demand is currently high between 5:00 PM and 7:00 PM. Premium pricing may apply."
)

st.markdown("""
This app simulates three dynamic pricing scenarios:

1. Same 70% reduction rule for everyone
2. Personal historical baseline pricing
3. Pay-to-play comfort pricing
""")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.image("Alex.jpg")

    st.header("Model Performance")

    st.metric(
        "MAE ( Mean Absolute Error )",
        f"{metrics['MAE']:.2f} kWh"
    )

    st.metric(
        "Model Score",
        f"{metrics['R2']:.2f}"
    )

    st.divider()

    st.header("Electricity Prices")

    st.write(f"Normal rate: **{BASE_RATE} EGP/kWh**")
    st.write(f"Peak rate: **{PEAK_RATE} EGP/kWh**")
    st.write(f"Penalty rate: **{PENALTY_RATE} EGP/kWh**")
    st.write(f"Loyalty discount: **{int(DISCOUNT_RATE * 100)}%**")

    st.image("Dr.jpg")

    st.header("Dynamic Pricing Simulator")

    st.write("Supervised by : Dr. Alaa Hamam")


# =========================================================
# HOUSEHOLDS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    person_a = household_input(
        "Person A: Low Baseline Home",
        default_lamps=3,
        default_acs=0,
        default_washing=0,
        default_heavy=0,
        default_occupants=1,
        default_size=60
    )

with col2:

    person_b = household_input(
        "Person B: Heavy Usage Home",
        default_lamps=10,
        default_acs=4,
        default_washing=1,
        default_heavy=3,
        default_occupants=5,
        default_size=220
    )


# =========================================================
# DYNAMIC AI PREDICTIONS
# =========================================================

baseline_a = float(
    model.predict(
        person_a.astype(float)
    )[0]
)

baseline_b = float(
    model.predict(
        person_b.astype(float)
    )[0]
)


# =========================================================
# METRICS
# =========================================================

st.divider()

m1, m2 = st.columns(2)

m1.metric(
    "Predicted Baseline - Person A",
    f"{baseline_a:.2f} kWh"
)

m2.metric(
    "Predicted Baseline - Person B",
    f"{baseline_b:.2f} kWh"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Scenario 1",
    "Scenario 2",
    "Scenario 3",
    "ML Dataset"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("Scenario 1: Same 70% Reduction Rule")

    st.markdown("""
    During high district load, every household must reduce consumption by 70%.
    Otherwise the extra usage is charged at penalty rate.
    """)

    actual_a = st.number_input(
        "Person A actual usage during peak event",
        value=float(baseline_a),
        step=0.1
    )

    actual_b = st.number_input(
        "Person B actual usage during peak event",
        value=float(baseline_b),
        step=0.1
    )

    result_a = calculate_scenario_1(
        "Person A",
        baseline_a,
        actual_a
    )

    result_b = calculate_scenario_1(
        "Person B",
        baseline_b,
        actual_b
    )

    result_df = pd.DataFrame([result_a, result_b])

    st.dataframe(result_df, use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result_df["Person"],
        y=result_df["Baseline kWh"],
        mode='lines+markers',
        name='Baseline',
        line=dict(width=4)
    ))

    fig.add_trace(go.Scatter(
        x=result_df["Person"],
        y=result_df["Allowed After 70% Reduction"],
        mode='lines+markers',
        name='Allowed Usage',
        line=dict(width=4)
    ))

    fig.add_trace(go.Scatter(
        x=result_df["Person"],
        y=result_df["Actual Usage kWh"],
        mode='lines+markers',
        name='Actual Usage',
        line=dict(width=4)
    ))

    fig.update_layout(
        title="Scenario 1",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=24),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "This scenario is unfair because low baseline users suffer more."
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header("Scenario 2: Personalized Historical Baseline")

    actual_a2 = st.number_input(
        "Person A usage",
        value=float(baseline_a),
        step=0.1,
        key="a2"
    )

    actual_b2 = st.number_input(
        "Person B usage",
        value=float(baseline_b),
        step=0.1,
        key="b2"
    )

    discount_a = st.checkbox(
        "Person A stayed below baseline",
        value=True
    )

    discount_b = st.checkbox(
        "Person B stayed below baseline",
        value=False
    )

    result_a2 = calculate_scenario_2(
        "Person A",
        baseline_a,
        actual_a2,
        discount_a
    )

    result_b2 = calculate_scenario_2(
        "Person B",
        baseline_b,
        actual_b2,
        discount_b
    )

    result_df2 = pd.DataFrame([result_a2, result_b2])

    st.dataframe(result_df2, use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=result_df2["Person"],
        y=result_df2["Personal Baseline kWh"],
        mode='lines+markers',
        name='Baseline',
        line=dict(width=4)
    ))

    fig2.add_trace(go.Scatter(
        x=result_df2["Person"],
        y=result_df2["Actual Usage kWh"],
        mode='lines+markers',
        name='Actual Usage',
        line=dict(width=4)
    ))

    fig2.update_layout(
        title="Scenario 2",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=24),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig_bill = px.bar(
        result_df2,
        x="Person",
        y=["Bill Before Discount", "Discount", "Final Bill"],
        barmode="group",
        title="Bill Comparison"
    )

    fig_bill.update_layout(
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=24)
    )

    st.plotly_chart(fig_bill, use_container_width=True)

    st.success(
        "This scenario is fairer because each user is compared to their own baseline."
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header("Scenario 3: Pay-to-Play")

    selected_person = st.radio(
        "Choose household",
        ["Person A", "Person B"]
    )

    if selected_person == "Person A":

        baseline = baseline_a
        default_usage = baseline_a + 1

    else:

        baseline = baseline_b
        default_usage = baseline_b + 3

    usage = st.number_input(
        "Requested usage during peak hours",
        value=float(default_usage),
        step=0.1
    )

    mode = st.selectbox(
        "Automation Response",
        [
            "Notify only",
            "Auto shed non-critical loads",
            "Take no action and pay premium"
        ]
    )

    final_usage, bill, comfort, premium_usage = calculate_scenario_3(
        baseline,
        usage,
        mode
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Baseline", f"{baseline:.2f} kWh")
    c2.metric("Final Usage", f"{final_usage:.2f} kWh")
    c3.metric("Premium Usage", f"{premium_usage:.2f} kWh")
    c4.metric("Comfort", f"{comfort}%")

    st.metric("Final Bill", f"{bill:.2f} EGP")

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=["Requested", "Final", "Baseline"],
        y=[usage, final_usage, baseline],
        mode='lines+markers',
        line=dict(width=4)
    ))

    fig3.update_layout(
        title="Pay-to-Play Usage",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=24),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.header("Synthetic ML Dataset")

    st.dataframe(
        training_df.head(100),
        use_container_width=True
    )

    mean_usage = training_df["historical_baseline_kwh"].mean()

    std_usage = training_df["historical_baseline_kwh"].std()

    x = np.linspace(
        training_df["historical_baseline_kwh"].min(),
        training_df["historical_baseline_kwh"].max(),
        500
    )

    y = norm.pdf(x, mean_usage, std_usage)

    fig_data = go.Figure()

    fig_data.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal Distribution',
        line=dict(width=5)
    ))

    fig_data.update_layout(
        title="Baseline Consumption Distribution",
        xaxis_title="Baseline kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=24),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    st.plotly_chart(fig_data, use_container_width=True)

    # =========================================================
    # USER POSITION ON BELL CURVE
    # =========================================================
    
    st.header("Your Position Relative To AI Baseline Distribution")
    
    selected_person_curve = st.radio(
        "Choose Person For Distribution Analysis",
        [
            "Person A",
            "Person B"
        ],
        horizontal=True
    )
    
    if selected_person_curve == "Person A":
        current_baseline = baseline_a
    else:
        current_baseline = baseline_b
    
    # Calculate Z-score
    z_score = (
        current_baseline - mean_usage
    ) / std_usage
    
    # Create bell curve figure
    fig_curve = go.Figure()
    
    # Normal distribution curve
    fig_curve.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Population Distribution',
        line=dict(width=5)
    ))
    
    # User position marker
    user_y = norm.pdf(
        current_baseline,
        mean_usage,
        std_usage
    )
    
    fig_curve.add_trace(go.Scatter(
        x=[current_baseline],
        y=[user_y],
        mode='markers+text',
        name='Your Home',
        marker=dict(
            size=18,
            color='red'
        ),
        text=["Your Baseline"],
        textposition="top center"
    ))
    
    # Mean line
    fig_curve.add_vline(
        x=mean_usage,
        line_width=3,
        line_dash="dash",
        line_color="yellow"
    )
    
    fig_curve.update_layout(
        title="Your Location On AI Baseline Bell Curve",
        xaxis_title="Baseline Consumption kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=24),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    st.plotly_chart(
        fig_curve,
        use_container_width=True
    )
    
    # =========================================================
    # ANALYSIS ENGINE
    # =========================================================
    
    st.subheader("AI Analysis")
    
    if z_score < -1.5:
    
        st.success(
            "Your consumption is VERY LOW compared to the average population. "
            "This usually represents highly efficient homes or minimal appliance usage."
        )
    
    elif z_score < -0.5:
    
        st.info(
            "Your consumption is BELOW average. "
            "Your household uses less electricity than most users."
        )
    
    elif z_score <= 0.5:
    
        st.warning(
            "Your consumption is CLOSE TO THE AVERAGE population baseline."
        )
    
    elif z_score <= 1.5:
    
        st.warning(
            "Your consumption is ABOVE average. "
            "This indicates higher appliance usage or larger household demand."
        )
    
    else:
    
        st.error(
            "Your consumption is EXTREMELY HIGH compared to most homes. "
            "This may heavily contribute to peak load stress during high demand events."
        )
    
    # =========================================================
    # PERCENTILE INFORMATION
    # =========================================================
    
    percentile = (
        norm.cdf(z_score) * 100
    )
    
    st.metric(
        "Population Percentile",
        f"{percentile:.2f}%"
    )
    
    st.markdown(f"""
    ### Interpretation
    
    - Mean Population Baseline: **{mean_usage:.2f} kWh**
    - Your Predicted Baseline: **{current_baseline:.2f} kWh**
    - Standard Deviation Position (Z-Score): **{z_score:.2f}**
    - Percentile Rank: **{percentile:.2f}%**
    
    This means your home consumes more electricity than approximately
    **{percentile:.2f}%** of the simulated population.
    """)
