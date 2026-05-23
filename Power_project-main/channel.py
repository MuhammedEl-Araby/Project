import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm
import base64
import os


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SCADA Dynamic Pricing & Smart Meter Simulator",
    page_icon="⚡",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "SCADA Dynamic Pricing & Smart Meter Simulator"
    }
)


# =========================================================
# SAFE IMAGE LOADER
# =========================================================

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None


bg_img = get_base64_image("gettyimages-1395219224.jpg")


# =========================================================
# STYLE
# =========================================================

if bg_img:
    background_css = f"""
    background-image:
    linear-gradient(rgba(0,0,0,0.62), rgba(0,0,0,0.62)),
    url("data:image/jpg;base64,{bg_img}");
    """
else:
    background_css = """
    background: linear-gradient(135deg, #050505, #111827, #1e293b);
    """

page_bg = f"""
<style>

[data-testid="stAppViewContainer"] {{
{background_css}
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
background: rgba(0,0,0,0.45);
}}

h1, h2, h3, h4, h5, h6, p, label, div, span {{
color: white;
}}

.stMetric {{
background: rgba(255,255,255,0.08);
border-radius: 16px;
padding: 12px;
border: 1px solid rgba(255,255,255,0.15);
}}

[data-testid="stDataFrame"] {{
background: rgba(255,255,255,0.05);
border-radius: 14px;
}}

@keyframes pulse {{
0% {{ transform: scale(1); }}
50% {{ transform: scale(1.01); }}
100% {{ transform: scale(1); }}
}}

.stAlert {{
animation: pulse 2.5s infinite;
}}

.scada-card {{
background: rgba(0,0,0,0.42);
border: 1px solid rgba(255,255,255,0.18);
border-radius: 18px;
padding: 18px;
margin-bottom: 12px;
}}

.big-status {{
font-size: 28px;
font-weight: 800;
}}

.ac-plan-container {{
position: relative;
width: 100%;
max-width: 1200px;
margin: auto;
border-radius: 18px;
overflow: hidden;
border: 2px solid rgba(255,255,255,0.25);
box-shadow: 0 0 25px rgba(0,0,0,0.55);
}}

.ac-plan-container img {{
width: 100%;
display: block;
}}

.ac-marker {{
position: absolute;
transform: translate(-50%, -50%);
font-weight: 900;
font-size: 44px;
line-height: 1;
text-shadow: 0 0 8px black, 0 0 14px black;
z-index: 10;
}}

.ac-working {{
color: #00ff66;
}}

.ac-disconnected {{
color: #ff2222;
}}

.ac-label {{
position: absolute;
transform: translate(-50%, -50%);
font-weight: 900;
font-size: 18px;
background: rgba(0,0,0,0.65);
color: yellow;
border: 1px solid rgba(255,255,255,0.35);
border-radius: 10px;
padding: 2px 8px;
z-index: 11;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


# =========================================================
# SESSION STATE DEFAULTS
# =========================================================

if "appliance_config" not in st.session_state:
    st.session_state.appliance_config = pd.DataFrame([
        {
            "Appliance": "Lights",
            "Quantity": 10,
            "Power per Unit kW": 0.02,
            "Connected": True,
            "Disconnectable": False,
            "Critical": True,
            "User Priority": 999,
            "Company Priority": 999,
            "Preserve Minimum Units": 10
        },
        {
            "Appliance": "Power Sockets",
            "Quantity": 8,
            "Power per Unit kW": 0.15,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 1,
            "Company Priority": 2,
            "Preserve Minimum Units": 0
        },
        {
            "Appliance": "Water Heater",
            "Quantity": 1,
            "Power per Unit kW": 2.0,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 2,
            "Company Priority": 4,
            "Preserve Minimum Units": 0
        },
        {
            "Appliance": "Hand Dryer",
            "Quantity": 1,
            "Power per Unit kW": 1.8,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 3,
            "Company Priority": 5,
            "Preserve Minimum Units": 0
        },
        {
            "Appliance": "Washing Machine",
            "Quantity": 1,
            "Power per Unit kW": 1.0,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 4,
            "Company Priority": 1,
            "Preserve Minimum Units": 0
        },
        {
            "Appliance": "ACs",
            "Quantity": 6,
            "Power per Unit kW": 1.3,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 5,
            "Company Priority": 3,
            "Preserve Minimum Units": 1
        },
        {
            "Appliance": "Heavy Machines",
            "Quantity": 2,
            "Power per Unit kW": 1.6,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 6,
            "Company Priority": 6,
            "Preserve Minimum Units": 0
        }
    ])

if "selected_user_policy" not in st.session_state:
    st.session_state.selected_user_policy = "Manual User Priority"

if "refuse_disconnect" not in st.session_state:
    st.session_state.refuse_disconnect = False

if "climate_mode" not in st.session_state:
    st.session_state.climate_mode = "Cooling Mode"

if "mandatory_reduction_percent" not in st.session_state:
    st.session_state.mandatory_reduction_percent = 20

if "voluntary_reduction_percent" not in st.session_state:
    st.session_state.voluntary_reduction_percent = 35

if "ac_unit_status" not in st.session_state:
    st.session_state.ac_unit_status = {
        "AC 1": True,
        "AC 2": True,
        "AC 3": True,
        "AC 4": True,
        "AC 5": True,
        "AC 6": True
    }

if "ac_marker_positions" not in st.session_state:
    # Percent positions on image.
    # You can adjust these from Real Life Simulation page to match ACs.png exactly.
    st.session_state.ac_marker_positions = pd.DataFrame([
        {"AC": "AC 1", "X %": 18, "Y %": 25},
        {"AC": "AC 2", "X %": 38, "Y %": 25},
        {"AC": "AC 3", "X %": 62, "Y %": 25},
        {"AC": "AC 4", "X %": 82, "Y %": 25},
        {"AC": "AC 5", "X %": 35, "Y %": 70},
        {"AC": "AC 6", "X %": 68, "Y %": 70},
    ])


# =========================================================
# CONSTANTS
# =========================================================

BASE_RATE = 0.25
PEAK_RATE = 0.80
PENALTY_RATE = 1.20
PREMIUM_PRESERVATION_RATE = 1.60
DISCOUNT_RATE = 0.15
BONUS_RATE = 0.10


# =========================================================
# DATA GENERATION
# =========================================================

def generate_training_data(n=5000):
    """
    Synthetic training data with wider ranges.
    This prevents the model from being limited to only small household values.
    """
    np.random.seed(42)

    lamps = np.random.randint(1, 301, n)
    acs = np.random.randint(0, 51, n)
    washing = np.random.randint(0, 31, n)
    heavy_machines = np.random.randint(0, 51, n)
    occupants = np.random.randint(1, 101, n)
    house_size = np.random.randint(20, 5001, n)

    baseline = (
        0.22 * lamps +
        1.15 * acs +
        0.90 * washing +
        1.55 * heavy_machines +
        0.32 * occupants +
        0.010 * house_size +
        np.random.normal(0, 0.45, n)
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
# MODEL TRAINING WITHOUT STREAMLIT CACHE
# =========================================================

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
        n_estimators=300,
        max_depth=14,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics, df


if "trained_model_bundle" not in st.session_state:
    st.session_state.trained_model_bundle = train_model()

model, metrics, training_df = st.session_state.trained_model_bundle

mean_usage = training_df["historical_baseline_kwh"].mean()
std_usage = training_df["historical_baseline_kwh"].std()


# =========================================================
# UNLIMITED BASELINE PREDICTION
# =========================================================

def predict_baseline_unlimited(model, input_df):
    """
    Random Forest models are not naturally good at extrapolating far outside
    their training range. This function allows any user input while still
    adding a logical correction if the input exceeds the synthetic dataset range.
    """
    row = input_df.iloc[0].astype(float)

    capped = row.copy()
    caps = {
        "lamps": 300,
        "acs": 50,
        "washing_machine": 30,
        "heavy_machines": 50,
        "occupants": 100,
        "house_size": 5000
    }

    for col, cap in caps.items():
        capped[col] = min(capped[col], cap)

    capped_df = pd.DataFrame([capped])
    base_prediction = float(model.predict(capped_df)[0])

    extra = 0
    extra += max(row["lamps"] - caps["lamps"], 0) * 0.22
    extra += max(row["acs"] - caps["acs"], 0) * 1.15
    extra += max(row["washing_machine"] - caps["washing_machine"], 0) * 0.90
    extra += max(row["heavy_machines"] - caps["heavy_machines"], 0) * 1.55
    extra += max(row["occupants"] - caps["occupants"], 0) * 0.32
    extra += max(row["house_size"] - caps["house_size"], 0) * 0.010

    return base_prediction + extra


# =========================================================
# HOUSEHOLD INPUT
# =========================================================

def adaptive_capacity_limits(size):
    """
    Practical warning limits only.
    They do NOT block calculation.
    """
    washing_limit = max(2, int(size / 70))
    heavy_limit = max(1, int(size / 90))
    ac_limit = max(1, int(size / 35))
    lamp_limit = max(10, int(size / 4))

    return washing_limit, heavy_limit, ac_limit, lamp_limit


def household_input(title, default_lamps, default_acs, default_washing,
                    default_heavy, default_occupants, default_size):

    st.subheader(title)

    lamps = st.number_input(
        f"{title} - Lamps",
        min_value=0,
        value=default_lamps,
        step=1
    )

    acs = st.number_input(
        f"{title} - ACs",
        min_value=0,
        value=default_acs,
        step=1
    )

    washing = st.number_input(
        f"{title} - Washing Machines",
        min_value=0,
        value=default_washing,
        step=1
    )

    heavy = st.number_input(
        f"{title} - Heavy Machines",
        min_value=0,
        value=default_heavy,
        step=1
    )

    occupants = st.number_input(
        f"{title} - Occupants",
        min_value=0,
        value=default_occupants,
        step=1
    )

    size = st.number_input(
        f"{title} - House Size m²",
        min_value=1,
        value=default_size,
        step=1
    )

    washing_limit, heavy_limit, ac_limit, lamp_limit = adaptive_capacity_limits(size)

    logical_warnings = []

    if washing > washing_limit:
        logical_warnings.append(
            f"Washing machines are high for {size} m². Suggested practical limit is about "
            f"{washing_limit}, but your entered value will still be fully calculated."
        )

    if heavy > heavy_limit:
        logical_warnings.append(
            f"Heavy machines are high for {size} m². Suggested practical limit is about "
            f"{heavy_limit}, but your entered value will still be fully calculated."
        )

    if acs > ac_limit:
        logical_warnings.append(
            f"AC count is high for {size} m². Suggested practical limit is about "
            f"{ac_limit}, but your entered value will still be fully calculated."
        )

    if lamps > lamp_limit:
        logical_warnings.append(
            f"Lamps are high for {size} m². Suggested practical limit is about "
            f"{lamp_limit}, but your entered value will still be fully calculated."
        )

    if occupants > max(2, int(size / 10)):
        logical_warnings.append(
            "Occupants are unusually high for the house size, but your entered value will still be calculated."
        )

    for warning in logical_warnings:
        st.warning(warning)

    return pd.DataFrame([{
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy,
        "occupants": occupants,
        "house_size": size
    }])


# =========================================================
# LOAD CALCULATION ENGINE
# =========================================================

def normalize_appliance_df(appliance_df):
    df = appliance_df.copy()

    numeric_cols = [
        "Quantity",
        "Power per Unit kW",
        "User Priority",
        "Company Priority",
        "Preserve Minimum Units"
    ]

    bool_cols = [
        "Connected",
        "Disconnectable",
        "Critical"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)

    return df


def calculate_current_connected_load(appliance_df):
    df = normalize_appliance_df(appliance_df)

    df["Connected Load kW"] = np.where(
        df["Connected"],
        df["Quantity"] * df["Power per Unit kW"],
        0
    )

    return df


def smart_meter_shed_load(
    appliance_df,
    requested_reduction_percent,
    policy_mode,
    refuse_disconnect,
    climate_mode,
    mandatory_minimum_percent,
    user_failed_to_respond,
    enforcement_enabled
):
    df = calculate_current_connected_load(appliance_df)

    original_load = df["Connected Load kW"].sum()

    if original_load <= 0:
        return df, 0, 0, 0, "No active load"

    requested_reduction_kw = original_load * requested_reduction_percent / 100
    mandatory_reduction_kw = original_load * mandatory_minimum_percent / 100

    if refuse_disconnect:
        if enforcement_enabled and user_failed_to_respond:
            target_reduction_kw = mandatory_reduction_kw
            active_policy = "Company Emergency Enforcement"
            enforcement_status = "User refused or ignored request. Mandatory reduction was enforced."
        else:
            target_reduction_kw = 0
            active_policy = "User Refused Disconnection"
            enforcement_status = "No load was disconnected. Premium pricing applied."
    else:
        target_reduction_kw = requested_reduction_kw
        active_policy = policy_mode
        enforcement_status = "User smart meter priority was applied."

    if target_reduction_kw <= 0:
        df["Disconnected Units"] = 0
        df["Remaining Units"] = df["Quantity"]
        df["Shed kW"] = 0
        final_load = original_load
        achieved_reduction_percent = 0
        return df, original_load, final_load, achieved_reduction_percent, enforcement_status

    df["Disconnected Units"] = 0
    df["Remaining Units"] = df["Quantity"]
    df["Shed kW"] = 0.0

    if active_policy == "Company Emergency Enforcement":
        priority_col = "Company Priority"
    elif policy_mode == "Company Priority":
        priority_col = "Company Priority"
    else:
        priority_col = "User Priority"

    shed_so_far = 0.0

    candidates = df[
        (df["Connected"] == True) &
        (df["Disconnectable"] == True) &
        (df["Critical"] == False)
    ].copy()

    candidates = candidates.sort_values(by=priority_col, ascending=True)

    for idx, row in candidates.iterrows():
        if shed_so_far >= target_reduction_kw:
            break

        quantity = float(row["Quantity"])
        power = float(row["Power per Unit kW"])
        appliance = row["Appliance"]
        preserve_minimum = float(row["Preserve Minimum Units"])

        if appliance == "ACs" and climate_mode in ["Cooling Mode", "Heating Mode"]:
            preserve_minimum = max(preserve_minimum, 1)

        max_disconnectable_units = max(quantity - preserve_minimum, 0)

        if max_disconnectable_units <= 0 or power <= 0:
            continue

        remaining_needed_kw = target_reduction_kw - shed_so_far
        units_needed = np.ceil(remaining_needed_kw / power)

        units_to_disconnect = min(max_disconnectable_units, units_needed)

        shed_kw = units_to_disconnect * power

        df.loc[idx, "Disconnected Units"] = units_to_disconnect
        df.loc[idx, "Remaining Units"] = quantity - units_to_disconnect
        df.loc[idx, "Shed kW"] = shed_kw

        shed_so_far += shed_kw

    final_load = max(original_load - shed_so_far, 0)
    achieved_reduction_percent = (shed_so_far / original_load) * 100

    return df, original_load, final_load, achieved_reduction_percent, enforcement_status


# =========================================================
# BILLING ENGINE
# =========================================================

def billing_engine(
    baseline,
    original_usage,
    final_usage,
    mean_usage,
    grid_stress,
    new_company_growth_mode,
    refused_disconnect,
    achieved_reduction_percent,
    mandatory_reduction_percent
):
    premium_usage = max(final_usage - baseline, 0)
    normal_usage = min(final_usage, baseline)

    bill = normal_usage * BASE_RATE

    bonus = 0
    penalty = 0
    premium_charge = 0
    discount = 0
    status = []

    if new_company_growth_mode and not grid_stress:
        if final_usage > mean_usage:
            bonus = bill * BONUS_RATE
            bill = bill - bonus
            status.append("Growth bonus applied because company wants to increase average demand.")
        else:
            status.append("Normal bill. Usage is still below desired growth level.")

    if grid_stress:
        if premium_usage > 0:
            premium_charge = premium_usage * PREMIUM_PRESERVATION_RATE
            bill += premium_charge
            status.append("Premium Load Preservation Pricing applied for usage above historical baseline.")

        if achieved_reduction_percent < mandatory_reduction_percent:
            penalty = final_usage * 0.20
            bill += penalty
            status.append("Mandatory reduction target was not achieved. Grid stress penalty applied.")

        if achieved_reduction_percent >= mandatory_reduction_percent:
            discount = bill * DISCOUNT_RATE
            bill -= discount
            status.append("Grid support discount applied because mandatory reduction was achieved.")

    if refused_disconnect and grid_stress:
        status.append("User refused smart meter disconnection. Premium convenience pricing applied.")

    return {
        "Normal Usage kWh": normal_usage,
        "Premium Usage kWh": premium_usage,
        "Premium Charge": premium_charge,
        "Penalty": penalty,
        "Bonus": bonus,
        "Discount": discount,
        "Final Bill": bill,
        "Status": " | ".join(status)
    }


# =========================================================
# AC PLAN HTML
# =========================================================

def render_ac_plan():
    ac_img = get_base64_image("ACs.png")

    if ac_img is None:
        st.error(
            "ACs.png was not found. Put your HVAC plan image in the same folder as this Streamlit app "
            "with the exact name: ACs.png"
        )
        return

    marker_html = ""

    positions = st.session_state.ac_marker_positions.copy()

    for _, row in positions.iterrows():
        ac_name = row["AC"]
        x = float(row["X %"])
        y = float(row["Y %"])

        is_working = st.session_state.ac_unit_status.get(ac_name, True)

        symbol = "O" if is_working else "X"
        css_class = "ac-working" if is_working else "ac-disconnected"

        marker_html += f"""
        <div class="ac-marker {css_class}" style="left:{x}%; top:{y}%;">{symbol}</div>
        <div class="ac-label" style="left:{x}%; top:calc({y}% + 38px);">{ac_name}</div>
        """

    html = f"""
    <div class="ac-plan-container">
        <img src="data:image/png;base64,{ac_img}">
        {marker_html}
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def sync_ac_plan_to_appliance_config():
    working_count = sum(1 for v in st.session_state.ac_unit_status.values() if v)

    df = st.session_state.appliance_config.copy()

    ac_mask = df["Appliance"] == "ACs"

    if ac_mask.any():
        df.loc[ac_mask, "Quantity"] = working_count
        df.loc[ac_mask, "Connected"] = working_count > 0
    else:
        df.loc[len(df)] = {
            "Appliance": "ACs",
            "Quantity": working_count,
            "Power per Unit kW": 1.3,
            "Connected": working_count > 0,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 5,
            "Company Priority": 3,
            "Preserve Minimum Units": 1
        }

    st.session_state.appliance_config = df


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "Navigation",
    [
        "SCADA Control Center",
        "Smart Meter Override Page",
        "Real Life Simulation",
        "How To Use",
        "AI & Model Details"
    ],
    horizontal=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    if os.path.exists("Alex.jpg"):
        st.image("Alex.jpg")

    st.header("Model Performance")

    st.metric(
        "MAE",
        f"{metrics['MAE']:.2f} kWh"
    )

    st.metric(
        "Model Score R²",
        f"{metrics['R2']:.2f}"
    )

    st.divider()

    st.header("Tariff Catalogue")

    st.write(f"Normal rate: **{BASE_RATE} EGP/kWh**")
    st.write(f"Peak rate: **{PEAK_RATE} EGP/kWh**")
    st.write(f"Penalty rate: **{PENALTY_RATE} EGP/kWh**")
    st.write(f"Premium preservation rate: **{PREMIUM_PRESERVATION_RATE} EGP/kWh**")
    st.write(f"Grid support discount: **{int(DISCOUNT_RATE * 100)}%**")
    st.write(f"Growth bonus: **{int(BONUS_RATE * 100)}%**")

    st.divider()

    if os.path.exists("Dr.jpg"):
        st.image("Dr.jpg")

    st.header("Dynamic Pricing Simulator")
    st.write("Supervised by: Dr. Alaa Hamam")


# =========================================================
# HOW TO USE PAGE
# =========================================================

if page == "How To Use":

    st.title("How To Use The Website - Manual Catalogue")

    st.markdown("""
    ## 1. Purpose of the Simulator

    This website is a research and educational simulation for a smart electrical distribution system.

    It combines:

    - Historical baseline estimation
    - Dynamic pricing
    - Peak event notification
    - Smart meter override
    - User priority control
    - Company emergency priority control
    - Load shedding
    - Premium uninterrupted consumption pricing
    - Real-life HVAC image simulation

    The website does not connect to a real smart meter or real SCADA system.

    ---

    ## 2. Main Pages

    ### SCADA Control Center

    This is the main dashboard.

    Use it to:

    1. Activate or deactivate grid stress.
    2. Activate or deactivate a peak event.
    3. Choose if the company is in growth mode.
    4. Enable emergency enforcement.
    5. Set the requested reduction percentage.
    6. Set the mandatory minimum reduction percentage.
    7. Enter household data.
    8. Select Person A or Person B.
    9. See final usage, load shedding, bill, and decision.

    ---

    ### Smart Meter Override Page

    This page controls the smart meter logic.

    You can decide:

    - Which appliances are connected.
    - Which appliances are disconnectable.
    - Which appliances are critical.
    - Which loads disconnect first.
    - Whether user priority or company priority is used.
    - Whether the user refuses all disconnection.
    - Whether the building is in cooling or heating condition.

    Important:

    - Lower priority number means the load disconnects earlier.
    - Critical loads will not disconnect.
    - Lights are protected by default.
    - If the user refuses disconnection, premium pricing is applied.
    - If the grid is really stressed and the user ignores the request, emergency enforcement can still happen.

    ---

    ### Real Life Simulation

    This page is for the HVAC plan image.

    Put your image in the project folder with this exact name:

    **ACs.png**

    The page will display the plan and place:

    - **O** on working AC units
    - **X** on disconnected AC units

    You can manually change each AC:

    - AC 1 working or disconnected
    - AC 2 working or disconnected
    - AC 3 working or disconnected
    - AC 4 working or disconnected
    - AC 5 working or disconnected
    - AC 6 working or disconnected

    If you change AC 3 from disconnected to working, the X will disappear and O will appear automatically.

    You can also edit the marker positions using X % and Y % if the symbols are not exactly on top of the ACs in your plan.

    ---

    ### AI & Model Details

    This page explains:

    - What the synthetic dataset means
    - Why random numbers are used
    - What Gaussian randomness means
    - What R² means
    - What MAE means
    - How the historical baseline estimation works

    ---

    ## 3. How to Use the Smart Meter Override

    Example:

    You want the system to disconnect power sockets first, then heater, then hand dryer.

    Set priorities like this:

    - Power Sockets: User Priority = 1
    - Water Heater: User Priority = 2
    - Hand Dryer: User Priority = 3
    - Washing Machine: User Priority = 4
    - ACs: User Priority = 5
    - Heavy Machines: User Priority = 6

    Keep lights protected by setting:

    - Disconnectable = False
    - Critical = True

    ---

    ## 4. How to Use Real Life AC Simulation

    1. Go to **Real Life Simulation**.
    2. Make sure **ACs.png** exists in the app folder.
    3. Use the AC checkboxes to select which ACs are working.
    4. Disconnected ACs will show red X.
    5. Working ACs will show green O.
    6. If symbol positions are wrong, open "Marker Position Calibration" and adjust X % / Y %.
    7. Press "Sync AC Plan With Smart Meter" if you want the working AC count to affect the SCADA calculation.

    ---

    ## 5. Important Notes

    - This is a simulator, not a real protection relay.
    - Premium payment can preserve comfort only if the physical grid can handle it.
    - If the line is in real danger, the company can enforce minimum reduction.
    - Warnings about too many machines do not block calculation.
    - You are free to enter any number.
    """)

    st.stop()


# =========================================================
# AI DETAILS PAGE
# =========================================================

if page == "AI & Model Details":

    st.title("AI Model & Dataset Information")

    st.markdown("""
    ## Dataset

    The dataset is synthetic, meaning it is generated inside the program for simulation and education.

    It contains different building and household cases using these features:

    - Lamps
    - ACs
    - Washing machines
    - Heavy machines
    - Occupants
    - House size

    The model estimates a historical baseline in kWh.

    ---

    ## Why Random Numbers Are Used

    Real consumption is never perfectly fixed.

    Two homes with the same number of ACs and lamps may not consume exactly the same energy because of:

    - Different usage habits
    - Different weather
    - Different insulation
    - Different appliance efficiency
    - Different operating hours

    So the dataset adds random variation to make the simulation more realistic.

    ---

    ## Gaussian Randomness

    The model adds a small random noise using a normal distribution.

    This is also called Gaussian randomness.

    A normal distribution means:

    - Most random errors are small
    - Very large random errors are rare
    - The distribution forms a bell curve

    This helps the synthetic data look closer to real-world measurement variation.

    ---

    ## MAE Meaning

    MAE means **Mean Absolute Error**.

    It tells you the average prediction error in kWh.

    Example:

    If MAE = 1.50 kWh, then on average the model prediction is about 1.50 kWh away from the target value.

    Lower MAE is better.

    ---

    ## R² Meaning

    R² means **Coefficient of Determination**.

    It shows how much of the data pattern is explained by the model.

    - R² close to 1.00 means strong prediction performance.
    - R² close to 0.00 means weak prediction performance.
    - Negative R² means the model is worse than using the average.

    ---

    ## Important Note

    The baseline is called an estimated historical baseline because, in a real company,
    this value would come from confidential customer history and smart meter records.

    In this simulator, we estimate it using the synthetic model.
    """)

    st.subheader("Synthetic Dataset Preview")
    st.dataframe(training_df.head(150), use_container_width=True)

    x = np.linspace(
        training_df["historical_baseline_kwh"].min(),
        training_df["historical_baseline_kwh"].max(),
        600
    )

    y = norm.pdf(x, mean_usage, std_usage)

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=training_df["historical_baseline_kwh"],
        histnorm="probability density",
        name="Synthetic Baseline Histogram",
        opacity=0.55
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Normal Distribution Bell Curve",
        line=dict(width=5, color="cyan")
    ))

    fig.add_vline(
        x=mean_usage,
        line_width=4,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Mean"
    )

    fig.update_layout(
        title="Baseline Consumption Bell Curve",
        xaxis_title="Historical Baseline kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=620
    )

    st.plotly_chart(fig, use_container_width=True)

    st.stop()


# =========================================================
# REAL LIFE SIMULATION PAGE
# =========================================================

if page == "Real Life Simulation":

    st.title("Real Life Simulation - HVAC Plan")

    st.warning(
        "This page is a trial SCADA-style visual simulation. "
        "It edits the visual status on the HVAC plan while the app is running."
    )

    st.markdown("""
    ## HVAC Plan Logic

    The image file must be named:

    **ACs.png**

    Symbols:

    - Green **O** = AC is working
    - Red **X** = AC is disconnected

    If you change an AC again, the old X or O is automatically replaced.
    """)

    st.divider()

    left, right = st.columns([1, 2])

    with left:
        st.subheader("AC Unit Override")

        for ac_name in list(st.session_state.ac_unit_status.keys()):
            st.session_state.ac_unit_status[ac_name] = st.checkbox(
                f"{ac_name} Working",
                value=st.session_state.ac_unit_status[ac_name]
            )

        working_acs = [k for k, v in st.session_state.ac_unit_status.items() if v]
        disconnected_acs = [k for k, v in st.session_state.ac_unit_status.items() if not v]

        st.metric("Working ACs", len(working_acs))
        st.metric("Disconnected ACs", len(disconnected_acs))

        st.write("Working:", ", ".join(working_acs) if working_acs else "None")
        st.write("Disconnected:", ", ".join(disconnected_acs) if disconnected_acs else "None")

        if st.button("Sync AC Plan With Smart Meter"):
            sync_ac_plan_to_appliance_config()
            st.success("AC working count has been synced with Smart Meter Override configuration.")

    with right:
        st.subheader("Live HVAC Plan")
        render_ac_plan()

    st.divider()

    with st.expander("Marker Position Calibration"):
        st.info(
            "If O/X symbols are not exactly on the AC symbols in the image, "
            "adjust X % and Y %. These values are percentages of the image width and height."
        )

        edited_positions = st.data_editor(
            st.session_state.ac_marker_positions,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "AC": st.column_config.TextColumn("AC"),
                "X %": st.column_config.NumberColumn("X %", min_value=0, max_value=100, step=1),
                "Y %": st.column_config.NumberColumn("Y %", min_value=0, max_value=100, step=1),
            }
        )

        st.session_state.ac_marker_positions = edited_positions

    st.stop()


# =========================================================
# SMART METER OVERRIDE PAGE
# =========================================================

if page == "Smart Meter Override Page":

    st.title("Smart Meter Override Page")
    st.warning(
        "This page lets the client manually override the smart meter priority rules. "
        "The user's choices affect the SCADA Control Center simulation."
    )

    st.markdown("""
    ## Manual Control Philosophy

    You can decide:

    - Which appliance is connected or disconnected
    - Which appliance disconnects first during peak events
    - Which appliance must never disconnect
    - Whether the company priority or user priority should be used
    - Whether you refuse all disconnections and pay premium pricing
    - Whether AC operation is cooling mode or heating mode
    """)

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.session_state.selected_user_policy = st.selectbox(
            "Priority Control Mode",
            [
                "Manual User Priority",
                "Company Priority"
            ],
            index=0 if st.session_state.selected_user_policy == "Manual User Priority" else 1
        )

    with col_b:
        st.session_state.refuse_disconnect = st.checkbox(
            "I do not want to disconnect anything",
            value=st.session_state.refuse_disconnect
        )

    with col_c:
        st.session_state.climate_mode = st.selectbox(
            "Climate / AC Operating Mode",
            [
                "Normal Mode",
                "Cooling Mode",
                "Heating Mode"
            ],
            index=["Normal Mode", "Cooling Mode", "Heating Mode"].index(st.session_state.climate_mode)
        )

    st.info(
        "In Cooling or Heating Mode, the system tries to preserve at least one AC if possible."
    )

    st.subheader("Edit Appliance Priority and Connection Status")

    edited_df = st.data_editor(
        st.session_state.appliance_config,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Appliance": st.column_config.TextColumn("Appliance"),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1),
            "Power per Unit kW": st.column_config.NumberColumn("Power per Unit kW", min_value=0.0, step=0.01),
            "Connected": st.column_config.CheckboxColumn("Connected"),
            "Disconnectable": st.column_config.CheckboxColumn("Disconnectable"),
            "Critical": st.column_config.CheckboxColumn("Critical"),
            "User Priority": st.column_config.NumberColumn("User Priority", step=1),
            "Company Priority": st.column_config.NumberColumn("Company Priority", step=1),
            "Preserve Minimum Units": st.column_config.NumberColumn("Preserve Minimum Units", min_value=0, step=1)
        }
    )

    st.session_state.appliance_config = edited_df

    st.divider()

    st.subheader("Current Smart Meter Load Summary")

    load_df = calculate_current_connected_load(st.session_state.appliance_config)

    total_connected_kw = load_df["Connected Load kW"].sum()

    st.metric("Total Connected Load", f"{total_connected_kw:.2f} kW")

    st.dataframe(load_df, use_container_width=True)

    fig = px.bar(
        load_df,
        x="Appliance",
        y="Connected Load kW",
        color="Connected",
        title="Current Connected Load by Appliance",
        text_auto=".2f"
    )

    fig.update_layout(
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=25),
        height=560
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Your smart meter override configuration has been saved in the session. "
        "Go to SCADA Control Center to see its effect."
    )

    st.stop()


# =========================================================
# SCADA CONTROL CENTER PAGE
# =========================================================

st.title("SCADA Dynamic Pricing & Smart Meter Control Center")

st.warning(
    "⚠ PEAK EVENT NOTIFICATION: High electrical demand is active. "
    "The smart meter may request load reduction. Premium Load Preservation Pricing may apply."
)

st.markdown("""
<div class="scada-card">
<div class="big-status">Integrated Operating Scenario</div>
This dashboard combines historical baseline estimation, dynamic tariffs, manual smart meter override,
priority-based load shedding, mandatory grid protection, and premium uninterrupted consumption pricing.
</div>
""", unsafe_allow_html=True)


# =========================================================
# GRID EVENT CONTROL
# =========================================================

st.header("Grid Event & Company Control Panel")

g1, g2, g3, g4 = st.columns(4)

with g1:
    grid_stress = st.checkbox(
        "Real Stress On Line",
        value=True
    )

with g2:
    peak_event = st.checkbox(
        "Peak Usage Event",
        value=True
    )

with g3:
    new_company_growth_mode = st.checkbox(
        "New Company Growth Mode",
        value=False
    )

with g4:
    enforcement_enabled = st.checkbox(
        "Emergency Enforcement Enabled",
        value=True
    )

st.subheader("SCADA Reduction Commands")

c1, c2, c3 = st.columns(3)

with c1:
    voluntary_reduction_percent = st.slider(
        "Requested Reduction Signal From Company (%)",
        min_value=0,
        max_value=90,
        value=st.session_state.voluntary_reduction_percent,
        step=1
    )

with c2:
    mandatory_reduction_percent = st.slider(
        "Mandatory Minimum Reduction During Real Stress (%)",
        min_value=0,
        max_value=60,
        value=st.session_state.mandatory_reduction_percent,
        step=1
    )

with c3:
    response_deadline_minutes = st.number_input(
        "Response Deadline Before Enforcement (minutes)",
        min_value=0,
        value=60,
        step=1
    )

st.session_state.voluntary_reduction_percent = voluntary_reduction_percent
st.session_state.mandatory_reduction_percent = mandatory_reduction_percent

user_failed_to_respond = st.checkbox(
    "User ignored repeated company requests until deadline expired",
    value=False
)

if grid_stress and peak_event:
    st.error(
        f"SCADA Alert: Line stress is real. User must reduce at least "
        f"{mandatory_reduction_percent}% even if premium payment is accepted."
    )
else:
    st.success(
        "Grid is stable. Pricing and bonus modes can operate without mandatory protection."
    )


# =========================================================
# HOUSEHOLD INPUTS
# =========================================================

st.divider()
st.header("Household Historical Baseline Inputs")

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

baseline_a = predict_baseline_unlimited(model, person_a)
baseline_b = predict_baseline_unlimited(model, person_b)


# =========================================================
# BASELINE METRICS
# =========================================================

st.divider()
st.header("Estimated Historical Baselines")

m1, m2, m3 = st.columns(3)

m1.metric(
    "Estimated Historical Baseline - Person A",
    f"{baseline_a:.2f} kWh"
)

m2.metric(
    "Estimated Historical Baseline - Person B",
    f"{baseline_b:.2f} kWh"
)

m3.metric(
    "Population Mean Baseline",
    f"{mean_usage:.2f} kWh"
)


# =========================================================
# PERSON SELECTION
# =========================================================

st.divider()
st.header("Client Selection")

selected_person = st.radio(
    "Choose Client / Smart Meter",
    ["Person A", "Person B"],
    horizontal=True
)

if selected_person == "Person A":
    selected_baseline = baseline_a
else:
    selected_baseline = baseline_b


# =========================================================
# USAGE COMMAND
# =========================================================

st.subheader("Client Usage During Peak Event")

requested_usage = st.number_input(
    "Requested / Original Usage During Peak Event kWh",
    min_value=0.0,
    value=float(selected_baseline + 2),
    step=0.1
)

st.info(
    "This value represents the user's intended usage before smart meter shedding or company enforcement."
)


# =========================================================
# SMART METER SIMULATION
# =========================================================

policy_mode = st.session_state.selected_user_policy
refuse_disconnect = st.session_state.refuse_disconnect
climate_mode = st.session_state.climate_mode

shed_df, original_load_kw, final_load_kw, achieved_reduction_percent, enforcement_status = smart_meter_shed_load(
    appliance_df=st.session_state.appliance_config,
    requested_reduction_percent=voluntary_reduction_percent,
    policy_mode=policy_mode,
    refuse_disconnect=refuse_disconnect,
    climate_mode=climate_mode,
    mandatory_minimum_percent=mandatory_reduction_percent,
    user_failed_to_respond=user_failed_to_respond,
    enforcement_enabled=enforcement_enabled and grid_stress and peak_event
)

if original_load_kw > 0:
    usage_ratio = final_load_kw / original_load_kw
else:
    usage_ratio = 1

final_usage = requested_usage * usage_ratio

billing = billing_engine(
    baseline=selected_baseline,
    original_usage=requested_usage,
    final_usage=final_usage,
    mean_usage=mean_usage,
    grid_stress=grid_stress and peak_event,
    new_company_growth_mode=new_company_growth_mode,
    refused_disconnect=refuse_disconnect,
    achieved_reduction_percent=achieved_reduction_percent,
    mandatory_reduction_percent=mandatory_reduction_percent
)


# =========================================================
# MAIN SCADA METRICS
# =========================================================

st.divider()
st.header("SCADA Live Status")

s1, s2, s3, s4, s5 = st.columns(5)

s1.metric("Original Connected Load", f"{original_load_kw:.2f} kW")
s2.metric("Final Connected Load", f"{final_load_kw:.2f} kW")
s3.metric("Achieved Reduction", f"{achieved_reduction_percent:.2f}%")
s4.metric("Final Usage", f"{final_usage:.2f} kWh")
s5.metric("Final Bill", f"{billing['Final Bill']:.2f} EGP")

if achieved_reduction_percent < mandatory_reduction_percent and grid_stress and peak_event:
    st.error(
        "Grid Protection Warning: The mandatory reduction target was not achieved."
    )
else:
    st.success(
        "Grid Protection Status: Reduction condition is acceptable."
    )

st.info(enforcement_status)


# =========================================================
# PREMIUM PRICING MESSAGE
# =========================================================

st.subheader("Intro Message Catalogue")

if grid_stress and peak_event:
    st.warning(
        "Peak Event Active: The system requested load reduction because the issue is physical line stress, "
        "not only electricity price. Premium payment may preserve comfort, but mandatory reduction can still be enforced."
    )

if refuse_disconnect:
    st.error(
        "Client Policy: The user selected no disconnection. The system will apply premium convenience pricing. "
        "If the deadline expires during real stress, emergency enforcement may override the refusal."
    )

if new_company_growth_mode and not grid_stress:
    st.success(
        "Growth Mode Active: The company wants to increase average consumption. "
        "Users above the average may receive a bonus instead of penalty."
    )

st.markdown("""
### Tariff Name

**Premium Load Preservation Pricing under Dynamic Tariffs:  
An Uninterrupted Consumption Pay-for-Convenience Model for Peak Load Retention**
""")


# =========================================================
# BILLING DETAILS
# =========================================================

st.divider()
st.header("Billing & Condition Results")

billing_df = pd.DataFrame([{
    "Client": selected_person,
    "Historical Baseline kWh": selected_baseline,
    "Requested Usage kWh": requested_usage,
    "Final Usage kWh": final_usage,
    "Normal Usage kWh": billing["Normal Usage kWh"],
    "Premium Usage kWh": billing["Premium Usage kWh"],
    "Premium Charge EGP": billing["Premium Charge"],
    "Penalty EGP": billing["Penalty"],
    "Bonus EGP": billing["Bonus"],
    "Discount EGP": billing["Discount"],
    "Final Bill EGP": billing["Final Bill"],
    "Condition Status": billing["Status"]
}])

st.dataframe(billing_df, use_container_width=True)


# =========================================================
# SMART METER LOAD TABLE
# =========================================================

st.divider()
st.header("Smart Meter Load Shedding Result")

st.dataframe(shed_df, use_container_width=True)


# =========================================================
# GRAPHS
# =========================================================

st.divider()
st.header("SCADA Visual Analytics")

tab1, tab2, tab3, tab4 = st.tabs([
    "Load Shedding Graph",
    "Usage & Billing Graph",
    "Baseline Bell Curve",
    "Priority Comparison"
])


with tab1:
    fig_load = go.Figure()

    fig_load.add_trace(go.Bar(
        x=shed_df["Appliance"],
        y=shed_df["Connected Load kW"],
        name="Original Connected Load",
        marker_color="deepskyblue",
        text=shed_df["Connected Load kW"].round(2),
        textposition="auto"
    ))

    fig_load.add_trace(go.Bar(
        x=shed_df["Appliance"],
        y=shed_df["Shed kW"],
        name="Disconnected / Shed Load",
        marker_color="red",
        text=shed_df["Shed kW"].round(2),
        textposition="auto"
    ))

    fig_load.update_layout(
        title="Original Load vs Disconnected Load",
        xaxis_title="Appliance",
        yaxis_title="kW",
        barmode="group",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_load, use_container_width=True)


with tab2:
    usage_fig = go.Figure()

    usage_fig.add_trace(go.Bar(
        x=["Historical Baseline", "Requested Usage", "Final Usage"],
        y=[selected_baseline, requested_usage, final_usage],
        marker_color=["yellow", "orange", "lime"],
        text=[
            round(selected_baseline, 2),
            round(requested_usage, 2),
            round(final_usage, 2)
        ],
        textposition="auto"
    ))

    usage_fig.update_layout(
        title="Historical Baseline vs Requested vs Final Usage",
        xaxis_title="Condition",
        yaxis_title="kWh",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=600
    )

    st.plotly_chart(usage_fig, use_container_width=True)

    bill_parts = pd.DataFrame({
        "Component": [
            "Premium Charge",
            "Penalty",
            "Bonus",
            "Discount",
            "Final Bill"
        ],
        "EGP": [
            billing["Premium Charge"],
            billing["Penalty"],
            billing["Bonus"],
            billing["Discount"],
            billing["Final Bill"]
        ]
    })

    fig_bill = px.bar(
        bill_parts,
        x="Component",
        y="EGP",
        title="Bill Components",
        text_auto=".2f"
    )

    fig_bill.update_layout(
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=600
    )

    st.plotly_chart(fig_bill, use_container_width=True)


with tab3:
    x = np.linspace(
        training_df["historical_baseline_kwh"].min(),
        training_df["historical_baseline_kwh"].max(),
        600
    )

    y = norm.pdf(x, mean_usage, std_usage)

    user_y = norm.pdf(selected_baseline, mean_usage, std_usage)

    z_score = (selected_baseline - mean_usage) / std_usage
    percentile = norm.cdf(z_score) * 100

    fig_curve = go.Figure()

    fig_curve.add_trace(go.Histogram(
        x=training_df["historical_baseline_kwh"],
        histnorm="probability density",
        name="Population Histogram",
        opacity=0.45,
        marker_color="gray"
    ))

    fig_curve.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Normal Bell Curve",
        line=dict(width=5, color="cyan")
    ))

    fig_curve.add_trace(go.Scatter(
        x=[selected_baseline],
        y=[user_y],
        mode="markers+text",
        name=f"{selected_person} Position",
        marker=dict(size=20, color="red"),
        text=[f"{selected_person}"],
        textposition="top center"
    ))

    fig_curve.add_vline(
        x=mean_usage,
        line_width=4,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Mean"
    )

    fig_curve.update_layout(
        title="Client Position on Historical Baseline Bell Curve",
        xaxis_title="Baseline Consumption kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Z-Score", f"{z_score:.2f}")
    cc2.metric("Percentile", f"{percentile:.2f}%")
    cc3.metric("Population Mean", f"{mean_usage:.2f} kWh")

    if z_score < -1.5:
        st.success("Consumption is very low compared to the simulated population.")
    elif z_score < -0.5:
        st.info("Consumption is below average.")
    elif z_score <= 0.5:
        st.warning("Consumption is close to average.")
    elif z_score <= 1.5:
        st.warning("Consumption is above average.")
    else:
        st.error("Consumption is extremely high compared to most simulated homes.")


with tab4:
    priority_df = st.session_state.appliance_config.copy()

    fig_priority = go.Figure()

    fig_priority.add_trace(go.Bar(
        x=priority_df["Appliance"],
        y=priority_df["User Priority"],
        name="User Priority",
        marker_color="lime",
        text=priority_df["User Priority"],
        textposition="auto"
    ))

    fig_priority.add_trace(go.Bar(
        x=priority_df["Appliance"],
        y=priority_df["Company Priority"],
        name="Company Priority",
        marker_color="orange",
        text=priority_df["Company Priority"],
        textposition="auto"
    ))

    fig_priority.update_layout(
        title="User Priority vs Company Priority",
        xaxis_title="Appliance",
        yaxis_title="Priority Number: Lower Means Disconnect First",
        barmode="group",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_priority, use_container_width=True)

    st.info(
        "Lower priority number means the appliance disconnects earlier. "
        "Critical appliances such as lights can be protected by setting Disconnectable = False."
    )


# =========================================================
# FINAL SYSTEM SUMMARY
# =========================================================

st.divider()
st.header("Final SCADA Decision Summary")

if grid_stress and peak_event and achieved_reduction_percent < mandatory_reduction_percent:
    st.error(
        "Final Decision: The user did not satisfy the minimum physical grid protection requirement. "
        "The company may apply enforcement, restriction, or blocking logic in this simulation."
    )
elif refuse_disconnect and grid_stress and peak_event:
    st.warning(
        "Final Decision: User preserved comfort and refused disconnection. "
        "Premium pricing is applied, but grid protection may still override if stress continues."
    )
elif achieved_reduction_percent >= mandatory_reduction_percent and grid_stress and peak_event:
    st.success(
        "Final Decision: User supported the grid by reducing enough load. "
        "Discount or positive reliability score can be applied."
    )
else:
    st.info(
        "Final Decision: Normal dynamic pricing mode. No emergency grid protection action required."
    )
