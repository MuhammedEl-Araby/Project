import streamlit as st
import streamlit.components.v1 as components
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
    page_title="SCADA Dynamic Pricing Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# IMAGE LOADER
# =========================================================

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# =========================================================
# STYLE
# =========================================================

bg_img = get_base64_image("gettyimages-1395219224.jpg")

if bg_img:
    bg_css = f"""
    background-image:
    linear-gradient(rgba(0,0,0,0.62), rgba(0,0,0,0.62)),
    url("data:image/jpg;base64,{bg_img}");
    """
else:
    bg_css = """
    background: linear-gradient(135deg, #050505, #111827, #1e293b);
    """

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        {bg_css}
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

    .scada-card {{
        background: rgba(0,0,0,0.45);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 12px;
    }}

    .big-status {{
        font-size: 28px;
        font-weight: 800;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


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
# SESSION DEFAULTS
# =========================================================

if "selected_user_policy" not in st.session_state:
    st.session_state.selected_user_policy = "Manual User Priority"

if "refuse_disconnect" not in st.session_state:
    st.session_state.refuse_disconnect = False

if "climate_mode" not in st.session_state:
    st.session_state.climate_mode = "Cooling Mode - Hot Weather"

if "voluntary_reduction_percent" not in st.session_state:
    st.session_state.voluntary_reduction_percent = 35

if "mandatory_reduction_percent" not in st.session_state:
    st.session_state.mandatory_reduction_percent = 20


# =========================================================
# SMART METER DEFAULT APPLIANCE TABLE
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
            "Power per Unit kW": 2.00,
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
            "Power per Unit kW": 1.80,
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
            "Power per Unit kW": 1.00,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 4,
            "Company Priority": 1,
            "Preserve Minimum Units": 0
        },
        {
            "Appliance": "ACs / Heat Pumps",
            "Quantity": 4,
            "Power per Unit kW": 1.30,
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
            "Power per Unit kW": 1.60,
            "Connected": True,
            "Disconnectable": True,
            "Critical": False,
            "User Priority": 6,
            "Company Priority": 6,
            "Preserve Minimum Units": 0
        }
    ])


# =========================================================
# REAL LIFE HVAC AC MAP
# =========================================================
# X % and Y % are marker coordinates over ACs.png.
# You can adjust them from the Real Life Simulation page.

if "real_life_ac_map" not in st.session_state:
    st.session_state.real_life_ac_map = pd.DataFrame([
        {
            "AC ID": "AC-1",
            "Plan Label": "C-1",
            "Room / Area": "Dining Room",
            "X %": 96.0,
            "Y %": 62.0,
            "Working": True,
            "Critical / Preserve": False,
            "Forced By Company": False,
            "Priority": 6
        },
        {
            "AC ID": "AC-2",
            "Plan Label": "C-2",
            "Room / Area": "Upper Right Room",
            "X %": 96.0,
            "Y %": 25.5,
            "Working": True,
            "Critical / Preserve": False,
            "Forced By Company": False,
            "Priority": 5
        },
        {
            "AC ID": "AC-3",
            "Plan Label": "C-3",
            "Room / Area": "Master Bedroom Upper",
            "X %": 2.8,
            "Y %": 61.0,
            "Working": True,
            "Critical / Preserve": False,
            "Forced By Company": False,
            "Priority": 2
        },
        {
            "AC ID": "AC-4",
            "Plan Label": "C-4",
            "Room / Area": "Master Bedroom Lower",
            "X %": 2.8,
            "Y %": 82.0,
            "Working": True,
            "Critical / Preserve": False,
            "Forced By Company": False,
            "Priority": 3
        },
        {
            "AC ID": "AC-5",
            "Plan Label": "C-5",
            "Room / Area": "Bedroom 1",
            "X %": 68.5,
            "Y %": 17.5,
            "Working": True,
            "Critical / Preserve": False,
            "Forced By Company": False,
            "Priority": 4
        },
        {
            "AC ID": "AC-6",
            "Plan Label": "C-6",
            "Room / Area": "Bedroom 2",
            "X %": 15.2,
            "Y %": 17.5,
            "Working": True,
            "Critical / Preserve": False,
            "Forced By Company": False,
            "Priority": 1
        }
    ])


# =========================================================
# DATASET AND MODEL DETAILS
# =========================================================

def generate_training_data(n=1500):
    np.random.seed(42)

    lamps = np.random.randint(1, 20, n)
    acs = np.random.randint(0, 8, n)
    washing = np.random.randint(0, 5, n)
    heavy = np.random.randint(0, 8, n)
    occupants = np.random.randint(1, 12, n)
    house_size = np.random.randint(40, 500, n)

    gaussian_noise = np.random.normal(0, 0.45, n)

    baseline = (
        0.22 * lamps +
        1.15 * acs +
        0.90 * washing +
        1.55 * heavy +
        0.32 * occupants +
        0.010 * house_size +
        gaussian_noise
    )

    baseline = np.clip(baseline, 0.5, None)

    return pd.DataFrame({
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy,
        "occupants": occupants,
        "house_size": house_size,
        "historical_baseline_kwh": baseline
    })


def train_model_without_cache():
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
        n_estimators=150,
        max_depth=9,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics, df


model, metrics, training_df = train_model_without_cache()
mean_usage = training_df["historical_baseline_kwh"].mean()
std_usage = training_df["historical_baseline_kwh"].std()


# =========================================================
# UNLIMITED ENGINEERING BASELINE
# =========================================================
# This is the main calculation.
# It has no cap for washing machines or heavy machines.

def engineering_historical_baseline(
    lamps,
    acs,
    washing_machine,
    heavy_machines,
    occupants,
    house_size,
    climate_mode
):
    if climate_mode == "Cooling Mode - Hot Weather":
        ac_factor = 1.15
    elif climate_mode == "Heating Mode - Cold Weather":
        ac_factor = 1.10
    else:
        ac_factor = 1.00

    baseline = (
        0.22 * float(lamps) +
        1.15 * float(acs) * ac_factor +
        0.90 * float(washing_machine) +
        1.55 * float(heavy_machines) +
        0.32 * float(occupants) +
        0.010 * float(house_size)
    )

    return max(baseline, 0.5)


def calculate_house_capacity_advisory(house_size):
    house_size = max(float(house_size), 1)

    return {
        "Advisory Washing Machine Capacity": max(1, int(house_size // 35)),
        "Advisory Heavy Machine Capacity": max(0, int(house_size // 50)),
        "Advisory AC / Heat Pump Capacity": max(1, int(house_size // 25)),
        "Advisory Lamp Capacity": max(1, int(house_size // 8))
    }


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
        step=1,
        min_value=0,
        key=f"{title}_lamps"
    )

    acs = st.number_input(
        f"{title} - ACs / Heat Pumps",
        value=default_acs,
        step=1,
        min_value=0,
        key=f"{title}_acs"
    )

    washing = st.number_input(
        f"{title} - Washing Machines",
        value=default_washing,
        step=1,
        min_value=0,
        key=f"{title}_washing"
    )

    heavy = st.number_input(
        f"{title} - Heavy Machines",
        value=default_heavy,
        step=1,
        min_value=0,
        key=f"{title}_heavy"
    )

    occupants = st.number_input(
        f"{title} - Occupants",
        value=default_occupants,
        step=1,
        min_value=0,
        key=f"{title}_occupants"
    )

    size = st.number_input(
        f"{title} - House Size m²",
        value=default_size,
        step=1,
        min_value=1,
        key=f"{title}_size"
    )

    capacity = calculate_house_capacity_advisory(size)

    if washing > capacity["Advisory Washing Machine Capacity"]:
        st.warning(
            f"Washing machines are higher than the advisory capacity for {size} m². "
            f"Advisory capacity: {capacity['Advisory Washing Machine Capacity']}. "
            f"The full number is still calculated."
        )

    if heavy > capacity["Advisory Heavy Machine Capacity"]:
        st.warning(
            f"Heavy machines are higher than the advisory capacity for {size} m². "
            f"Advisory capacity: {capacity['Advisory Heavy Machine Capacity']}. "
            f"The full number is still calculated."
        )

    baseline = engineering_historical_baseline(
        lamps=lamps,
        acs=acs,
        washing_machine=washing,
        heavy_machines=heavy,
        occupants=occupants,
        house_size=size,
        climate_mode=st.session_state.climate_mode
    )

    df = pd.DataFrame([{
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy,
        "occupants": occupants,
        "house_size": size
    }])

    return df, baseline, capacity


# =========================================================
# LOAD SHEDDING ENGINE
# =========================================================

def calculate_current_connected_load(appliance_df):
    df = appliance_df.copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["Power per Unit kW"] = pd.to_numeric(df["Power per Unit kW"], errors="coerce").fillna(0)

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

    df["Disconnected Units"] = 0.0
    df["Remaining Units"] = df["Quantity"]
    df["Shed kW"] = 0.0

    original_load = df["Connected Load kW"].sum()

    if original_load <= 0:
        return df, 0, 0, 0, "No active connected load."

    requested_reduction_kw = original_load * requested_reduction_percent / 100
    mandatory_reduction_kw = original_load * mandatory_minimum_percent / 100

    if refuse_disconnect:
        if enforcement_enabled and user_failed_to_respond:
            target_reduction_kw = mandatory_reduction_kw
            active_policy = "Company Emergency Enforcement"
            status = "User refused or ignored the request. Mandatory reduction was enforced."
        else:
            target_reduction_kw = 0
            active_policy = "User Refused Disconnection"
            status = "User refused disconnection. No load was disconnected. Premium pricing applies."
    else:
        target_reduction_kw = requested_reduction_kw
        active_policy = policy_mode
        status = "User smart meter priority was applied."

    if target_reduction_kw <= 0:
        return df, original_load, original_load, 0, status

    if active_policy == "Company Emergency Enforcement":
        priority_col = "Company Priority"
    elif policy_mode == "Company Priority":
        priority_col = "Company Priority"
    else:
        priority_col = "User Priority"

    candidates = df[
        (df["Connected"] == True) &
        (df["Disconnectable"] == True) &
        (df["Critical"] == False)
    ].copy()

    candidates = candidates.sort_values(by=priority_col, ascending=True)

    shed_so_far = 0.0

    for idx, row in candidates.iterrows():
        if shed_so_far >= target_reduction_kw:
            break

        quantity = float(row["Quantity"])
        power = float(row["Power per Unit kW"])
        appliance = str(row["Appliance"])
        preserve_minimum = float(row["Preserve Minimum Units"])

        if climate_mode == "Cooling Mode - Hot Weather" and "AC" in appliance:
            preserve_minimum = max(preserve_minimum, 1)

        if climate_mode == "Heating Mode - Cold Weather":
            if "Heater" in appliance or "Heat Pump" in appliance or "AC" in appliance:
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

    return df, original_load, final_load, achieved_reduction_percent, status


# =========================================================
# BILLING ENGINE
# =========================================================

def billing_engine(
    baseline,
    final_usage,
    mean_usage,
    grid_stress,
    new_company_growth_mode,
    refused_disconnect,
    achieved_reduction_percent,
    mandatory_reduction_percent
):
    normal_usage = min(final_usage, baseline)
    premium_usage = max(final_usage - baseline, 0)

    bill = normal_usage * BASE_RATE
    premium_charge = 0
    penalty = 0
    bonus = 0
    discount = 0
    status = []

    if new_company_growth_mode and not grid_stress:
        if final_usage > mean_usage:
            bonus = bill * BONUS_RATE
            bill -= bonus
            status.append("Growth bonus applied because the company wants to increase average demand.")
        else:
            status.append("Normal billing. Usage is below the growth target.")

    if grid_stress:
        if premium_usage > 0:
            premium_charge = premium_usage * PREMIUM_PRESERVATION_RATE
            bill += premium_charge
            status.append("Premium Load Preservation Pricing applied for usage above historical baseline.")

        if achieved_reduction_percent < mandatory_reduction_percent:
            penalty = final_usage * 0.20
            bill += penalty
            status.append("Mandatory reduction was not achieved. Grid stress penalty applied.")

        if achieved_reduction_percent >= mandatory_reduction_percent:
            discount = bill * DISCOUNT_RATE
            bill -= discount
            status.append("Grid support discount applied.")

    if refused_disconnect and grid_stress:
        status.append("User refused disconnection. Premium convenience pricing applies.")

    if not status:
        status.append("Normal operating condition.")

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
# REAL LIFE HVAC IMAGE OVERLAY
# =========================================================

def render_real_life_ac_plan(image_path, ac_df):
    img64 = get_base64_image(image_path)

    if img64 is None:
        st.error(
            f"Image file not found: {image_path}. "
            f"Put ACs.png in the same folder as app.py or upload it to your GitHub repository."
        )
        return

    markers_html = ""

    for _, row in ac_df.iterrows():
        ac_id = str(row["AC ID"])
        plan_label = str(row["Plan Label"])
        room = str(row["Room / Area"])
        x = float(row["X %"])
        y = float(row["Y %"])
        working = bool(row["Working"])
        critical = bool(row["Critical / Preserve"])
        forced = bool(row["Forced By Company"])

        if forced:
            symbol = "X"
            color = "#ff0000"
            label = f"{ac_id} / {plan_label} - FORCED DISCONNECTED - {room}"
        elif working:
            symbol = "O"
            color = "#00ff88"
            label = f"{ac_id} / {plan_label} - WORKING - {room}"
        else:
            symbol = "X"
            color = "#ff3333"
            label = f"{ac_id} / {plan_label} - DISCONNECTED - {room}"

        if critical and working:
            color = "#00e5ff"
            label = f"{ac_id} / {plan_label} - PRESERVED / WORKING - {room}"

        markers_html += f"""
        <div class="ac-marker" style="
            left:{x}%;
            top:{y}%;
            color:{color};
            border-color:{color};
        ">
            {symbol}
            <div class="ac-label">{label}</div>
        </div>
        """

    html = f"""
    <style>
    .plan-container {{
        position: relative;
        width: 100%;
        border: 2px solid rgba(255,255,255,0.45);
        border-radius: 14px;
        overflow: hidden;
        background: #111827;
    }}

    .plan-container img {{
        width: 100%;
        display: block;
    }}

    .ac-marker {{
        position: absolute;
        transform: translate(-50%, -50%);
        width: 46px;
        height: 46px;
        border: 4px solid;
        border-radius: 50%;
        background: rgba(0,0,0,0.78);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        font-weight: 900;
        font-family: Arial, sans-serif;
        box-shadow: 0 0 20px currentColor;
        z-index: 20;
        cursor: pointer;
    }}

    .ac-label {{
        display: none;
        position: absolute;
        top: 52px;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        background: rgba(0,0,0,0.92);
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 14px;
        border: 1px solid rgba(255,255,255,0.3);
        z-index: 50;
    }}

    .ac-marker:hover .ac-label {{
        display: block;
    }}
    </style>

    <div class="plan-container">
        <img src="data:image/png;base64,{img64}" />
        {markers_html}
    </div>
    """

    components.html(html, height=820, scrolling=True)


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
    st.metric("MAE", f"{metrics['MAE']:.2f} kWh")
    st.metric("R² Score", f"{metrics['R2']:.2f}")

    st.divider()

    st.header("Electricity Prices")
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
# REAL LIFE SIMULATION PAGE
# =========================================================

if page == "Real Life Simulation":

    st.title("Real Life Simulation - HVAC Plan SCADA Trial")

    st.warning(
        "This page uses ACs.png. It displays the HVAC plan and places live X/O marks over the six AC units."
    )

    st.markdown("""
    ### Symbol Meaning

    - **O** = AC is working
    - **X** = AC is disconnected
    - **Forced By Company** = emergency forced disconnection
    - **X % / Y %** = marker location on the image
    """)

    ac_df = st.session_state.real_life_ac_map.copy()
    all_acs = ac_df["AC ID"].tolist()

    st.subheader("Trial SCADA AC Control")

    disconnected_default = ac_df.loc[ac_df["Working"] == False, "AC ID"].tolist()

    selected_disconnected = st.multiselect(
        "Choose ACs to disconnect",
        all_acs,
        default=disconnected_default
    )

    ac_df["Working"] = ~ac_df["AC ID"].isin(selected_disconnected)

    forced_default = ac_df.loc[ac_df["Forced By Company"] == True, "AC ID"].tolist()

    forced_acs = st.multiselect(
        "Choose ACs forced disconnected by company",
        all_acs,
        default=forced_default
    )

    ac_df["Forced By Company"] = ac_df["AC ID"].isin(forced_acs)
    ac_df.loc[ac_df["Forced By Company"] == True, "Working"] = False

    st.subheader("X and Y Coordinate Editor")

    st.info(
        "If the X or O is not exactly on the AC symbol, edit X % and Y %. "
        "After editing, the marker position updates on the HVAC plan."
    )

    edited_ac_df = st.data_editor(
        ac_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "AC ID": st.column_config.TextColumn("AC ID"),
            "Plan Label": st.column_config.TextColumn("Plan Label"),
            "Room / Area": st.column_config.TextColumn("Room / Area"),
            "X %": st.column_config.NumberColumn("X %", step=0.1, min_value=0.0, max_value=100.0),
            "Y %": st.column_config.NumberColumn("Y %", step=0.1, min_value=0.0, max_value=100.0),
            "Working": st.column_config.CheckboxColumn("Working"),
            "Critical / Preserve": st.column_config.CheckboxColumn("Critical / Preserve"),
            "Forced By Company": st.column_config.CheckboxColumn("Forced By Company"),
            "Priority": st.column_config.NumberColumn("Priority", step=1)
        }
    )

    edited_ac_df.loc[edited_ac_df["Forced By Company"] == True, "Working"] = False
    st.session_state.real_life_ac_map = edited_ac_df

    working_count = int(st.session_state.real_life_ac_map["Working"].sum())
    disconnected_count = int((st.session_state.real_life_ac_map["Working"] == False).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Working ACs", working_count)
    c2.metric("Disconnected ACs", disconnected_count)
    c3.metric("Total ACs", len(st.session_state.real_life_ac_map))

    st.divider()

    st.subheader("Live HVAC Plan With X / O Overlay")

    render_real_life_ac_plan(
        image_path="ACs.png",
        ac_df=st.session_state.real_life_ac_map
    )

    st.divider()

    fig_ac = px.bar(
        st.session_state.real_life_ac_map,
        x="AC ID",
        y="Priority",
        color="Working",
        title="AC Status and Priority",
        text="Plan Label"
    )

    fig_ac.update_layout(
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=520
    )

    st.plotly_chart(fig_ac, use_container_width=True)

    st.stop()


# =========================================================
# HOW TO USE PAGE
# =========================================================

if page == "How To Use":

    st.title("How To Use The Website - Manual Catalogue")

    st.markdown("""
    ## 1. Purpose

    This website simulates a smart electricity distribution system with SCADA-style control.

    It allows the user to:

    - Estimate historical baseline consumption
    - Simulate peak load events
    - Apply smart meter priority control
    - Override company disconnection rules
    - Force emergency reduction during real line stress
    - Visualize AC operation on a real HVAC plan image

    ---

    ## 2. SCADA Control Center

    This is the main operating page.

    Use it to control:

    - Real Stress On Line
    - Peak Usage Event
    - New Company Growth Mode
    - Emergency Enforcement
    - Climate mode
    - Requested reduction percentage
    - Mandatory reduction percentage

    If there is real stress on the line, the issue is physical grid safety.  
    In that case, payment alone is not enough, and mandatory reduction may be required.

    ---

    ## 3. Smart Meter Override Page

    This page lets the client control appliance priority.

    You can edit:

    - Quantity
    - Power per unit
    - Connected status
    - Disconnectable status
    - Critical status
    - User priority
    - Company priority
    - Minimum preserved units

    Lower priority number means the appliance disconnects first.

    ---

    ## 4. Real Life Simulation Page

    This page uses:

    `ACs.png`

    It places live symbols on the plan:

    - O means working
    - X means disconnected

    You can disconnect AC-1, AC-2, AC-3, AC-4, AC-5, or AC-6.

    If you change the disconnected ACs, the image overlay updates immediately.

    ---

    ## 5. X and Y Coordinates

    X % and Y % control the location of each mark.

    If the mark is not exactly on the AC symbol, edit the values until the symbol is centered.

    ---

    ## 6. Unlimited Input Calculation

    You can enter any number of washing machines or heavy machines.

    The system will show advisory warnings if the value is unrealistic for the house size.

    However, the system still calculates the full value.

    Example:

    If washing machines = 500, the formula includes:

    `0.90 × 500`

    There is no maximum cap.
    """)

    st.stop()


# =========================================================
# AI & MODEL DETAILS PAGE
# =========================================================

if page == "AI & Model Details":

    st.title("AI & Model Details")

    st.markdown("""
    ## Random Numbers

    Random numbers are used to generate many different synthetic households.

    This makes the dataset contain different numbers of lamps, ACs, washing machines, heavy machines, occupants, and house sizes.

    ## Gaussian Randomness

    Gaussian randomness means bell-curve noise.

    The code uses:

    `np.random.normal(0, 0.45, n)`

    This means:

    - Average noise = 0
    - Standard deviation = 0.45
    - Most random changes are small
    - Very large random changes are rare

    This simulates real-life variation between homes.

    ## MAE

    MAE means **Mean Absolute Error**.

    It is the average absolute difference between predicted values and actual values.

    Lower MAE is better.

    ## R²

    R² means **Coefficient of Determination**.

    It shows how much variation in the target value is explained by the model.

    R² close to 1 means strong model performance.

    ## Important Note

    The main SCADA calculation does not depend on Random Forest prediction.

    The main calculation uses an engineering formula so that very large numbers like 500 washing machines are still fully calculated.
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
        name="Historical Baseline Histogram",
        opacity=0.55,
        marker_color="gray"
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Normal Bell Curve",
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
        title="Historical Baseline Bell Curve",
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
# SMART METER OVERRIDE PAGE
# =========================================================

if page == "Smart Meter Override Page":

    st.title("Smart Meter Override Page")

    st.warning(
        "This page allows the client to override smart meter priority rules. "
        "Changes here affect the SCADA Control Center."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.selected_user_policy = st.selectbox(
            "Priority Control Mode",
            ["Manual User Priority", "Company Priority"],
            index=["Manual User Priority", "Company Priority"].index(st.session_state.selected_user_policy)
        )

    with col2:
        st.session_state.refuse_disconnect = st.checkbox(
            "I do not want to disconnect anything",
            value=st.session_state.refuse_disconnect
        )

    with col3:
        st.session_state.climate_mode = st.selectbox(
            "Climate Mode",
            [
                "Cooling Mode - Hot Weather",
                "Heating Mode - Cold Weather",
                "Neutral Weather"
            ],
            index=[
                "Cooling Mode - Hot Weather",
                "Heating Mode - Cold Weather",
                "Neutral Weather"
            ].index(st.session_state.climate_mode)
        )

    st.subheader("Edit Appliance Priority and Connection Status")

    edited_df = st.data_editor(
        st.session_state.appliance_config,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Appliance": st.column_config.TextColumn("Appliance"),
            "Quantity": st.column_config.NumberColumn("Quantity", step=1, min_value=0),
            "Power per Unit kW": st.column_config.NumberColumn("Power per Unit kW", step=0.01, min_value=0.0),
            "Connected": st.column_config.CheckboxColumn("Connected"),
            "Disconnectable": st.column_config.CheckboxColumn("Disconnectable"),
            "Critical": st.column_config.CheckboxColumn("Critical"),
            "User Priority": st.column_config.NumberColumn("User Priority", step=1),
            "Company Priority": st.column_config.NumberColumn("Company Priority", step=1),
            "Preserve Minimum Units": st.column_config.NumberColumn("Preserve Minimum Units", step=1, min_value=0)
        }
    )

    st.session_state.appliance_config = edited_df

    st.divider()

    load_df = calculate_current_connected_load(st.session_state.appliance_config)

    st.metric("Total Connected Load", f"{load_df['Connected Load kW'].sum():.2f} kW")

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

    st.stop()


# =========================================================
# SCADA CONTROL CENTER PAGE
# =========================================================

st.title("SCADA Dynamic Pricing & Smart Meter Control Center")

st.warning(
    "⚠ PEAK EVENT NOTIFICATION: High electrical demand may be active. "
    "The smart meter may request load reduction. Premium Load Preservation Pricing may apply."
)

st.markdown(
    """
    <div class="scada-card">
    <div class="big-status">Integrated Operating Scenario</div>
    This dashboard combines historical baseline calculation, smart meter override,
    dynamic tariff pricing, emergency reduction, and climate-based preservation.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GRID CONTROL
# =========================================================

st.header("Grid Event & Company Control Panel")

g1, g2, g3, g4 = st.columns(4)

with g1:
    grid_stress = st.checkbox("Real Stress On Line", value=True)

with g2:
    peak_event = st.checkbox("Peak Usage Event", value=True)

with g3:
    new_company_growth_mode = st.checkbox("New Company Growth Mode", value=False)

with g4:
    enforcement_enabled = st.checkbox("Emergency Enforcement Enabled", value=True)

st.session_state.climate_mode = st.selectbox(
    "Climate / Seasonal Operating Mode",
    [
        "Cooling Mode - Hot Weather",
        "Heating Mode - Cold Weather",
        "Neutral Weather"
    ],
    index=[
        "Cooling Mode - Hot Weather",
        "Heating Mode - Cold Weather",
        "Neutral Weather"
    ].index(st.session_state.climate_mode)
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
        value=60,
        step=1,
        min_value=0
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
    st.success("Grid is stable. Mandatory emergency protection is not required.")


# =========================================================
# HOUSEHOLD INPUTS
# =========================================================

st.divider()
st.header("Household Inputs and Historical Baseline Estimate")

col_a, col_b = st.columns(2)

with col_a:
    person_a, baseline_a, capacity_a = household_input(
        "Person A: Low Baseline Home",
        default_lamps=3,
        default_acs=0,
        default_washing=0,
        default_heavy=0,
        default_occupants=1,
        default_size=60
    )

with col_b:
    person_b, baseline_b, capacity_b = household_input(
        "Person B: Heavy Usage Home",
        default_lamps=10,
        default_acs=4,
        default_washing=1,
        default_heavy=3,
        default_occupants=5,
        default_size=220
    )


# =========================================================
# BASELINE METRICS
# =========================================================

st.divider()
st.header("Historical Baseline Estimate")

m1, m2, m3 = st.columns(3)

m1.metric("Historical Baseline Estimate - Person A", f"{baseline_a:.2f} kWh")
m2.metric("Historical Baseline Estimate - Person B", f"{baseline_b:.2f} kWh")
m3.metric("Population Mean Baseline", f"{mean_usage:.2f} kWh")


# =========================================================
# CLIENT SELECTION
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
    selected_capacity = capacity_a
else:
    selected_baseline = baseline_b
    selected_capacity = capacity_b

with st.expander("Show Advisory Capacity According To House Size"):
    st.json(selected_capacity)


# =========================================================
# USAGE INPUT
# =========================================================

st.subheader("Client Usage During Peak Event")

requested_usage = st.number_input(
    "Requested / Original Usage During Peak Event kWh",
    value=float(selected_baseline + 2),
    step=0.1,
    min_value=0.0
)


# =========================================================
# SMART METER SIMULATION
# =========================================================

shed_df, original_load_kw, final_load_kw, achieved_reduction_percent, enforcement_status = smart_meter_shed_load(
    appliance_df=st.session_state.appliance_config,
    requested_reduction_percent=voluntary_reduction_percent,
    policy_mode=st.session_state.selected_user_policy,
    refuse_disconnect=st.session_state.refuse_disconnect,
    climate_mode=st.session_state.climate_mode,
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
    final_usage=final_usage,
    mean_usage=mean_usage,
    grid_stress=grid_stress and peak_event,
    new_company_growth_mode=new_company_growth_mode,
    refused_disconnect=st.session_state.refuse_disconnect,
    achieved_reduction_percent=achieved_reduction_percent,
    mandatory_reduction_percent=mandatory_reduction_percent
)


# =========================================================
# LIVE STATUS
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
    st.error("Grid Protection Warning: The mandatory reduction target was not achieved.")
else:
    st.success("Grid Protection Status: Reduction condition is acceptable.")

st.info(enforcement_status)


# =========================================================
# BILLING DETAILS
# =========================================================

st.divider()
st.header("Billing & Condition Results")

billing_df = pd.DataFrame([{
    "Client": selected_person,
    "Historical Baseline Estimate kWh": selected_baseline,
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
# SMART METER TABLE
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
    "Historical Baseline Bell Curve",
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
    x_max = max(training_df["historical_baseline_kwh"].max(), selected_baseline + 2)

    x = np.linspace(
        training_df["historical_baseline_kwh"].min(),
        x_max,
        800
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
        xaxis_title="Historical Baseline Consumption kWh",
        yaxis_title="Probability Density",
        template="plotly_dark",
        font=dict(size=18),
        title_font=dict(size=26),
        height=650
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Z-Score", f"{z_score:.2f}")
    c2.metric("Percentile", f"{percentile:.2f}%")
    c3.metric("Population Mean", f"{mean_usage:.2f} kWh")


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


# =========================================================
# FINAL SUMMARY
# =========================================================

st.divider()
st.header("Final SCADA Decision Summary")

if grid_stress and peak_event and achieved_reduction_percent < mandatory_reduction_percent:
    st.error(
        "Final Decision: The user did not satisfy the minimum physical grid protection requirement. "
        "The company may apply enforcement, restriction, or blocking logic in this simulation."
    )

elif st.session_state.refuse_disconnect and grid_stress and peak_event:
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
