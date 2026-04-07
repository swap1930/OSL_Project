import os
import sys
from datetime import datetime

import streamlit as st

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.auto_retrain import log_prediction
from src.predict import generate_realistic_data, predict
from src.huggingface_explanation import generate_ai_explanation


def cleanup_session_state():
    """Clean up corrupted session state data"""
    required_live_keys = ["timestamp", "result", "probability", "data"]
    required_manual_keys = ["timestamp", "result", "probability", "data", "parameters"]

    # Clean live predictions
    if "live_predictions" in st.session_state:
        st.session_state.live_predictions = [
            p
            for p in st.session_state.live_predictions
            if all(k in p for k in required_live_keys)
        ]

    # Clean manual predictions
    if "manual_predictions" in st.session_state:
        st.session_state.manual_predictions = [
            p
            for p in st.session_state.manual_predictions
            if all(k in p for k in required_manual_keys)
        ]

    # Clean manual history
    if "manual_history" in st.session_state:
        st.session_state.manual_history = [
            p
            for p in st.session_state.manual_history
            if all(k in p for k in required_manual_keys)
        ]


def main():
    st.set_page_config(
        page_title="Machine Failure Prediction System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean up corrupted session state
    cleanup_session_state()

    st.markdown(
        """
    <style>
        .main-header { text-align: center; padding: 2rem 0; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    st.sidebar.title("🎛️ Control Panel")

    selected_mode = st.sidebar.radio(
        "Select Mode", ["Live Streaming", "Manual Input"], index=0
    )

    st.sidebar.markdown("---")

    # Mode-specific controls rendered ONLY in the sidebar
    # Initialize variables to avoid UnboundLocalError
    streaming_active = False
    alert_threshold = 0.7
    stream_interval = 2

    if selected_mode == "Live Streaming":
        st.sidebar.subheader("Streaming Controls")
        streaming_active = st.sidebar.checkbox(
            " Start Live Streaming", value=False, key="live_streaming"
        )
        alert_threshold = st.sidebar.slider(
            " Alert Threshold", 0.0, 1.0, 0.7, 0.05, key="live_alert_threshold"
        )
        stream_interval = st.sidebar.slider(
            " Update Interval (s)", 1, 10, 2, key="live_stream_interval"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader(" Machine Connection")
        machine_status = st.sidebar.selectbox(
            "Select Machine",
            ["Machine-001", "Machine-002", "Machine-003", "Simulator"],
            key="machine_select",
        )
        if machine_status == "Simulator":
            st.sidebar.info(" Using data simulator")
        else:
            st.sidebar.success(f" Connected to {machine_status}")
            st.sidebar.write(f" Last ping: {datetime.now().strftime('%H:%M:%S')}")

        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear History", key="clear_history_live"):
            st.session_state.live_predictions = []
            st.session_state.live_failure_count = 0
            st.session_state.live_total_count = 0
            st.rerun()

    else:  # Manual Input
        st.sidebar.subheader("Manual Input Controls")
        alert_threshold = st.sidebar.slider(
            " Alert Threshold", 0.0, 1.0, 0.7, 0.05, key="manual_alert_threshold"
        )

        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Clear History", key="clear_history_sidebar"):
            st.session_state.manual_predictions = []
            st.session_state.manual_history = []
            st.rerun()

    # ── MAIN CONTENT ─────────────────────────────────────────────────────────
    st.title(" Machine Failure Prediction System")

    if selected_mode == "Live Streaming":
        render_live_streaming_page(
            streaming_active=streaming_active,
            alert_threshold=alert_threshold,
            stream_interval=stream_interval,
        )
    else:
        render_manual_input_page(alert_threshold=alert_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Live Streaming
# ─────────────────────────────────────────────────────────────────────────────
def _initialize_live_session_state():
    """Initialize session state for live streaming"""
    if "live_predictions" not in st.session_state:
        st.session_state.live_predictions = []
    if "live_failure_count" not in st.session_state:
        st.session_state.live_failure_count = 0
    if "live_total_count" not in st.session_state:
        st.session_state.live_total_count = 0


def _update_live_predictions(data, result, probability, timestamp):
    """Update session state with new prediction data"""
    if result is not None and probability is not None:
        entry = {
            "timestamp": timestamp,
            "result": result,
            "probability": probability,
            "data": data,
        }
        st.session_state.live_predictions.append(entry)
        st.session_state.live_total_count += 1
        if result == 1:
            st.session_state.live_failure_count += 1
        if len(st.session_state.live_predictions) > 100:
            st.session_state.live_predictions.pop(0)
        if st.session_state.live_total_count % 20 == 0:
            log_prediction(data, result, probability, timestamp)


def _render_live_metrics(data, result, probability, alert_threshold):
    """Render key metrics for live streaming"""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(" Total Readings", st.session_state.live_total_count)
    col2.metric(" Failures", st.session_state.live_failure_count)

    if st.session_state.live_total_count > 0:
        failure_rate = (
            st.session_state.live_failure_count
            / st.session_state.live_total_count
        ) * 100
        col3.metric(" Failure Rate", f"{failure_rate:.1f}%")
    else:
        col3.metric(" Failure Rate", "0.0%")

    if st.session_state.live_predictions:
        latest_entry = st.session_state.live_predictions[-1]
        if "probability" in latest_entry:
            latest_prob = latest_entry["probability"]
            col4.metric(" Current Risk", f"{latest_prob:.1%}")
        else:
            col4.metric(" Current Risk", "N/A")
    else:
        col4.metric(" Current Risk", "0.0%")

    # Current metrics display
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(" Air Temp", f"{data[0]:.1f}K")
    c2.metric(" Process Temp", f"{data[1]:.1f}K")
    c3.metric(" RPM", f"{data[2]:.0f}")
    c4.metric(" Torque", f"{data[3]:.1f}Nm")
    c5.metric(" Wear", f"{data[4]:.1f}min")

    if probability >= alert_threshold:
        st.error(f" CRITICAL ALERT: Failure probability {probability:.1%}!")
    elif result == 1:
        st.warning(" Machine Failure Predicted!")
    else:
        st.success(" Machine Operating Normally")


def _render_live_analytics(alert_threshold):
    """Render live analytics charts"""
    if not st.session_state.live_predictions:
        return
        
    st.markdown("---")
    st.subheader(" Live Analytics")

    import pandas as pd
    import plotly.graph_objects as go

    # Filter out any entries without required keys
    valid_predictions = [
        p
        for p in st.session_state.live_predictions
        if all(k in p for k in ["timestamp", "probability"])
    ]

    if valid_predictions:
        df = pd.DataFrame(valid_predictions)
        col1, col2 = st.columns(2)

        with col1:
            fig_prob = go.Figure()
            fig_prob.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["probability"],
                    mode="lines+markers",
                    name="Failure Probability",
                    line=dict(color="red", width=2),
                )
            )
            fig_prob.add_hline(
                y=alert_threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text="Alert Threshold",
            )
            fig_prob.update_layout(
                title=" Real-time Failure Probability",
                xaxis_title="Time",
                yaxis_title="Probability",
                height=300,
            )
            st.plotly_chart(fig_prob, width="stretch")

        with col2:
            st.subheader(" Recent Events")
            recent = df.tail(50)[
                ["timestamp", "probability"]
            ].copy()  # Show more events
            recent["Risk Level"] = recent["probability"].apply(
                lambda x: (
                    " Critical"
                    if x >= alert_threshold
                    else " Medium" if x >= 0.3 else " Low"
                )
            )
            recent["Time"] = recent["timestamp"].dt.strftime("%H:%M:%S")
            recent["Probability"] = recent["probability"].apply(
                lambda x: f"{x:.1%}"
            )

            # Use dataframe with height parameter for scrolling
            st.dataframe(
                recent[["Time", "Probability", "Risk Level"]],
                width="stretch",
                height=300,
            )


def _render_static_metrics(alert_threshold):
    """Render static metrics when streaming is stopped"""
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(" Total Readings", st.session_state.live_total_count)
    col2.metric(" Failures", st.session_state.live_failure_count)

    if st.session_state.live_total_count > 0:
        failure_rate = (
            st.session_state.live_failure_count / st.session_state.live_total_count
        ) * 100
        col3.metric(" Failure Rate", f"{failure_rate:.1f}%")
    else:
        col3.metric(" Failure Rate", "0.0%")

    if st.session_state.live_predictions:
        latest_entry = st.session_state.live_predictions[-1]
        if "probability" in latest_entry:
            latest_prob = latest_entry["probability"]
            col4.metric(" Current Risk", f"{latest_prob:.1%}")
        else:
            col4.metric(" Current Risk", "N/A")
    else:
        col4.metric(" Current Risk", "0.0%")


def render_live_streaming_page(streaming_active, alert_threshold, stream_interval):
    """Render live streaming page with machine monitoring"""
    st.subheader("Live Machine Monitoring")
    st.markdown("---")

    _initialize_live_session_state()

    # Live data stream
    if streaming_active:
        placeholder = st.empty()

        while True:  # Streamlit re-runs stop this loop when checkbox unchecked
            data = generate_realistic_data()
            result, probability = predict(data)
            timestamp = datetime.now()

            _update_live_predictions(data, result, probability, timestamp)

            with placeholder.container():
                _render_live_metrics(data, result, probability, alert_threshold)
                _render_live_analytics(alert_threshold)

            import time
            time.sleep(stream_interval)

    # Show static metrics when streaming is stopped
    else:
        _render_static_metrics(alert_threshold)
        
        # Show analytics when streaming is stopped (static view)
        if st.session_state.live_predictions:
            # Filter out any entries without required keys
            valid_predictions = [
                p
                for p in st.session_state.live_predictions
                if all(k in p for k in ["timestamp", "probability"])
            ]

            if valid_predictions:
                st.markdown("---")
                st.subheader("Live Analytics")

                import pandas as pd
                import plotly.graph_objects as go

                df = pd.DataFrame(valid_predictions)
                col1, col2 = st.columns(2)

                with col1:
                    fig_prob = go.Figure()
                    fig_prob.add_trace(
                        go.Scatter(
                            x=df["timestamp"],
                            y=df["probability"],
                            mode="lines+markers",
                            name="Failure Probability",
                            line=dict(color="red", width=2),
                        )
                    )
                    fig_prob.add_hline(
                        y=alert_threshold,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="Alert Threshold",
                    )
                    fig_prob.update_layout(
                        title="Real-time Failure Probability",
                        xaxis_title="Time",
                        yaxis_title="Probability",
                        height=300,
                    )
                    st.plotly_chart(fig_prob, width="stretch")

                with col2:
                    st.subheader(" Recent Events")
                    recent = df.tail(50)[
                        ["timestamp", "probability"]
                    ].copy()  # Show more events
                    recent["Risk Level"] = recent["probability"].apply(
                        lambda x: (
                            " Critical"
                            if x >= alert_threshold
                            else " Medium" if x >= 0.3 else " Low"
                        )
                    )
                    recent["Time"] = recent["timestamp"].dt.strftime("%H:%M:%S")
                    recent["Probability"] = recent["probability"].apply(
                        lambda x: f"{x:.1%}"
                    )

                    # Use dataframe with height parameter for scrolling
                    st.dataframe(
                        recent[["Time", "Probability", "Risk Level"]],
                        width="stretch",
                        height=300,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Manual Input
# ─────────────────────────────────────────────────────────────────────────────
def render_manual_input_page(alert_threshold):
    st.subheader("Manual Machine Prediction")
    st.markdown("---")

    if "manual_predictions" not in st.session_state:
        st.session_state.manual_predictions = []
    if "manual_history" not in st.session_state:
        st.session_state.manual_history = []

    # Initialize preset values if not exists
    if "preset_values" not in st.session_state:
        st.session_state.preset_values = {
            "air_temp": 300.0,
            "process_temp": 310.0,
            "rpm": 1500.0,
            "torque": 40.0,
            "wear": 50.0,
        }

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Input Parameters**")
        air_temp = st.number_input(
            " Air Temperature (K)",
            value=st.session_state.preset_values["air_temp"],
            min_value=250.0,
            max_value=350.0,
            step=0.1,
            key="manual_air_temp",
        )
        process_temp = st.number_input(
            " Process Temperature (K)",
            value=st.session_state.preset_values["process_temp"],
            min_value=250.0,
            max_value=400.0,
            step=0.1,
            key="manual_process_temp",
        )
        rpm = st.number_input(
            " Rotational Speed (RPM)",
            value=st.session_state.preset_values["rpm"],
            min_value=1000.0,
            max_value=3000.0,
            step=10.0,
            key="manual_rpm",
        )

    with col2:
        st.write("**Advanced Parameters**")
        torque = st.number_input(
            " Torque (Nm)",
            value=st.session_state.preset_values["torque"],
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            key="manual_torque",
        )
        wear = st.number_input(
            " Tool Wear (min)",
            value=st.session_state.preset_values["wear"],
            min_value=0.0,
            max_value=300.0,
            step=1.0,
            key="manual_wear",
        )

        st.write("**Quick Presets**")
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            if st.button(" Normal", key="preset_normal"):
                st.session_state.preset_values = {
                    "air_temp": 300.0,
                    "process_temp": 310.0,
                    "rpm": 1500.0,
                    "torque": 40.0,
                    "wear": 50.0,
                }
                st.rerun()
        with cp2:
            if st.button(" High Stress", key="preset_high"):
                st.session_state.preset_values = {
                    "air_temp": 305.0,
                    "process_temp": 320.0,
                    "rpm": 1800.0,
                    "torque": 60.0,
                    "wear": 150.0,
                }
                st.rerun()
        with cp3:
            if st.button(" Critical", key="preset_critical"):
                st.session_state.preset_values = {
                    "air_temp": 310.0,
                    "process_temp": 330.0,
                    "rpm": 2000.0,
                    "torque": 80.0,
                    "wear": 200.0,
                }
                st.rerun()

    st.markdown("---")

    if st.button(
        " Predict Failure Risk", type="primary", width="stretch", key="predict_button"
    ):
        data = [air_temp, process_temp, rpm, torque, wear]
        result, probability = predict(data)
        timestamp = datetime.now()

        # Validate prediction results before adding to session state
        if result is not None and probability is not None:
            # Generate AI explanation
            feature_names = [
                "Air temperature [K]",
                "Process temperature [K]", 
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
            ai_explanation = generate_ai_explanation(
                prediction_result=result,
                probability=probability,
                input_data=data,
                feature_names=feature_names
            )
            
            entry = {
                "timestamp": timestamp,
                "result": result,
                "probability": probability,
                "data": data,
                "parameters": {
                    "air_temp": air_temp,
                    "process_temp": process_temp,
                    "rpm": rpm,
                    "torque": torque,
                    "wear": wear,
                },
                "ai_explanation": ai_explanation,
            }
            st.session_state.manual_predictions.append(entry)
            st.session_state.manual_history.append(entry)
            log_prediction(data, result, probability, timestamp)
            st.rerun()
        else:
            st.error(" Prediction failed. Please try again.")

    if st.session_state.manual_predictions:
        render_prediction_result(
            st.session_state.manual_predictions[-1], alert_threshold
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction result card
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction_result(prediction, alert_threshold):
    import plotly.graph_objects as go

    # Validate prediction has required keys
    if not all(
        k in prediction for k in ["result", "probability", "parameters", "data"]
    ):
        st.error(" Invalid prediction data. Please try again.")
        return

    st.markdown("---")
    st.subheader(" Prediction Result")

    result = prediction["result"]
    probability = prediction["probability"]

    col1, col2, col3 = st.columns(3)

    with col1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Failure Risk (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgray"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": alert_threshold * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width="stretch")

    with col2:
        risk_level = get_risk_level(probability)
        st.metric(" Risk Level", risk_level["label"])
        st.metric(" Confidence", f"{probability:.1%}")
        st.metric(" Status", "FAILURE" if result == 1 else "NORMAL")

        if probability >= alert_threshold:
            st.error(" **IMMEDIATE ACTION REQUIRED**")
            st.write("• Stop machine immediately")
            st.write("• Perform maintenance check")
            st.write("• Notify supervisor")
        elif result == 1:
            st.warning(" **MONITOR CLOSELY**")
            st.write("• Increase monitoring frequency")
            st.write("• Schedule maintenance soon")
        else:
            st.success(" **OPERATE NORMALLY**")
            st.write("• Continue normal operation")
            st.write("• Regular monitoring")

    with col3:
        st.write("**Input Summary**")
        p = prediction["parameters"]
        st.write(f" Air Temp: {p['air_temp']:.1f}K")
        st.write(f" Process Temp: {p['process_temp']:.1f}K")
        st.write(f" RPM: {p['rpm']:.0f}")
        st.write(f" Torque: {p['torque']:.1f}Nm")
        st.write(f" Wear: {p['wear']:.1f}min")
    
    # AI Explanation Section
    if "ai_explanation" in prediction:
        st.markdown("---")
        st.subheader(" AI-Powered Analysis")
        
        ai_exp = prediction["ai_explanation"]
        
        # Create expandable sections for AI analysis
        with st.expander(" Risk Assessment", expanded=True):
            st.write(ai_exp.get("risk_assessment", "Analysis not available"))
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            with st.expander(" Key Risk Factors"):
                st.write(ai_exp.get("key_factors", "Analysis not available"))
            with st.expander(" Immediate Actions"):
                st.write(ai_exp.get("immediate_actions", "No immediate actions required"))
        
        with col_a2:
            with st.expander(" Preventive Measures"):
                st.write(ai_exp.get("preventive_measures", "No specific preventive measures"))
            with st.expander(" Monitoring Recommendations"):
                st.write(ai_exp.get("monitoring", "Standard monitoring recommended"))
        
        with st.expander(" Maintenance Suggestions"):
            st.write(ai_exp.get("maintenance", "Regular maintenance recommended"))


def get_risk_level(probability):
    if probability >= 0.8:
        return {"label": "CRITICAL", "color": "red"}
    elif probability >= 0.6:
        return {"label": "HIGH", "color": "orange"}
    elif probability >= 0.3:
        return {"label": "MEDIUM", "color": "yellow"}
    else:
        return {"label": "LOW", "color": "green"}


if __name__ == "__main__":
    main()
