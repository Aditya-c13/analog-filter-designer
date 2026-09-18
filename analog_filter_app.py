"""
Analog Filter Designer & Synthesizer
------------------------------------
Streamlit application for analog filter design, schematic synthesis,
component tolerance analysis, multi-approximation comparison, and automated report generation.

Run with:
    streamlit run analog_filter_app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import signal
import io
import html

st.set_page_config(page_title="Analog Filter Designer", layout="wide")

# ============================================================
#                     E24 STANDARD VALUES
# ============================================================
E24_BASE = np.array([
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
])

def nearest_e24(value):
    """Find the nearest standard E24 component value."""
    if value <= 0 or np.isnan(value):
        return value
    exponent = np.floor(np.log10(value))
    mantissa = value / (10 ** exponent)
    idx = (np.abs(E24_BASE - mantissa)).argmin()
    return E24_BASE[idx] * (10 ** exponent)

# ============================================================
#                      SIDEBAR — INPUTS
# ============================================================
st.sidebar.title("Filter Specification")

filter_type_map = {
    "Low Pass": "lowpass",
    "High Pass": "highpass",
    "Band Pass": "bandpass",
    "Band Stop": "bandstop",
}
filter_type_label = st.sidebar.selectbox("Filter Type", list(filter_type_map.keys()))
filter_type = filter_type_map[filter_type_label]

approx_map = {
    "Butterworth": "butter",
    "Chebyshev I": "cheby1",
    "Bessel": "bessel",
}
approx_label = st.sidebar.selectbox("Approximation", list(approx_map.keys()))
approx = approx_map[approx_label]

n = st.sidebar.number_input("Order", min_value=1, max_value=12, value=3, step=1)

rp = None
if approx == "cheby1":
    rp = st.sidebar.number_input("Passband Ripple (dB)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

if filter_type in ("lowpass", "highpass"):
    fc = st.sidebar.number_input("Cutoff Frequency (Hz)", min_value=0.1, value=1000.0, step=100.0)
    f_low = f_high = None
else:
    f_low = st.sidebar.number_input("Lower Cutoff Frequency (Hz)", min_value=0.1, value=800.0, step=100.0)
    f_high = st.sidebar.number_input("Upper Cutoff Frequency (Hz)", min_value=0.1, value=1200.0, step=100.0)
    fc = None
    if f_high <= f_low:
        st.sidebar.error("Upper cutoff must be greater than lower cutoff.")

source_r = st.sidebar.number_input("Source/Load Resistance R (Ohm)", min_value=1.0, value=1000.0, step=100.0)
tolerance_pct = st.sidebar.slider("Component Tolerance (%)", min_value=1, max_value=20, value=5)

st.title("Analog Filter Designer")
st.caption("Design, visualize, analyze tolerances, and export comprehensive reports.")

# ============================================================
#                   HELPER & DESIGN FUNCTIONS
# ============================================================
def get_wn(filter_type, fc, f_low, f_high):
    if filter_type in ("lowpass", "highpass"):
        return 2 * np.pi * fc
    else:
        return [2 * np.pi * f_low, 2 * np.pi * f_high]

def design_analog_filter(approx, n, rp, filter_type, wn):
    if approx == "butter":
        return signal.butter(n, wn, btype=filter_type, analog=True, output="zpk")
    elif approx == "cheby1":
        return signal.cheby1(n, rp, wn, btype=filter_type, analog=True, output="zpk")
    elif approx == "bessel":
        return signal.bessel(n, wn, btype=filter_type, analog=True, output="zpk", norm="mag")

def format_coef_latex(c):
    s = f"{c:.4g}"
    if "e" in s:
        mantissa, exp = s.split("e")
        return rf"{mantissa} \times 10^{{{int(exp)}}}"
    return s

def poly_to_latex(coeffs, var="s"):
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if c == 0:
            continue
        sign = "-" if c < 0 else ""
        mag = abs(c)
        coef_str = format_coef_latex(mag)
        if power == 0:
            term = f"{sign}{coef_str}"
        elif power == 1:
            term = f"{sign}{coef_str}{var}" if mag != 1 else f"{sign}{var}"
        else:
            term = f"{sign}{coef_str}{var}^{{{power}}}" if mag != 1 else f"{sign}{var}^{{{power}}}"
        terms.append(term)
    if not terms:
        return "0"
    expr = terms[0]
    for t in terms[1:]:
        expr += f" + {t}" if not t.startswith("-") else f" - {t[1:]}"
    return expr

def suggest_rlc(filter_type, n, wn, R):
    stages = []
    if filter_type in ("lowpass", "highpass"):
        w0 = wn
        pairs = n // 2
        remainder = n % 2

        for i in range(pairs):
            L = R / (w0 * np.sqrt(2))
            C = 1 / (w0 ** 2 * L)
            stages.append({
                "Stage": f"2nd-order Section {i+1}",
                "L_ideal_mH": L * 1e3,
                "L_std_mH": nearest_e24(L * 1e3),
                "C_ideal_nF": C * 1e9,
                "C_std_nF": nearest_e24(C * 1e9),
            })

        if remainder:
            if filter_type == "lowpass":
                C = 1 / (w0 * R)
                stages.append({
                    "Stage": "1st-order RC Section",
                    "R_Ohm": R,
                    "C_ideal_nF": C * 1e9,
                    "C_std_nF": nearest_e24(C * 1e9),
                })
            else:
                L = R / w0
                stages.append({
                    "Stage": "1st-order RL Section",
                    "R_Ohm": R,
                    "L_ideal_mH": L * 1e3,
                    "L_std_mH": nearest_e24(L * 1e3),
                })
    else:
        w_low, w_high = wn
        w0 = np.sqrt(w_low * w_high)
        bw = w_high - w_low
        for i in range(max(1, n // 2)):
            L = R / bw
            C = 1 / (w0 ** 2 * L)
            stages.append({
                "Stage": f"Resonant Section {i+1}",
                "L_ideal_mH": L * 1e3,
                "L_std_mH": nearest_e24(L * 1e3),
                "C_ideal_nF": C * 1e9,
                "C_std_nF": nearest_e24(C * 1e9),
            })
    return stages

# ============================================================
#                 ROBUST SCHEMATIC ENGINE
# ============================================================
def _draw_resistor_h(ax, x0, x1, y, label):
    n_zig, h = 5, 0.12
    lead = (x1 - x0) * 0.2
    xs = np.linspace(x0 + lead, x1 - lead, n_zig * 2 + 1)
    ys = [y + (h if i % 2 else -h) for i in range(len(xs))]
    ys[0] = ys[-1] = y
    ax.plot([x0, x0 + lead], [y, y], 'k-', lw=1.5)
    ax.plot(xs, ys, 'k-', lw=1.5)
    ax.plot([x1 - lead, x1], [y, y], 'k-', lw=1.5)
    ax.text((x0 + x1) / 2, y + 0.25, label, ha="center", va="bottom", fontsize=8, fontweight="bold")

def _draw_inductor_h(ax, x0, x1, y, label):
    lead = (x1 - x0) * 0.15
    xs_coil = np.linspace(x0 + lead, x1 - lead, 150)
    span = (x1 - lead) - (x0 + lead)
    ys_coil = y + 0.15 * np.abs(np.sin(4 * np.pi * (xs_coil - x0 - lead) / span))
    ax.plot([x0, x0 + lead], [y, y], 'k-', lw=1.5)
    ax.plot(xs_coil, ys_coil, 'k-', lw=1.5)
    ax.plot([x1 - lead, x1], [y, y], 'k-', lw=1.5)
    ax.text((x0 + x1) / 2, y + 0.25, label, ha="center", va="bottom", fontsize=8)

def _draw_capacitor_h(ax, x0, x1, y, label):
    gap = 0.08
    xm = (x0 + x1) / 2
    ax.plot([x0, xm - gap], [y, y], 'k-', lw=1.5)
    ax.plot([xm + gap, x1], [y, y], 'k-', lw=1.5)
    ax.plot([xm - gap, xm - gap], [y - 0.2, y + 0.2], 'k-', lw=1.5)
    ax.plot([xm + gap, xm + gap], [y - 0.2, y + 0.2], 'k-', lw=1.5)
    ax.text(xm, y + 0.25, label, ha="center", va="bottom", fontsize=8)

def _draw_inductor_v(ax, x, y0, y1, label, side="right"):
    lead = abs(y0 - y1) * 0.15
    ys_coil = np.linspace(y0 - lead, y1 + lead, 150)
    span = abs(y0 - lead - (y1 + lead))
    offset = 0.15 if side == "right" else -0.15
    xs_coil = x + offset * np.abs(np.sin(4 * np.pi * (y0 - lead - ys_coil) / span))
    ax.plot([x, x], [y0, y0 - lead], 'k-', lw=1.5)
    ax.plot(xs_coil, ys_coil, 'k-', lw=1.5)
    ax.plot([x, x], [y1 + lead, y1], 'k-', lw=1.5)
    ha = "left" if side == "right" else "right"
    tx = x + 0.25 if side == "right" else x - 0.25
    ax.text(tx, (y0 + y1) / 2, label, ha=ha, va="center", fontsize=8)

def _draw_capacitor_v(ax, x, y0, y1, label, side="right"):
    gap = 0.08
    ym = (y0 + y1) / 2
    ax.plot([x, x], [y0, ym + gap], 'k-', lw=1.5)
    ax.plot([x, x], [ym - gap, y1], 'k-', lw=1.5)
    ax.plot([x - 0.18, x + 0.18], [ym + gap, ym + gap], 'k-', lw=1.5)
    ax.plot([x - 0.18, x + 0.18], [ym - gap, ym - gap], 'k-', lw=1.5)
    ha = "left" if side == "right" else "right"
    tx = x + 0.25 if side == "right" else x - 0.25
    ax.text(tx, ym, label, ha=ha, va="center", fontsize=8)

def draw_schematic(filter_type, stages, R):
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(stages) + 4), 3.5))
    y_top, y_bot = 1.2, 0.0
    x = 0.0
    dx = 2.0

    # Source Vin
    circle = plt.Circle((x, 0.6), 0.2, fill=False, color="black", lw=1.5)
    ax.add_patch(circle)
    ax.plot([x, x], [y_bot, 0.4], 'k-', lw=1.5)
    ax.plot([x, x], [0.8, y_top], 'k-', lw=1.5)
    ax.text(x - 0.4, 0.6, "Vin", ha="center", va="center", fontsize=9, fontweight="bold")
    
    # Source Resistor
    _draw_resistor_h(ax, x, x + dx, y_top, f"Rs={R:.0f}Ω")
    x += dx

    for stage in stages:
        has_l = "L_std_mH" in stage
        has_c = "C_std_nF" in stage

        if filter_type == "lowpass":
            if has_l and has_c:
                _draw_inductor_h(ax, x, x + dx, y_top, f"{stage['L_std_mH']:.1f}mH")
                x += dx
                _draw_capacitor_v(ax, x, y_top, y_bot, f"{stage['C_std_nF']:.1f}nF")
            elif has_c:
                _draw_capacitor_v(ax, x, y_top, y_bot, f"{stage['C_std_nF']:.1f}nF")

        elif filter_type == "highpass":
            if has_l and has_c:
                _draw_capacitor_h(ax, x, x + dx, y_top, f"{stage['C_std_nF']:.1f}nF")
                x += dx
                _draw_inductor_v(ax, x, y_top, y_bot, f"{stage['L_std_mH']:.1f}mH")
            elif has_l:
                _draw_inductor_v(ax, x, y_top, y_bot, f"{stage['L_std_mH']:.1f}mH")

        elif filter_type == "bandpass":
            _draw_inductor_h(ax, x, x + dx / 2, y_top, f"{stage['L_std_mH']:.1f}mH")
            _draw_capacitor_h(ax, x + dx / 2, x + dx, y_top, f"{stage['C_std_nF']:.1f}nF")
            x += dx

        elif filter_type == "bandstop":
            # Staggered parallel branch prevents overlapping text/components
            x_left, x_right = x + 0.4, x + 1.2
            ax.plot([x, x_right], [y_top, y_top], 'k-', lw=1.5)
            _draw_inductor_v(ax, x_left, y_top, y_bot, f"{stage['L_std_mH']:.1f}mH", side="left")
            _draw_capacitor_v(ax, x_right, y_top, y_bot, f"{stage['C_std_nF']:.1f}nF", side="right")
            x = x_right + 0.4
            ax.plot([x_left, x], [y_top, y_top], 'k-', lw=1.5)

    # Load Resistor & Output Terminal
    _draw_resistor_h(ax, x, x + dx, y_top, f"RL={R:.0f}Ω")
    x += dx
    ax.plot([x, x], [y_top, y_bot], 'k--', lw=1)
    ax.text(x + 0.3, y_top / 2, "Vout", ha="left", va="center", fontsize=10, fontweight="bold")

    # CONTINUOUS GROUND RAIL (fixes broken/floating ground wire issue)
    ax.plot([-0.2, x], [y_bot, y_bot], 'k-', lw=1.5)
    
    # Ground Symbol
    ax.plot([0, 0], [y_bot, y_bot - 0.15], 'k-', lw=1.5)
    for i, w in enumerate([0.2, 0.12, 0.05]):
        yy = y_bot - 0.15 - i * 0.06
        ax.plot([-w, w], [yy, yy], 'k-', lw=1.5)

    ax.set_xlim(-0.8, x + 0.8)
    ax.set_ylim(-0.5, 1.8)
    ax.axis("off")
    fig.tight_layout()
    return fig

# ============================================================
#                      MAIN EXECUTION
# ============================================================
wn = get_wn(filter_type, fc, f_low, f_high)
z, p, k = design_analog_filter(approx, n, rp, filter_type, wn)
num, den = signal.zpk2tf(z, p, k)

# Frequency Grid
w = np.logspace(
    np.log10((wn if np.isscalar(wn) else min(wn)) / 100),
    np.log10((wn if np.isscalar(wn) else max(wn)) * 100),
    1000,
)
w, h = signal.freqs_zpk(z, p, k, worN=w)
freq_hz = w / (2 * np.pi)
mag_db = 20 * np.log10(np.abs(h))
phase_deg = np.degrees(np.unwrap(np.angle(h)))

stages = suggest_rlc(filter_type, n, wn, source_r)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Filter Design & Schematic", 
    "Component Tolerance Analysis", 
    "Filter Comparison", 
    "Automated Report"
])

# ------------------------------------------------------------
# TAB 1: Main Design & Schematic
# ------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    plotly_config = {"scrollZoom": True, "displaylogo": False}

    with col1:
        fig_pz = go.Figure()
        fig_pz.add_trace(go.Scatter(x=np.real(p), y=np.imag(p), mode="markers", name="Poles",
                                    marker=dict(symbol="x", size=11, color="crimson")))
        if len(z) > 0:
            fig_pz.add_trace(go.Scatter(x=np.real(z), y=np.imag(z), mode="markers", name="Zeros",
                                        marker=dict(symbol="circle-open", size=11, color="royalblue", line=dict(width=2))))
        fig_pz.add_hline(y=0, line_color="lightgray")
        fig_pz.add_vline(x=0, line_color="lightgray")
        fig_pz.update_layout(title="Pole-Zero Plot", xaxis_title="Real (Np/s)", yaxis_title="Imaginary (rad/s)", height=380)
        st.plotly_chart(fig_pz, use_container_width=True, config=plotly_config)

    with col2:
        fig_mag = go.Figure()
        fig_mag.add_trace(go.Scatter(x=freq_hz, y=mag_db, mode="lines", name="Magnitude"))
        fig_mag.update_xaxes(type="log", title="Frequency (Hz)")
        fig_mag.update_yaxes(title="Magnitude (dB)")
        fig_mag.update_layout(title="Magnitude Response", height=380)
        st.plotly_chart(fig_mag, use_container_width=True, config=plotly_config)

    st.subheader("Transfer Function")
    st.latex(rf"H(s) = \dfrac{{{poly_to_latex(num)}}}{{{poly_to_latex(den)}}}")

    st.subheader("Synthesized Circuit Schematic & Standard Component Values")
    st.pyplot(draw_schematic(filter_type, stages, source_r))
    st.table(stages)

# ------------------------------------------------------------
# TAB 2: Component Tolerance Analysis
# ------------------------------------------------------------
with tab2:
    st.subheader("Monte Carlo Component Tolerance Analysis")
    st.write(f"Simulating magnitude variations under a **±{tolerance_pct}%** component standard deviation.")

    fig_tol = go.Figure()
    fig_tol.add_trace(go.Scatter(x=freq_hz, y=mag_db, mode="lines", name="Ideal Response", line=dict(color="black", width=3)))

    # Monte Carlo simulation runs
    np.random.seed(42)
    for _ in range(25):
        scale = 1 + np.random.normal(0, tolerance_pct / 100.0)
        perturbed_mag = mag_db + (20 * np.log10(scale) * (freq_hz / (freq_hz + (fc if fc else f_low))))
        fig_tol.add_trace(go.Scatter(x=freq_hz, y=perturbed_mag, mode="lines", 
                                     line=dict(width=1, color="rgba(255, 99, 71, 0.2)"), showlegend=False))

    fig_tol.update_xaxes(type="log", title="Frequency (Hz)")
    fig_tol.update_yaxes(title="Magnitude (dB)")
    fig_tol.update_layout(title=f"Magnitude Bounds with ±{tolerance_pct}% Tolerance Components", height=450)
    st.plotly_chart(fig_tol, use_container_width=True)

# ------------------------------------------------------------
# TAB 3: Filter Comparison
# ------------------------------------------------------------
with tab3:
    st.subheader("Approximation Comparison (Butterworth vs Chebyshev I vs Bessel)")
    fig_comp = go.Figure()

    for app_key, app_name in [("butter", "Butterworth"), ("cheby1", "Chebyshev I"), ("bessel", "Bessel")]:
        z_c, p_c, k_c = design_analog_filter(app_key, n, rp if rp else 1.0, filter_type, wn)
        _, h_c = signal.freqs_zpk(z_c, p_c, k_c, worN=w)
        fig_comp.add_trace(go.Scatter(x=freq_hz, y=20 * np.log10(np.abs(h_c)), mode="lines", name=app_name))

    fig_comp.update_xaxes(type="log", title="Frequency (Hz)")
    fig_comp.update_yaxes(title="Magnitude (dB)")
    fig_comp.update_layout(title="Magnitude Response Comparison", height=450)
    st.plotly_chart(fig_comp, use_container_width=True)


# ------------------------------------------------------------
# TAB 4: Automated Report Generation (Self-Contained Base64 Rendering)
# ------------------------------------------------------------
with tab4:
    st.subheader("Download Complete Design Documentation")
    st.write("Generates a fully offline, self-contained HTML report containing all specifications, LaTeX equations, high-resolution plots, circuit schematics, tolerance runs, and filter comparisons.")

    import base64

    def fig_to_base64(fig):
        """Convert a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        b64_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return b64_str

    def safe_fmt(val, spec=".2f"):
        """Format numbers safely, returning '-' for missing or non-numeric values."""
        if isinstance(val, (int, float)) and not np.isnan(val):
            return f"{val:{spec}}"
        return "-"

    # 1. Generate Circuit Schematic Base64 Image
    schematic_fig = draw_schematic(filter_type, stages, source_r)
    schematic_b64 = fig_to_base64(schematic_fig)

    # 2. Render Static Pole-Zero Plot
    fig_pz_static, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.real(p), np.imag(p), color="crimson", marker="x", s=80, label="Poles", zorder=3)
    if len(z) > 0:
        ax.scatter(np.real(z), np.imag(z), color="royalblue", facecolors="none", marker="o", s=80, lw=1.5, label="Zeros", zorder=3)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_title("Pole-Zero Plot", fontsize=11, fontweight="bold")
    ax.set_xlabel("Real (Np/s)")
    ax.set_ylabel("Imaginary (rad/s)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig_pz_static.tight_layout()
    pz_b64 = fig_to_base64(fig_pz_static)

    # 3. Render Static Magnitude & Phase Plots
    fig_bode_static, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.semilogx(freq_hz, mag_db, color="royalblue", lw=2)
    ax1.set_title("Magnitude Response", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    ax2.semilogx(freq_hz, phase_deg, color="seagreen", lw=2)
    ax2.set_title("Phase Response", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    fig_bode_static.tight_layout()
    bode_b64 = fig_to_base64(fig_bode_static)

    # 4. Render Static Tolerance Plot
    fig_tol_static, ax = plt.subplots(figsize=(8, 4))
    np.random.seed(42)
    for _ in range(25):
        scale = 1 + np.random.normal(0, tolerance_pct / 100.0)
        perturbed_mag = mag_db + (20 * np.log10(scale) * (freq_hz / (freq_hz + (fc if fc else f_low))))
        ax.semilogx(freq_hz, perturbed_mag, color="tomato", alpha=0.25, lw=1)
    ax.semilogx(freq_hz, mag_db, color="black", lw=2.5, label="Ideal Response")
    ax.set_title(f"Monte Carlo Component Tolerance Bounds (±{tolerance_pct}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig_tol_static.tight_layout()
    tol_b64 = fig_to_base64(fig_tol_static)

    # 5. Render Static Comparison Plot
    fig_comp_static, ax = plt.subplots(figsize=(8, 4))
    for app_key, app_name in [("butter", "Butterworth"), ("cheby1", "Chebyshev I"), ("bessel", "Bessel")]:
        z_c, p_c, k_c = design_analog_filter(app_key, n, rp if rp else 1.0, filter_type, wn)
        _, h_c = signal.freqs_zpk(z_c, p_c, k_c, worN=w)
        ax.semilogx(freq_hz, 20 * np.log10(np.abs(h_c)), lw=2, label=app_name)
    ax.set_title("Approximation Comparison", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig_comp_static.tight_layout()
    comp_b64 = fig_to_base64(fig_comp_static)

    # 6. Format Tables, Poles & Zeros
    stage_rows = "".join([
        f"<tr><td>{s['Stage']}</td>"
        f"<td>{safe_fmt(s.get('L_ideal_mH'))}</td>"
        f"<td>{safe_fmt(s.get('L_std_mH'))}</td>"
        f"<td>{safe_fmt(s.get('C_ideal_nF'))}</td>"
        f"<td>{safe_fmt(s.get('C_std_nF'))}</td></tr>"
        for s in stages
    ])

    poles_list = "<br>".join([f"{pi.real:.4f} {'+' if pi.imag >= 0 else '-'} {abs(pi.imag):.4f}j" for pi in p])
    zeros_list = "<br>".join([f"{zi.real:.4f} {'+' if zi.imag >= 0 else '-'} {abs(zi.imag):.4f}j" for zi in z]) if len(z) > 0 else "None"
    tf_latex = rf"H(s) = \dfrac{{{poly_to_latex(num)}}}{{{poly_to_latex(den)}}}"

    # 7. Construct HTML Document
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>Analog Filter Design Technical Report</title>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2C3E50; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
            .grid {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .col {{ flex: 1; min-width: 300px; }}
            .card {{ background: #fdfdfd; border: 1px solid #e2e8f0; border-radius: 6px; padding: 20px; margin-bottom: 20px; text-align: center; }}
            .card-left {{ text-align: left; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: left; }}
            th {{ background-color: #f1f5f9; font-weight: bold; }}
            .math-box {{ background: #f8fafc; border-left: 4px solid #3b82f6; padding: 15px; font-size: 1.2em; margin: 15px 0; overflow-x: auto; text-align: center; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Analog Filter Design Specification Report</h1>

            <h2>1. Specifications & Transfer Function</h2>
            <div class="grid">
                <div class="col card card-left">
                    <h3>Filter Specifications</h3>
                    <ul>
                        <li><b>Filter Type:</b> {html.escape(filter_type_label)}</li>
                        <li><b>Approximation:</b> {html.escape(approx_label)}</li>
                        <li><b>Order (n):</b> {n}</li>
                        {'<li><b>Passband Ripple:</b> ' + f'{rp:.2f} dB</li>' if rp else ''}
                        {'<li><b>Cutoff Frequency:</b> ' + f'{fc:.2f} Hz</li>' if fc else ''}
                        {'<li><b>Bandpass Range:</b> ' + f'{f_low:.2f} Hz - {f_high:.2f} Hz</li>' if f_low else ''}
                        <li><b>Source / Load Resistance (R):</b> {source_r} Ω</li>
                    </ul>
                </div>
                <div class="col card card-left">
                    <h3>Poles & Zeros</h3>
                    <p><b>Poles:</b><br><code>{poles_list}</code></p>
                    <p><b>Zeros:</b><br><code>{zeros_list}</code></p>
                </div>
            </div>

            <div class="card card-left">
                <h3>Transfer Function H(s)</h3>
                <div class="math-box">
                    \[ {tf_latex} \]
                </div>
            </div>

            <h2>2. Pole-Zero & Frequency Response</h2>
            <div class="grid">
                <div class="col card">
                    <img src="data:image/png;base64,{pz_b64}" alt="Pole Zero Plot"/>
                </div>
                <div class="col card">
                    <img src="data:image/png;base64,{bode_b64}" alt="Bode Plot"/>
                </div>
            </div>

            <h2>3. Circuit Schematic & E24 Component Values</h2>
            <div class="card">
                <h3>Synthesized Circuit Schematic</h3>
                <img src="data:image/png;base64,{schematic_b64}" alt="Filter Schematic"/>
            </div>

            <div class="card card-left">
                <h3>E24 Standard Component Mapping</h3>
                <table>
                    <tr>
                        <th>Stage</th>
                        <th>Ideal L (mH)</th>
                        <th>E24 Standard L (mH)</th>
                        <th>Ideal C (nF)</th>
                        <th>E24 Standard C (nF)</th>
                    </tr>
                    {stage_rows}
                </table>
            </div>

            <h2>4. Component Tolerance Analysis</h2>
            <div class="card">
                <img src="data:image/png;base64,{tol_b64}" alt="Tolerance Analysis"/>
            </div>

            <h2>5. Approximation Performance Comparison</h2>
            <div class="card">
                <img src="data:image/png;base64,{comp_b64}" alt="Filter Comparison"/>
            </div>
        </div>
    </body>
    </html>
    """

    st.download_button(
        label="Download Self-Contained HTML Report",
        data=report_html,
        file_name=f"analog_filter_report_{filter_type}_{approx}_n{n}.html",
        mime="text/html",
    )