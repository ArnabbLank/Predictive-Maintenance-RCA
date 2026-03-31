# Turbofan Engine Fault Tree — Knowledge Base

> Student-curated fault tree for the C-MAPSS turbofan engine simulation.
> This is a simplified reference — not a real maintenance manual.

---

## High Pressure Compressor (HPC) Degradation

### Affected sensors
- **sensor_2** (T2): Total temperature at fan inlet — may shift as HPC efficiency drops
- **sensor_3** (T24): Total temperature at LPC outlet — rises with compressor inefficiency
- **sensor_4** (T30): Total temperature at HPC outlet — direct indicator of HPC health
- **sensor_7** (T50): Total temperature at LPT outlet — affected downstream
- **sensor_8** (P15): Total pressure in bypass-duct — changes with flow redistribution
- **sensor_9** (P2): Total pressure at fan inlet — inlet condition reference
- **sensor_11** (P30): Total pressure at HPC outlet — direct HPC health indicator
- **sensor_12** (Nf): Physical fan speed — compensatory speed changes
- **sensor_13** (Nc): Physical core speed — core responds to HPC degradation
- **sensor_14** (epr): Engine pressure ratio — overall engine efficiency metric
- **sensor_15** (Ps30): Static pressure at HPC outlet — direct degradation signal
- **sensor_17** (htBleed): Bleed enthalpy — changes with turbine degradation
- **sensor_20** (BPR): Bypass ratio — shifts as core degrades
- **sensor_21** (farB): Burner fuel-air ratio — fuel control compensates

### Failure progression
1. **Early stage**: Slight efficiency loss in HPC; Nc and P30 show subtle trends.
2. **Mid stage**: Temperature margins shrink. T30 and Ps30 diverge from baseline.
3. **Late stage**: Compressor surge risk. Multiple sensors show rapid divergence.
4. **Failure**: Engine shutdown due to insufficient pressure ratio or temperature exceedance.

### Maintenance recommendations
- **RUL > 80 cycles**: Continue monitoring. Log sensor trends.
- **RUL 30–80 cycles**: Schedule borescope inspection of HPC blades.
- **RUL < 30 cycles**: Immediate maintenance action — compressor wash or blade replacement.
- **High uncertainty**: Do NOT rely on automated prediction. Escalate to expert review.

---

## Fan Degradation (FD003/FD004 fault mode)

### Affected sensors
- **sensor_2** (T2): Fan inlet temperature — baseline reference
- **sensor_8** (P15): Bypass-duct pressure — primary fan health indicator
- **sensor_12** (Nf): Fan speed — direct measurement
- **sensor_20** (BPR): Bypass ratio — primary indicator of fan balance

### Failure progression
1. **Early stage**: Subtle BPR and Nf deviations.
2. **Mid stage**: Vibration indicators (not in sensor set) would trigger alerts.
3. **Late stage**: Significant thrust loss, fan imbalance.

---

## Operating Conditions (FD002/FD004)

### Operational settings
- **op_setting_1**: Altitude (flight level) — affects all temperatures and pressures
- **op_setting_2**: Mach number — affects inlet conditions
- **op_setting_3**: Throttle resolver angle (TRA) — engine power demand

### Impact on predictions
- Multiple operating conditions (6 in FD002/FD004) create different sensor baselines
- Models must normalize for operating conditions or learn them implicitly
- Regime-based normalization can improve prediction accuracy

---

## Sensor Groups for Quick Reference

| Group | Sensors | What it measures |
|-------|---------|-----------------|
| Temperatures | sensor_2, 3, 4, 7 | Thermal state through engine |
| Pressures | sensor_8, 9, 11, 15 | Pressure ratios across components |
| Speeds | sensor_12, 13 | Fan and core shaft speeds |
| Efficiency | sensor_14, 20, 21 | Overall engine performance |
| Bleed/Flow | sensor_17 | Bleed system health |

---

## When to Escalate (Copilot Decision Guide)

### Escalate (do NOT auto-recommend) when:
1. Prediction uncertainty (std) exceeds 20 cycles
2. Multiple top sensors are unusual (not typical degradation pattern)
3. Operating conditions are outside the training distribution
4. RUL prediction is near zero but uncertainty is high
5. Two consecutive predictions for the same engine diverge significantly

### Safe to auto-recommend when:
1. Uncertainty is low (std < 10 cycles)
2. Top sensors match known fault patterns (above)
3. Prediction is consistent with recent trend
