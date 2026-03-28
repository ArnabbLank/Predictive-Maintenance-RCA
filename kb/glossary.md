# C-MAPSS Glossary — Sensor & Setting Definitions

> Quick reference for the 26 columns in C-MAPSS data.

---

## Index Columns

| Column | Name | Description |
|--------|------|-------------|
| 1 | `unit_id` | Engine unit number (1–100 in FD001) |
| 2 | `cycle` | Operating cycle number (time index) |

---

## Operational Settings (3)

| Column | Name | Symbol | Description | Unit |
|--------|------|--------|-------------|------|
| 3 | `op_setting_1` | Altitude | Flight altitude / condition | ft (×1000) |
| 4 | `op_setting_2` | Mach | Mach number | — |
| 5 | `op_setting_3` | TRA | Throttle resolver angle | deg |

> In FD001 and FD003, there is only **1 operating condition** (all settings are near-constant).
> In FD002 and FD004, there are **6 operating conditions**.

---

## Sensor Measurements (21)

| Column | Name | Symbol | Description | Typical behavior at failure |
|--------|------|--------|-------------|----------------------------|
| 6 | `sensor_1` | T2 | Total temp at fan inlet | Near constant (FD001) |
| 7 | `sensor_2` | T2 | Total temp at fan inlet | Slight drift |
| 8 | `sensor_3` | T24 | Total temp at LPC outlet | ↑ Increases |
| 9 | `sensor_4` | T30 | Total temp at HPC outlet | ↑ Increases significantly |
| 10 | `sensor_5` | T50 | Total temp at LPT outlet | Near constant (FD001) |
| 11 | `sensor_6` | P2 | Pressure at fan inlet | Near constant (FD001) |
| 12 | `sensor_7` | P15 | Total pressure in bypass-duct | ↑↓ Varies |
| 13 | `sensor_8` | Ps30 | Static pressure at HPC outlet | ↓ Decreases |
| 14 | `sensor_9` | phi | Ratio of fuel flow to Ps30 | ↑ Increases |
| 15 | `sensor_10` | — | — | Near constant (drop) |
| 16 | `sensor_11` | P30 | Total pressure at HPC outlet | ↓ Decreases |
| 17 | `sensor_12` | Nf | Physical fan speed (rpm) | ↓ Slight decrease |
| 18 | `sensor_13` | Nc | Physical core speed (rpm) | ↑ Increases |
| 19 | `sensor_14` | epr | Engine pressure ratio | ↓ Decreases |
| 20 | `sensor_15` | Ps30 | Static pressure at HPC outlet | ↓ Decreases |
| 21 | `sensor_16` | — | — | Near constant (drop) |
| 22 | `sensor_17` | htBleed | Bleed enthalpy | ↓ Decreases |
| 23 | `sensor_18` | — | — | Near constant (drop) |
| 24 | `sensor_19` | — | — | Near constant (drop) |
| 25 | `sensor_20` | BPR | Bypass ratio | ↓ Decreases |
| 26 | `sensor_21` | farB | Burner fuel-air ratio | ↑ Increases |

---

## Commonly Dropped Sensors (FD001)

These sensors have near-zero variance in FD001 and carry no predictive signal:
- `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`

**Remaining 14 informative sensors** are used for modeling.

---

## RUL Labeling

- **Raw RUL** = `max_cycle(unit) - current_cycle`
- **Capped RUL** = `min(raw_rul, cap)` where cap is typically **125 cycles**
  - Rationale: early cycles are "healthy" and RUL differences are meaningless
  - Creates a piece-wise linear target: flat at 125, then linear decay to 0

---

## NASA Scoring Function

The standard asymmetric scoring function penalizes late predictions more:

$$
S = \sum_{i=1}^{n} s_i, \quad s_i = \begin{cases} e^{-d_i/13} - 1, & d_i < 0 \text{ (early)} \\ e^{d_i/10} - 1, & d_i \geq 0 \text{ (late)} \end{cases}
$$

where $d_i = \hat{RUL}_i - RUL_i$ (prediction error).

Late predictions are penalized more because they could lead to in-service failures.
