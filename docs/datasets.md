# Datasets

## NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

**Source**: NASA Ames Prognostics Data Repository  
**Direct link**: https://ti.arc.nasa.gov/c/6/

### Description
Run-to-failure simulation data from turbofan engines with 4 subsets:
- **FD001**: Single operating condition, single fault mode
- **FD002**: Six operating conditions, single fault mode  
- **FD003**: Single operating condition, two fault modes
- **FD004**: Six operating conditions, two fault modes

### Files per subset
- `train_FD00X.txt` - Training data with full run-to-failure trajectories
- `test_FD00X.txt` - Test data (truncated before failure)
- `RUL_FD00X.txt` - True RUL values for test set

### Features
- 21 sensor measurements (temperature, pressure, speed, etc.)
- 3 operational settings
- Unit ID and cycle number

### Citation
```
Saxena, A., & Goebel, K. (2008). 
Turbofan Engine Degradation Simulation Data Set. 
NASA Ames Prognostics Data Repository.
```

## Numenta Anomaly Benchmark (NAB)

**Source**: https://github.com/numenta/NAB

### Description
Labeled time series data for streaming anomaly detection evaluation.

### Categories
- Real traffic data
- AWS server metrics
- Twitter volume
- Advertisement clicks
- Machine temperature sensors

### Citation
```
Lavin, A., & Ahmad, S. (2015). 
Evaluating Real-time Anomaly Detection Algorithms – the Numenta Anomaly Benchmark. 
arXiv:1510.03336
```

## Alternative: Manual Download

If automated download fails:

1. **C-MAPSS**: Visit https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ → Download "Turbofan Engine Degradation Simulation Data Set"
2. **NAB**: `git clone https://github.com/numenta/NAB.git data/raw/nab`
