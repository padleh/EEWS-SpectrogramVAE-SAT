import pandas as pd
import numpy as np
import obspy
import obspy.core as oc
import h5py
import seisbench
import seisbench.util
import seisbench.data as sbd
from scipy import constants



def check_unit_type(dataframe, idx):
    """
    Check to see if the unit type is either mps2 or mps
    """
    unit_type = dataframe["trace_deconvolved_units"].iloc[idx]  # Get unit type
    return unit_type

def compute_peak_ground_motion(trace, unit_type):
    """
    Compute Peak Ground Motion parameters (PGA, PGV, PGD).
    
    If `unit_type` is "mps", differentiate to get acceleration.
    If `unit_type` is "mps2", integrate to get velocity and displacement.
    
    Returns: (PGA in g, PGV in m/s, PGD in m)
    """
    if unit_type == "mps":
        acc_trace = trace.copy().differentiate(method="gradient")  # Velocity → Acceleration
        velocity_trace = trace.copy()  # Already in velocity (m/s)
    else:
        acc_trace = trace.copy()  # Already acceleration (m/s²)
        velocity_trace = trace.copy().integrate(method="cumtrapz")  # Acceleration → Velocity
    
    displacement_trace = velocity_trace.copy().integrate(method="cumtrapz")  # Velocity → Displacement
    
    pga = max(np.abs(acc_trace.data)) 
    pgv = max(np.abs(velocity_trace.data))  # m/s
    pgd = max(np.abs(displacement_trace.data))  # m
    
    return pga, pgv, pgd

def compute_vs30(dataframe, idx):
    """Get Vs30 (m/s) from metadata for the given index."""
    return dataframe["station_vs_30_mps"].iloc[idx]

def compute_z25(dataframe, idx):
    """Compute depth to Vs = 2.5 km/s (Z2.5 in meters) from Vs30."""
    Vs30 = dataframe["station_vs_30_mps"].iloc[idx]
    Z10_m = np.exp(28.5) * ((Vs30**8 + 378.7**8)**(-0.4775))
    Z10_km = Z10_m / 1000
    Z25_km = 0.519 + 3.595 * Z10_km
    return Z25_km * 1000  # Convert to meters

def compute_arias_intensity(trace, unit_type):
    """
    Compute Arias Intensity (Ia in m/s).
    """
    if unit_type == "mps":
        acc_trace = trace.copy().differentiate(method="gradient") 
    else:
        acc_trace = trace.copy()
        
    squared_trace = acc_trace.copy()
    squared_trace.data = squared_trace.data**2  
    squared_trace.integrate(method="cumtrapz") 
    arias_intensity_3s = (np.pi / (2 * constants.g)) * squared_trace.data
    
    return max(arias_intensity_3s)  # Maximum Arias Intensity

def compute_cav(trace, unit_type):
    """
    Compute the Cumulative Absolute Velocity (CAV in m/s).
    """
    if unit_type == "mps":
        acc_trace = trace.copy().differentiate(method="gradient") 
    else:
        acc_trace = trace.copy()
        
    cav_trace = acc_trace.copy()
    cav_trace.integrate(method="cumtrapz")
    cav_trace.data = np.abs(cav_trace.data)
    cav_trace.integrate(method="cumtrapz")
    return max(cav_trace.data)

def compute_predominant_period(trace, unit_type):
    """
    Compute the Predominant Period (T_m in seconds).
    """
    if unit_type == "mps":
        acc_trace = trace.copy().differentiate(method="gradient") 
    else:
        acc_trace = trace.copy()

    dt = acc_trace.stats.delta  # Time step (sampling interval)
    n = len(acc_trace.data)
    freqs = np.fft.rfftfreq(n, d=dt)
    fourier_spectrum = np.fft.rfft(acc_trace.data)
    C_i = np.abs(fourier_spectrum)
    
    # Filter frequencies between 0.25 Hz and 20 Hz
    valid_indices = (freqs >= 0.25) & (freqs <= 20)
    filtered_freqs = freqs[valid_indices]
    filtered_Ci = C_i[valid_indices]
    T_m = np.sum((filtered_Ci**2) * (1 / filtered_freqs)) / np.sum(filtered_Ci**2)
    return T_m

def compute_significant_duration(trace, arias_intensity_3s, unit_type):
    """
    Compute the Significant Duration (D5-95 in seconds).
    """
    if unit_type == "mps":
        acc_trace = trace.copy().differentiate(method="gradient") 
    else:
        acc_trace = trace.copy()

    squared_trace = acc_trace.copy()
    squared_trace.data = squared_trace.data**2  
    squared_trace.integrate(method="cumtrapz") 
    arias_intensity_3s = (np.pi / (2 * constants.g)) * squared_trace.data
    
    arias_intensity_norm = arias_intensity_3s / np.max(arias_intensity_3s)  # Normalize
    time = np.linspace(0, 3, len(arias_intensity_norm))  # Generate time vector
    
    t5_index = np.where(arias_intensity_norm >= 0.05)[0][0]  # First 5% time index
    t95_index = np.where(arias_intensity_norm >= 0.95)[0][0]  # First 95% time index
    
    return time[t95_index] - time[t5_index]  # Significant duration D5-95

def best_trace_checker(trimmed_streams,idx):
    """
    Checking which trace has the highest peak ground motion (PGA or PGV)
    """
    Z_data = trimmed_streams[idx][0].data
    N_data = trimmed_streams[idx][1].data
    E_data = trimmed_streams[idx][2].data

    if Z_data.size == 0 or N_data.size == 0 or E_data.size == 0:
        return None
    else:        
        Z_Channels = max(np.abs(Z_data))
        N_Channels = max(np.abs(N_data))
        E_Channels = max(np.abs(E_data))

        channels = [Z_Channels, N_Channels, E_Channels]
        
        max_value = max(channels)

        # Check which channel has the maximum value and retrieve the corresponding data
        if max_value == Z_Channels:
            return trimmed_streams[idx][0].copy() # Data from Z_Channels (index 0)
        elif max_value == N_Channels:
            return trimmed_streams[idx][1].copy() # Data from N_Channels (index 1)
        elif max_value == E_Channels:
            return trimmed_streams[idx][2].copy()  # Data from E_Channels (index 2)
        else:
            return None


