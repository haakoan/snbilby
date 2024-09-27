import bilby
import numpy as np

label = "CoreCollapse"


import numpy as np

# Define R_PNS_fit function
def R_PNS_fit(t, A, k, x0, B, C):
    return A * np.exp(-k * (t - x0)) + B * (t - x0)**2 + C * (t - x0)**3

# Define R_sh_fit function
def R_sh_fit(t, alpha, mu, sigma, beta, gamma):
    return alpha * np.exp(-((t - mu)**2) / (2 * sigma**2)) + beta * np.exp(gamma * t)

# Define M_PNS_fit function
def M_PNS_fit(t, a1, b1):
    return a1 * t + b1

# Define M_sh_fit function
def M_sh_fit(t, a2, b2):
    return a2 * t + b2

# Define P_c_fit function
def P_c_fit(t, a3, b3):
    return a3 * t + b3

# Define rho_c_fit function
def rho_c_fit(t, a4, b4):
    return a4 * t + b4

def mode_2f(M_shock, R_shock):
    x = np.sqrt(M_shock) / (R_shock**3)
    a = 0
    b = 1.410e-5
    c = -4.23e-6
    d = 0
    return a + b * x + c * x**2 + d * x**3

def mode_2p1(M_shock, R_shock):
    x = np.sqrt(M_shock) / (R_shock**3)
    a = 0
    b = 2.205e-5
    c = 4.63e-6
    d = 0
    return a + b * x + c * x**2 + d * x**3

def mode_2p2(M_shock, R_shock):
    x = np.sqrt(M_shock) / (R_shock**3)
    a = 0
    b = 4.02e-5
    c = 7.4e-6
    d = 0
    return a + b * x + c * x**2 + d * x**3

def mode_2p3(M_shock, R_shock):
    x = np.sqrt(M_shock) / (R_shock**3)
    a = 0
    b = 6.21e-5
    c = -1.9e-6
    d = 0
    return a + b * x + c * x**2 + d * x**3

def mode_2g1(M_pns, R_pns):
    x = M_pns / (R_pns**2)
    a = 0
    b = 8.67e-5
    c = -5.19e-6
    d = 0
    return a + b * x + c * x**2 + d * x**3

def mode_2g2(M_pns, R_pns):
    x = M_pns / (R_pns**2)
    a = 0
    b = 5.88e-5
    c = -8.62e-6
    d = 4.67e-9
    return a + b * x + c * x**2 + d * x**3

def mode_2g3(M_shock, R_shock, P_c, rho_c):
    x = np.sqrt(M_shock) / (R_shock**3) * (P_c / rho_c**2.5)
    a = 905
    b = -7.99e-5
    c = -1.1e-3
    d = 0
    return a + b * x + c * x**2 + d * x**3

def ccsn_model(f, duration, sampling_rate, modes_to_include, A, k, x0, B, C, alpha, mu, sigma, beta, gamma, 
                        a1, b1, a2, b2, a3, b3, a4, b4,phase_sigma,
                        A1=1,A2=1,A3=1,A4=1,A5=1,A6=1,A7=1):
    # Create the time array
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Calculate intermediate parameters
    R_PNS = R_PNS_fit(time, A, k, x0, B, C)
    R_sh = R_sh_fit(time, alpha, mu, sigma, beta, gamma)
    M_PNS = M_PNS_fit(time, a1, b1)
    M_sh = M_sh_fit(time, a2, b2)
    P_c_val = P_c_fit(time, a3, b3)
    rho_c_val = rho_c_fit(time, a4, b4)
    
    # Initialize the combined signal
    combined_signal = np.zeros_like(time)
    
    # Compute each mode if included and add to the combined signal
    if 'f_2' in modes_to_include:
        freq_f_2 = mode_2f(M_sh, R_sh)
        combined_signal += A1*np.sin(2 * np.pi * np.random.normal(freq_f_2,phase_sigma)*time)
    if 'p1_2' in modes_to_include:
        freq_p1_2 = mode_2p1(M_sh, R_sh)
        combined_signal += A2*np.sin(2 * np.pi * np.random.normal(freq_p1_2,phase_sigma)*time)
    if 'p2_2' in modes_to_include:
        freq_p2_2 = mode_2p2(M_sh, R_sh)
        combined_signal += A3*np.sin(2 * np.pi * np.random.normal(freq_p2_2,phase_sigma)*time)
    if 'p3_2' in modes_to_include:
        freq_p3_2 = mode_2p3(M_sh, R_sh)
        combined_signal += A4*np.sin(2 * np.pi * np.random.normal(freq_p3_2,phase_sigma)*time)
    if 'g1_2' in modes_to_include:
        freq_g1_2 = mode_2g1(M_PNS, R_PNS)
        combined_signal += A5*np.sin(2 * np.pi * np.random.normal(freq_g1_2,phase_sigma)*time)
    if 'g2_2' in modes_to_include:
        freq_g2_2 = mode_2g2(M_PNS, R_PNS)
        combined_signal += A6*np.sin(2 * np.pi * np.random.normal(freq_g2_2,phase_sigma)*time)
    if 'g3_2' in modes_to_include:
        freq_g3_2 = mode_2g3(M_sh, R_sh, P_c_val, rho_c_val)
        combined_signal += A7*np.sin(2 * np.pi * np.random.normal(freq_g3_2,phase_sigma)*time)

    
    # Convert the time-domain signal to the frequency domain using Bilby's FFT
    frequency_domain_signal, frequency_array = bilby.core.utils.series.nfft(combined_signal, sampling_rate)
    
    # Interpolate the frequency-domain signal using numpy.interp
    interpolated_signal = np.interp(f, frequency_array, frequency_domain_signal, left=0, right=0)
    
    # Return the interpolated value(s)
    return {"plus": interpolated_signal, "cross": interpolated_signal}
    
