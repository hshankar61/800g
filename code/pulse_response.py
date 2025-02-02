# Convert S-parameters to mixed mode and plot.  Also plot impulse response.
import sys
import skrf as rf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# following line assumes that the directory 800g is opened from VSCode->Open folder 800g
file_dir = Path(r"data\tracy_3df_01_2211_sparameters\tracy_3df_01_2211_CR_CONV_HOST")
file_name = Path("TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_THRU.s4p")
file_path = file_dir / file_name
try:
    snp = rf.Network(file_path)
    if snp.f[0] != 0:
        snp = snp.extrapolate_to_dc(kind="linear")
except FileNotFoundError:
    sys.exit(f"Error: File not found at {file_path}")
except Exception as e:
    sys.ext(f"An error occurred: {e}")

frequency = snp.f
# Convert to mixed mode
snp.renumber([0, 1, 2, 3],  # renumbering ports as required by se2gmm()
             [0, 2, 1, 3])  # [left_p, left_n, right_p, right_n]
snp.se2gmm(p=2)  # single ended to mixed mode
t, h_channel = snp.s21.impulse_response(window='boxcar', pad=0)  # hamming boxcar
plt.figure()
plt.plot(t, h_channel, 'k')

# Create an input pulse
bit_rate = 200e9
pam_levels = 4
symbol_rate = bit_rate / np.log2(pam_levels)
symbol_period = 1.0 / symbol_rate
samples_per_symbol = 12
sample_period = symbol_period / samples_per_symbol
pulse_t = np.zeros(10*samples_per_symbol)
pulse_t[2*samples_per_symbol:3*samples_per_symbol] = 1.0
from scipy.interpolate import interp1d
f = interp1d(t, h_channel, kind='linear')
time = np.arange(0, t[len(t)-1], sample_period)
h_channel = f(time)
plt.plot(time, h_channel, '--m')
y = np.convolve(pulse_t, h_channel, mode="valid")
dt = time[1] - time[0]
plt.plot(np.arange(0, len(pulse_t))* dt, pulse_t, 'b')
plt.plot(np.arange(0, len(y))* dt, y, 'r')
plt.grid(True)
plt.show()
