# Convert S-parameters to mixed mode and plot.  Also plot impulse response.
import sys
import skrf as rf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plotdata(snp, ax1, ax2, ax3, trace_color, trace_label):
    frequency = snp.f
    # Convert to mixed mode
    snp.renumber([0, 1, 2, 3],  # renumbering ports as required by se2gmm()
                 [0, 2, 1, 3])  # [left_p, left_n, right_p, right_n]
    snp.se2gmm(p=2)
    ax1.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), trace_color, label=trace_label)
    ax2.plot(frequency/1e9, np.real(snp.group_delay[:,1,0]), trace_color, label=trace_label)

    t, y_dd = snp.s21.impulse_response(window='boxcar', pad=2000)  # hamming boxcar
    ax3.plot(t, np.real(y_dd), trace_color, label=trace_label, linewidth=1)
    return ax1, ax2

fig1 = plt.figure(1)
axis1 = fig1.add_axes([0.15, 0.1, 0.75, 0.8])  # for s-parameter magnitude plots
fig2 = plt.figure(2)
axis2 = fig2.add_axes([0.15, 0.1, 0.75, 0.8])  # for s-parameter group delay plots
fig3 = plt.figure(3)  # for impulse response plots
axis3 = fig3.add_axes([0.15, 0.1, 0.75, 0.8])

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
axis1, axis2 = plotdata(snp, ax1=axis1, ax2=axis2, ax3=axis3, trace_color='b',  trace_label="CR_CONV_HOST")

file_dir = Path(r"data\tracy_3df_01_2211_sparameters\tracy_3df_01_2211_CR_CPC_HOST")
file_name = Path("TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_THRU.s4p")
file_path = file_dir / file_name
try:
    snp = rf.Network(file_path)
    if snp.f[0] != 0:
        snp = snp.extrapolate_to_dc(kind="linear")
except FileNotFoundError:
    sys.exit(f"Error: File not found at {file_path}")
except Exception as e:
    sys.ext(f"An error occurred: {e}")
axis1, axis2 = plotdata(snp, ax1=axis1, ax2=axis2, ax3=axis3, trace_color='g',  trace_label="CR_CPC_HOST")

file_dir = Path(r"data\tracy_3df_01_2211_sparameters\tracy_3df_01_2211_CR_NCC_HOST")
file_name = Path("TE_224G_CR_TP0_TP5_NCC_1mDAC_100622_THRU.s4p")
file_path = file_dir / file_name
try:
    snp = rf.Network(file_path)
    if snp.f[0] != 0:
        snp = snp.extrapolate_to_dc(kind="linear")
except FileNotFoundError:
    sys.exit(f"Error: File not found at {file_path}")
except Exception as e:
    sys.ext(f"An error occurred: {e}")
axis1, axis2 = plotdata(snp, ax1=axis1, ax2=axis2, ax3=axis3, trace_color='r',  trace_label="CR_NCC_HOST")

plt.figure(1)
plt.title("tracy_3df_01_2211")
axis1.set_xlabel("f (GHz)")
axis1.set_ylabel("Sd2d1 (dB)")
# axis1.set_ylim(-100, 0)
plt.grid(True)
plt.legend()

plt.figure(2)
plt.title("tracy_3df_01_2211")
axis2.set_xlabel("f (GHz)")
axis2.set_ylabel("Group delay (sec)")
# axis1.set_ylim(-100, 0)
plt.grid(True)
plt.legend()


plt.figure(3)
plt.title("tracy_3df_01_2211 with padding")
axis3.set_xlabel("t (sec)")
axis3.set_ylabel("V")
axis3.set_xlim(0*0.85e-8, 15e-9)
# axis3.set_ylim(-0.02, 0.2)
plt.grid(True)
plt.legend()

plt.show()