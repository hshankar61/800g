# Check passitivty of S-parameters using scikit-rf function and also by computing S * dagger(S) and its eigen values
import sys
import skrf as rf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# following line assumes that the directory 800g is opened from VSCode->Open folder 800g
file_dir = Path(r"data\tracy_3df_01_2211_sparameters\tracy_3df_01_2211_CR_NCC_HOST")
file_name = Path("TE_224G_CR_TP0_TP5_NCC_1mDAC_100622_THRU.s4p")
file_path = file_dir / file_name

# Create a Network object from the Touchstone file
try:
    snp = rf.Network(file_path)
except FileNotFoundError:
    sys.exit(f"Error: File not found at {file_path}")
except Exception as e:
    sys.ext(f"An error occurred: {e}")

print(f"scikit-rf passivity test returns {snp.is_passive()}")  # Built in passivity test

# The following code computes the eigenvalues of S * dagger(S) at each frequency and checks if it is < 1.
# The following gives more information, the frequency at which passivity fails
if 'snp' in locals():
    frequency = snp.f
    s_pars = snp.s
    max_singular_value = np.zeros(len(frequency))
    for k in range(len(frequency)):
        Snn= s_pars[k]
        S = np.linalg.svd(Snn, compute_uv=False)
        max_singular_value[k] = S[0]

plt.figure()
plt.plot(frequency, max_singular_value)
plt.grid(True)
plt.title("max eigenvalues of S * dagger(S) for each frequency")
plt.xlabel("f (Hz)")
plt.show()

