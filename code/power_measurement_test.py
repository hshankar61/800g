# Generate psd of a pam signal with Proakis formula and check if the power is correct by integrating the psd.
import numpy as np
import matplotlib.pyplot as plt

symbol_rate = 80e9
Tsymbol = 1.0/symbol_rate
fmax=1000e9
nf=10000
df = fmax/nf
A = 1
M = 8
f= np.arange(-fmax, fmax + df, df)
P_f = Tsymbol * np.sinc(f * Tsymbol)
Sp = A**2 * ((M**2 -1) / 3) * (1.0/Tsymbol) * (abs(P_f))**2  # Proakis formula
from scipy import integrate
Pmeasured = integrate.simpson(Sp, x=f)
Pformula = A**2 * (M**2 -1) / 3  # Proakis formula
print(f"Power from psd = {Pmeasured:0.1f}, Power from formula = {Pformula}")
plt.figure()
plt.plot(f, Sp)
plt.grid(True)
plt.ylabel("psd (W/Hz)")
plt.xlabel("f (Hz)")
plt.title("PAM signal psd from formula")
plt.text(0.1*max(f), 0.3*max(Sp), "Parseval:")
plt.text(0.1*max(f), 0.2*max(Sp), f"power from formula = {Pformula:0.2f}")
plt.text(0.1*max(f), 0.1*max(Sp), f"power from integration = {Pmeasured:0.2f}")
plt.show()


