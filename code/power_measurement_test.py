# Generate psd of a pam signal with Proakis formul and check if the power is correct by integrating the psd.
import numpy as np
import matplotlib.pyplot as plt

symbol_rate = 80e9
Tsymbol = 1.0/symbol_rate
fmax=800e9
nf=8000
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
plt.show()
