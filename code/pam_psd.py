# Create psd of a PAM signal per formula from Proakis

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson as simpson

def pam_psdt(num_levels, A, symbol_rate, samples_per_symbol):
    # M-PAM levels are X = A*{-(M-1), ..., -3, ,-1, 1, 3, ... (M-1)}
    # E{X**2} = A**2 * (M**2 - 1)/3 = average symbol energy
    # Assume that the symbol sequence {X_n} are independent
    # Hence correlation R_x(k) = 0 for k!= 0, = E(X**2} for k == 0)
    num_levels = 4  # number of pam levels
    levels = A * np.array(range(-(num_levels-1), (num_levels+1), 2))
    num_levels = len(levels)
    if num_levels != num_levels:
        raise ValueError(f"Something wrong with number of symbol levels {levels}")
    var_levels = (A**2) * ((num_levels**2) - 1) / 3
    print(f"P_formua= {var_levels:0.2f}")
    Nsymbols = 4000
    Navg = 1000
    # symbol_rate = 54e9
    Tsymbol = 1.0/symbol_rate
    # samples_per_symbol = 9
    Ts = Tsymbol / samples_per_symbol
    Fs = 1.0 / Ts
    N = Nsymbols * samples_per_symbol  # total number of samples
    N1 = N - 1  # number of samples that are captured for spectral analysis
    df = Fs / N1
    f = np.arange(-N1//2, N1//2) * df
    # f= np.arange(-fmax, fmax, 2*fmax/nf)

    # modulated spectrum from Proakis
    P_f = Tsymbol * np.sinc(f * Tsymbol)  # Fourier xform of rect(t/T)
    Sp_f = var_levels * (1/Tsymbol) * (np.abs(P_f))**2  # psd
    
    # This section of code creates a PAM signal in the time domain and measures its psd usig Welch to comapre to Proakis formula.
    # total_power = simpson(Sp_f, x=None, dx=df)  # sum(Sp_f * df)  # this should equal var_levels
    # print(f"Pt = {var_levels}, Pf={total_power:0.2f}")

    window_types = ['rect    ', 'hanning ', 'hamming ', 'blackman', 'bartlett']
    window_type = window_types[0]
    if window_type == window_types[0]:
        w = np.ones(N1)
    elif window_type == window_types[1]:
        w = np.hanning(N1)
    elif window_type == window_types[2]:
        w = np.hamming(N1)
    elif window_type == window_types[3]:
        w = np.blackman(N1)
    elif window_type == window_types[4]:
        w = np.bartlett(N1)
    S1 = np.sum(w)
    S1n = S1/N1  # normalized
    S2 = np.sum(w**2)
    S2n = S2/N1
    ENBW = S2n / S1n**2

    Xsqr_avg = np.zeros(N1)
    for k0 in range(Navg):  # run Navgtimes and average
        symbols = A * (2 * np.random.randint(0, num_levels, Nsymbols) - (num_levels - 1))
        x = np.zeros(N)
        x[0:N+1:samples_per_symbol] = symbols
        x = np.convolve(x, np.ones(samples_per_symbol), mode='same')
        x = x[0:N1]
        X = np.fft.fft(x * w)
        Xsqr_avg = Xsqr_avg + ((np.abs(X))**2)

    Xsqr_avg = Xsqr_avg / Navg
    psdX = Xsqr_avg / (Fs * S2)  # units = Vrms^2/Hz, power spectral density
    psdX = np.fft.fftshift(psdX)

    ymax = max(psdX)
    ymax = np.ceil(10*np.log10(max(psdX)/10)/10) * 10 + 10
    plotnum = 1
    plt.figure(plotnum)
    plt.plot(f/1e9, 10*np.log10(psdX), label="from fft")
    plt.title("Simulation")
    plt.ylim(ymax-60, ymax)
    plt.grid(True)

    # modulated spectrum from Proakis
    # P_f = Tsymbol * np.sinc(f * Tsymbol)  # Fourier xform of rect(t/T)
    # Sp_f = var_levels * (1/Tsymbol) * (np.abs(P_f))**2  # psd
    plt.plot(f/1e9, 10*np.log10(Sp_f), '--r', label="formula")


    # Welch
    from scipy.signal import welch
    Nw = N * Navg
    Nsymbols = Nw // samples_per_symbol
    symbols = A * (2 * np.random.randint(0, num_levels, Nsymbols) - (num_levels - 1))
    x = np.zeros(Nw)
    x[0:Nw+1:samples_per_symbol] = symbols
    x = np.convolve(x, np.ones(samples_per_symbol), mode='same')
    P_t = np.var(x)
    print(f"P_t= {P_t:0.2f}")

    fw, pw = welch(x, Fs, window='boxcar', detrend=False, return_onesided=False)
    fw = np.fft.fftshift(fw)
    pw = np.fft.fftshift(pw)
    plt.plot(fw/1e9, 10*np.log10(pw + 1e-1000), '--k', label="welch")
    plt.title("PSD")
    plt.ylabel("V^2/Hz")
    plt.xlabel("GHz")
    plt.ylim(ymax-60, ymax)
    plt.grid(True)
    plt.legend()
    # plt.show()

    return f, Sp_f


def pam_psdf(num_levels, power, symbol_rate, fmax=100e9, nf=1024):
    # Returns psd of a PAM signal using Proakis formula. No time domain signal is created.
    # num_levels = # of PAM levels, 2, 4, 8, etc
    # power = power of signal in Watts
    # symbol_rate in symbols/sec
    # fmax = max frequency to which spectrum is generated
    # nf = # of samples of power spectral density

    # M-PAM levels are X = A*{-(M-1), ..., -3, ,-1, 1, 3, ... (M-1)}
    # E{X**2} = A**2 * (M**2 - 1)/3
    # Assume that the symbol sequence {X_n} are independent
    # Hence correlation R_x(k) = 0 for k!= 0, = E(X**2} for k == 0)
    # pulse p(t) = rect(t/T) and P(f) = T sinc(f T)
    
    A = np.sqrt(3 * power / (num_levels**2 -1) )  # symbol levels = +/-A, +/-3A, +/-5A, ... +/-(num_levels -1)A, num_levels even
    Tsymbol = 1.0/symbol_rate
    df = fmax/nf
    f= np.arange(-fmax, fmax + df, df)

    # modulated spectrum from Proakis eqn 3.4-16
    P_f = Tsymbol * np.sinc(f * Tsymbol)  # Fourier xform of rect(t/T)
    var_pam_symbols =  (A**2) * ((num_levels**2 -1) / 3)
    Sp_f = var_pam_symbols * (1/Tsymbol) * (np.abs(P_f))**2  # psd of power signal = |X(f)|**2/T
    
    total_power = simpson(Sp_f, x=None, dx=df)  # sum(Sp_f * df)  # this should equal var_levels
    var_levels = (A**2) * ((num_levels**2) - 1) / 3
    print(f"P_formula = {var_levels:0.2f}, Pf={total_power:0.2f}")
    plt.figure()
    plt.plot(f, 10*np.log10(Sp_f))
    plt.grid(True)
    ymax = np.ceil(10*np.log10(max(Sp_f)/10)/10) * 10 + 10
    plt.ylim(ymax-60, ymax+10)
    plt.ylabel("V^2/Hz (dB)")

    return f, Sp_f

if __name__ == "__main__":
    f, Sp_f = pam_psdt(num_levels=4, A=1, symbol_rate=60e9, samples_per_symbol=10)
    f1, Sp_f1 = pam_psdf(num_levels=4, power=1, symbol_rate=60e9, fmax=3e11, nf=1025)

    plt.show()



