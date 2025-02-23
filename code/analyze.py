# Convert S-parameters to mixed mode and plot.  Also plot impulse response.
import sys
import skrf as rf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.special

def pam_ber_theory(SNR_dB, M):
    # M is the number of levels, PAM4 has 4 levels, etc.
    # Is this ber or ser?
    ber_array = np.zeros(len(SNR_dB))
    k = 0
    for x in SNR_dB:
        snr = 10**(x/10)
        ber_array[k] = 0.5 * scipy.special.erfc(np.sqrt( (3/((M**2)-1)) * snr/2))
        k += 1
    return ber_array


def find_nearest(array, value):
    """Find the entry in array closest to value

    Args:
        array (numpy array): input arry of values
        value (float): Want to find entry in array closest to value

    Returns:
        idx (int): index into array where the entry is closest to value
        array[idx] (float): the value in array closest to value
    """
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_bw(frequency, Hf_dB, loss= 30, poly_order=9):
    """Find bandwith by fitting a polynomial to the transfer function Hf_dB
    where bandwidth is the frequency at which Hf_dB(bandwidth) = Hf_dB(0) - loss

    Args:
        frequency (float): frequencies for Hf_dB
        Hf_dB (numpy array): Transfer function Hf in dB
        loss (int, optional): loss at bandwidth Defaults to 30.
        poly_order (int, optional): order of polynomial used to fit Hf_dB

    Returns:
        Hf_dB_fit (numpy array): The polynomial fit to Hf_dB
        fbw (float) = bandwidth
        idx (int) = index into frequency array for fbw
    """

    coeffs = np.polyfit(frequency/1e9, Hf_dB, poly_order)
    poly = np.poly1d(coeffs)
    Hf_dB_fit = poly(frequency/1e9)
    dc_error = Hf_dB[0] - Hf_dB_fit[0]
    print(f"Hf_dB[0] {Hf_dB[0]}, dc error of fit = {Hf_dB[0] - Hf_dB_fit[0]}")
    idx, val = find_nearest(Hf_dB_fit, Hf_dB[0]-loss)
    fbw = frequency[idx]
    return Hf_dB_fit, fbw, idx

def find_pulse_response(t, impulse_t, bit_rate=400e9, bits_per_symbol=3, samples_per_symbol=20):
    """Find the pulse response given the impulse response

    Args:
        t (np array): time for impulse response
        impulse_t (np array): _description_
        bit_rate (floar): required bit rate
        bits_per_symbol (float): bits per symbol (to calculate the symbol rate)
        samples_per_symbol (integer): sets the sample rate for the pulse response.  Resample the impulse response as neede.
    """
    symbol_rate = bit_rate / bits_per_symbol
    symbol_period = 1.0 / symbol_rate
    samples_per_symbol = 20
    sample_period = symbol_period / samples_per_symbol
    print(f"symbol period (ps), sample period (ps) = {symbol_period * 1e12}, {sample_period * 1e12}")
    pulse_delay =   2 * samples_per_symbol
    pulse_t = np.zeros(pulse_delay + 2 * samples_per_symbol)
    pulse_t[pulse_delay: pulse_delay + samples_per_symbol] = 1.0
    from scipy.interpolate import interp1d
    f = interp1d(t, impulse_t, kind='cubic')  # linear cubic
    time = np.arange(0, t[len(t)-1], sample_period)
    # Interpolate impulse response to new time step.
    # Note that the channel impulse response is Ts*h(k Ts) and hence amplitude should be adjusted to the new time step
    h_channel = f(time) * (time[1] - time[0]) / (t[1] - t[0])
    p_t = np.convolve(pulse_t, h_channel, mode="full")  # valid, full
    time = np.arange(0, len(p_t))* sample_period
    return time, p_t

def read_snp(file):
    try:
        snp = rf.Network(file)
        if snp.f[0] != 0:
            snp = snp.extrapolate_to_dc(kind="linear")
    except FileNotFoundError:
        sys.exit(f"Error: File not found at {file}")
    except Exception as e:
        sys.ext(f"An error occurred: {e}")
    return snp

def se2mm_4port(snp):
    snp.renumber([0, 1, 2, 3],  # renumbering ports as required by se2gmm()
                 [0, 2, 1, 3])  # [left_p, left_n, right_p, right_n]
    snp.se2gmm(p=2)  # single ended to mixed mode
    return snp

def find_cursors(p_t, samples_per_symbol, threshold=0.015):
    c0 = np.max(p_t)
    idx0 = np.argmax(p_t)
    x1 = int(idx0 - np.floor(idx0/samples_per_symbol) * samples_per_symbol)
    idx_cursors = list(range(x1, len(p_t), samples_per_symbol))
    cursors_all = p_t[idx_cursors]
    threshold = c0 * threshold  # cursors above this value are to be removed by equalization and smaller cursors remain
    idx1 = np.where(np.abs(cursors_all) > threshold)
    cursors_above_threshold = cursors_all[idx1]
    t_cursors_above_threshold = [time[idx_cursors[xx]] for xx in idx1[0]]
    return cursors_all, cursors_above_threshold, t_cursors_above_threshold

def pam_psd(Tsymbol, f, A, bits_per_symbol):
    # modulated spectrum from Proakis
    # M-PAM levels are X = A*{-(M-1), ..., -3, ,-1, 1, 3, ... (M-1)}
    # E{X**2} = A**2 * (M**2 - 1)/3 = average symbol energy
    # Assume that the symbol sequence {X_n} are independent
    # Hence correlation R_x(k) = 0 for k!= 0, = E(X**2} for k == 0)
    num_levels = 2**bits_per_symbol
    levels = A * np.array(range(-(num_levels-1), (num_levels+1), 2))
    num_levels = len(levels)
    if num_levels != num_levels:
        raise ValueError(f"Something wrong with PAM psd calculation {levels}")
    var_levels = (A**2) * ((num_levels**2) - 1) / 3
    P_f = Tsymbol * np.sinc(f * Tsymbol)  # Fourier xform of rect(t/T)
    Sp_f = var_levels * (1/Tsymbol) * (np.abs(P_f))**2  # psd
    return var_levels, Sp_f


# file_dir = Path(r"..\data\tracy_3df_01_2211_sparameters\tracy_3df_01_2211_CR_CPC_HOST\\")
# thru_file_name = Path(r"TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_THRU.s4p")
# fext1_file_name = Path(r"TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_FEXT1.s4p")
# fext2_file_name = Path(r"TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_FEXT2.s4p")
# fext3_file_name = Path(r"TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_FEXT3.s4p")
# next1_file_name = Path(r"TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_NEXT1.s4p")
# next2_file_name = Path(r"TE_224G_CR_TP0_TP5_CPC_1mDAC_071922_NEXT2.s4p")
## or
file_dir = Path(r"..\data\tracy_3df_01_2211_sparameters\tracy_3df_01_2211_CR_CONV_HOST\\")
thru_file_name  = Path(r"TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_THRU.s4p")
fext1_file_name = Path(r"TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_FEXT1.s4p")
fext2_file_name = Path(r"TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_FEXT2.s4p")
fext3_file_name = Path(r"TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_FEXT3.s4p")
next1_file_name = Path(r"TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_NEXT1.s4p")
next2_file_name = Path(r"TE_224G_CR_TP0_TP5_Conventional_1mDAC_7inHst_100622_NEXT2.s4p")

file = file_dir / thru_file_name
snp = read_snp(file)
se2mm_4port(snp)
frequency = snp.f

# plot S21dd and -30 dB bandwidth
Hf_dB = 20*np.log10(np.abs(snp.s[:,1,0])) # S21, all frequencies
Hf_dB_fit, fbw, idx = find_bw(frequency, Hf_dB, loss= 30, poly_order=9) # find 30 dB bandwidth
# plot S21dd(f) and the poly fit
plt.figure(1)
plt.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), 'b', label="S21DD")
plt.plot(frequency/1e9, Hf_dB_fit , 'k', label="S21DD_fit")
plt.plot(frequency[idx]/1e9, Hf_dB[idx], 'xr')
print(f"30dB bandwidth {frequency[idx]/1e9} GHz")
plt.grid(True)
plt.ylabel("dB")
plt.xlabel("freq (GHz)")
plt.title("Thru S21DD, " + str(file_dir.name), fontsize=10)

# Get impulse response
t, h_channel = snp.s21.impulse_response(window='boxcar', pad=0)  # hamming boxcar
idx = np.argmax(h_channel)
idx_min_plot = t[idx] - 200e-12
idx_max_plot = t[idx] + 300e-12
print(f"Ts(ps) from impulse_response() {(t[1] - t[0])*1e12:0.2f}")
plt.figure(2)
plt.plot(t, h_channel, 'k', label="impulse_response")
plt.xlim(idx_min_plot, idx_max_plot)
plt.grid(True)
plt.xlabel("sec")
plt.ylabel("V")
plt.title(f"Impulse Response Ts*h(kTs), Ts(ps) {(t[1] - t[0])*1e12:0.2f}, " + str(file_dir.name) , fontsize=9)

# pulse response
bit_rate=200e9
bits_per_symbol=2
samples_per_symbol=20
symbol_rate = bit_rate / bits_per_symbol
symbol_period = 1.0 / symbol_rate
samples_per_symbol = 20
sample_period = symbol_period / samples_per_symbol
time, p_t = find_pulse_response(t, h_channel, bit_rate=bit_rate, bits_per_symbol=bits_per_symbol, samples_per_symbol=samples_per_symbol)
plt.figure(3)
idx_min_plot = t[idx] - 200e-12
idx_max_plot = t[idx] + 300e-12
plt.plot(time, p_t, 'b', label="output pulse")
plt.xlim(idx_min_plot, idx_max_plot)
plt.grid(True)
plt.ylabel("V")
plt.xlabel("time")
plt.title("Pulse Response , " + f"{bit_rate/bits_per_symbol/1e9:0.1f} GB, " + str(file_dir.name) , fontsize=9)
plt.legend(fontsize=10)

# Get the cursors of the pulse response
c0 = np.max(p_t)  # symbol sample
threshold=0.015 # cursors with abs value larger than threshold*c0
cursors_all, cursors_above_threshold, t_cursors_above_threshold = find_cursors(p_t, samples_per_symbol, threshold=threshold)
eye_closure = 2*c0 - np.sum(abs(cursors_all)) + np.sum(cursors_above_threshold) # assuming cursors above a threshold are equalized
print(f"eye opening without {len(t_cursors_above_threshold)} cursors {eye_closure:0.2f}")
plt.plot(t_cursors_above_threshold, cursors_above_threshold, 'rx')  # plot cursors on top of symbol response


# Transmitter psd of a PAM signal and Tx noise psd
A = 1
signal_power, Sp_f = pam_psd(symbol_period, frequency, A, bits_per_symbol)  # this is the Tx psd
# Tx noise
# quantization noise + thermal noise + jitter
tx_snr_dB = 30  # roughly 5 bit snr ~= 6 * # bits
tx_noise_power = signal_power * 10**(-tx_snr_dB/10)
tx_noise_psd = tx_noise_power / max(frequency) * np.ones(len(frequency))


plt.figure(5)
psd_rx_signal = np.abs(snp.s[:,1,0])**2 * Sp_f
plt.plot(frequency/1e9, 30+10*np.log10(psd_rx_signal), 'c', label="PSD Rx Signal")
psd_tx_noise_at_rx = np.abs(snp.s[:,1,0])**2 * tx_noise_psd
total_noise_psd = psd_tx_noise_at_rx
plt.plot(frequency/1e9, 30+10*np.log10(psd_tx_noise_at_rx), '--c', label="PSD Tx Noise at Rx")
plt.grid(True)
plt.ylabel("dBm/Hz")
plt.xlabel("freq (GHz)")
plt.title("PSD Signal/Fext/Next, " + str(file_dir.name), fontsize=10)

# Calculate fext, next
file = file_dir / fext1_file_name
snp = read_snp(file)
se2mm_4port(snp)
frequency = snp.f
Hf_dB = 20*np.log10(np.abs(snp.s[:,1,0]))
plt.figure(4)
plt.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), 'b', label="S21 Fext1")
plt.grid(True)
plt.ylabel("dB")
plt.xlabel("freq (GHz)")
plt.title("S21DD Fext/Next, " + str(file_dir.name), fontsize=10)

plt.figure(5)
noise_psd = np.abs(snp.s[:,1,0])**2 * Sp_f
total_noise_psd = total_noise_psd + noise_psd
plt.plot(frequency/1e9, 30+10*np.log10(noise_psd), 'b', label="PSD Fext1")
plt.grid(True)


file = file_dir / fext2_file_name
snp = read_snp(file)
frequency = snp.f
Hf_dB = 20*np.log10(np.abs(snp.s[:,1,0]))
plt.figure(4)
plt.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), 'r', label="S21 Fext2")
plt.figure(5)
noise_psd = np.abs(snp.s[:,1,0])**2 * Sp_f
total_noise_psd = total_noise_psd + noise_psd
plt.plot(frequency/1e9, 30+10*np.log10(noise_psd), 'r', label="PSD Fext2")

file = file_dir / fext3_file_name
snp = read_snp(file)
frequency = snp.f
Hf_dB = 20*np.log10(np.abs(snp.s[:,1,0]))
plt.figure(4)
plt.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), 'k', label="Fext3")
plt.figure(5)
noise_psd = np.abs(snp.s[:,1,0])**2 * Sp_f
total_noise_psd = total_noise_psd + noise_psd
plt.plot(frequency/1e9, 30+10*np.log10(noise_psd), 'k', label="PSD Fext3")

file = file_dir / next1_file_name
snp = read_snp(file)
frequency = snp.f
Hf_dB = 20*np.log10(np.abs(snp.s[:,1,0]))
plt.figure(4)
plt.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), 'g', label="Next1")
plt.figure(5)
noise_psd = np.abs(snp.s[:,1,0])**2 * Sp_f
total_noise_psd = total_noise_psd + noise_psd
plt.plot(frequency/1e9, 30+10*np.log10(noise_psd), 'g', label="PSD Next1")

file = file_dir / next2_file_name
snp = read_snp(file)
frequency = snp.f
Hf_dB = 20*np.log10(np.abs(snp.s[:,1,0]))
plt.figure(4)
plt.plot(frequency/1e9, 20*np.log10(np.abs(snp.s[:,1,0])), 'm', label="Next2")
plt.legend()
plt.figure(5)
noise_psd = np.abs(snp.s[:,1,0])**2 * Sp_f
total_noise_psd = total_noise_psd + noise_psd
plt.plot(frequency/1e9, 30+10*np.log10(noise_psd), 'm', label="PSD Next2")
plt.plot(frequency/1e9, 30+10*np.log10(total_noise_psd), 'y', label="PSD total noise")
plt.legend()

plt.figure(6)
plt.plot(frequency/1e9, 10*np.log10(psd_rx_signal) - 10*np.log10(total_noise_psd ), 'b')
plt.grid(True)
plt.title("SNR(f)")
plt.ylabel("dB")
plt.xlabel("freq (GHz)")


# AWGN Theory
SNR_dB_array = np.arange(0, 28.0, 2.0)
ber_array = pam_ber_theory(SNR_dB_array, 4)
plt.figure(0)
plt.semilogy(SNR_dB_array, ber_array, label="PAM4")
plt.grid(True)
SNR_dB_array = np.arange(0, 32.0, 2.0)
ber_array = pam_ber_theory(SNR_dB_array, 8)
plt.semilogy(SNR_dB_array, ber_array, label="PAM8")
plt.legend()
plt.title("BER Theory AWGN")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.show(block=True)