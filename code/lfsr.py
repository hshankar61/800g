# to do - rewrite for all polynomials from wikipedia page https://en.wikipedia.org/wiki/Linear-feedback_shift_register
# rewrite it as a class
import numpy as np
import matplotlib.pyplot as plt


# feedback polynomial = x**7 + x**6 + 1
def lfsr7(N):
  L = 7
  bit_array = np.zeros(N)
  state = 0x1
  k1 = 0
  for k1 in range(N):
    newbit = ((state>>(L-7)) ^ (state>>(L-6))) & 1
    state = (state >> 1) | (newbit<<(L-1))
    bit_array[k1] = newbit
    k1 += 1
  return bit_array

if __name__ == "__main__":
    bits = lfsr7(127*2)
    foo = bits[0:126] - bits[127:-1]
    if max(foo) != 0:
       print("ERROR: sequence does NOT have period of 127")
    else:
       print("sequence has period of 127")

    bits = lfsr7(127*1)
    bits = 2*bits - 1  # convert to levels -1, 1
    foo = np.correlate(bits,bits,"full")  # whyd does valid return a sequence of length 1?
    plt.plot(foo, '-x')
    plt.title("cross correlation of prbs7 sequence")
    plt.grid(True)
    plt.show()