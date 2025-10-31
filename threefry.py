import numpy as np
from numpy import uint64 as u64, uint32 as u32


# uint64_t rotl_64(uint64_t x, int d) {
#   return ((x << d) | (x >> (64-d)));
# }
# https://github.com/pdebuyl/threefry/blob/master/threefry/_threefry.c
def rotlu64(x: u64, d: int):
    d = u64(d)
    return (x << d) | (x >> (64 - d))


def mix(state: (u64, u64), round_key: int):
    state[0] += state[1]
    state[1] = rotlu64(state[1], round_key) ^ state[0]
    return state


ROTATION_2X64 = (16, 42, 12, 31, 16, 32, 24, 21)

# Key schedule constant C240. "The constant C240 defends against generating extended
# keys which are all zero or almost zero. It also provides an additional defense against
# rotational attacks. C240 is the AES encryption of the plaintext 240 (in decimal) under
# the all-zero 256-bit key; i.e., C240 = AES-256_0(240)."
# In the Random123 library, this constant is named SKEIN_KS_PARITY64
# https://www.schneier.com/wp-content/uploads/2016/02/skein.pdf pp 12
C240 = 0x1BD11BDAA9FC1A22
KEY_LENGTH = 3


def threefry_2x64(seed: (u64, u64), counter: (u64, u64), rounds: int = 13):
    """"""
    np.seterr(over="ignore")
    k = np.zeros(KEY_LENGTH, dtype=u64)
    k[0:2] = seed
    k[2] = C240 ^ k[0] ^ k[1]  # "Parity" key

    # Generate the subkeys. These "mix" in data from the seed on appropriate rounds,
    # using an addition operation. Apparently they protect against slide attacks and
    # rotational cryptanalysis
    # num_subkeys = rounds // 4 + 1
    # subkeys = np.zeros((num_subkeys, 2), dtype=u64)
    # for s in range(num_subkeys):
    #     # s_u64 = u64(s)
    #     subkeys[s, 0] = k[s % KEY_LENGTH]
    #     subkeys[s, 1] = k[(s + 1) % KEY_LENGTH] #+ u64(s)

    # print(subkeys)

    counter = np.asarray(counter, dtype=u64)

    for d in range(rounds):
        # Subkey injection every 4 rounds
        if d % 4 == 0:
            s = u64(d // 4)

            # NEON: vector add
            counter[0] += k[s % KEY_LENGTH]
            counter[1] += k[(s + 1) % KEY_LENGTH] + s

        counter = mix(counter, ROTATION_2X64[d % 8])

        # Permutation. This swaps the places of elements for the next step, accelerating
        # diffusion of bits through the state. NOT NECESSARY IN THREEFRY?
        # counter[0], counter[1] = counter[1], counter[0]

    # Final subkey addition
    # NEON: vector add
    counter[0] += k[(rounds // 4) % KEY_LENGTH]
    counter[1] += k[((rounds // 4) + 1) % KEY_LENGTH] + rounds // 4

    return counter


new = threefry_2x64((0, 123459191283490), (0, 0), rounds=20)
print(new)
# print([*map(hex, new)])
