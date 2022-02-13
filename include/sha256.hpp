#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sha256 {

// Compile time check to ensure that following circular right shift's maximum
// requested bit pos < 32
static inline constexpr bool
lt_32(const uint8_t n)
{
  return n < uint8_t(32);
}

// Circular right shift of 32 -bit sha256 word, with compile-time check
// for rotation bit position n ( < 32 )
template<uint8_t n>
static inline const uint32_t
rotr(const uint32_t x) requires(lt_32(n))
{
  return (x >> n) | (x << (32 - n));
}

// Initial hash values for SHA2-256, as specified in section 5.3.3 of Secure
// Hash Standard http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Copied from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2_256.hpp#L15-L20
constexpr uint32_t IV[8] = {
  0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

// SHA2-256 variants uses 64 words as constants, which are
// specified in section 4.2.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L20-L35
constexpr uint32_t K[64] = {
  0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
  0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
  0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
  0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
  0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
  0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
  0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
  0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
  0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
  0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
  0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// SHA2-256 function, defined in section 4.1.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L37-L45
static inline const uint32_t
ch(const uint32_t x, const uint32_t y, const uint32_t z)
{
  return (x & y) ^ (~x & z);
}

// SHA2-256 function, defined in section 4.1.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L47-L55
static inline const uint32_t
maj(const uint32_t x, const uint32_t y, const uint32_t z)
{
  return (x & y) ^ (x & z) ^ (y & z);
}

// SHA2-256 function, defined in section 4.1.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L57-L63
static inline const uint32_t
Σ_0(const uint32_t x)
{
  return rotr<2>(x) ^ rotr<13>(x) ^ rotr<22>(x);
}

// SHA2-256 function, defined in section 4.1.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L65-L71
static inline const uint32_t
Σ_1(const uint32_t x)
{
  return rotr<6>(x) ^ rotr<11>(x) ^ rotr<25>(x);
}

// SHA2-256 function, defined in section 4.1.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L73-L79
static inline const uint32_t
σ_0(const uint32_t x)
{
  return rotr<7>(x) ^ rotr<18>(x) ^ (x >> 3);
}

// SHA2-256 function, defined in section 4.1.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Taken from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2.hpp#L81-L87
static inline const uint32_t
σ_1(const uint32_t x)
{
  return rotr<17>(x) ^ rotr<19>(x) ^ (x >> 10);
}

// Given 512 -bit message block ( i.e. 16 message words ), to be consumed into
// hash state, this routine prepares 64 message schedules ( i.e. total 64
// message words, where each word of sha256 is 32 -bit unsigned integer )
// which will be mixed into hash state in 64 sha256 rounds
inline void
prepare_message_schedule(sycl::private_ptr<uint32_t> in,
                         sycl::private_ptr<uint32_t> out)
{
  // first 16 message schedules are same as original message words
  // of 512 -bit message block
#pragma unroll 16 // 512 -bit burst coalesced loading
  for (size_t i = 0; i < 16; i++) {
    out[i & 0xf] = in[i & 0xf];
  }

  // 48 iteration rounds, preparing 48 remaining message schedules
  // of total 64 message schedules for sha256
#pragma unroll 16
  for (size_t i = 16; i < 64; i++) {
    const uint32_t tmp0 = σ_1(out[(i - 2) & 0x3f]) + out[(i - 7) & 0x3f];
    const uint32_t tmp1 = σ_0(out[(i - 15) & 0x3f]) + out[(i - 16) & 0x3f];

    out[i & 0x3f] = tmp0 + tmp1;
  }
}

// 512 -bit input to sha256 2-to-1 hash function requires me to pad 16 more
// words ( note, sha256 word size is 32 -bit ) making total of 1024 -bit padded
// input, which will be digested into 256 -bit output, as two consecutively
// consumed message blocks ( note, sha256 message block is 512 -bit wide )
//
// See section 5.1.1 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Copied from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2_256.hpp#L64-L100
inline void
pad_input_message(sycl::private_ptr<uint32_t> in,
                  sycl::private_ptr<uint32_t> out)
{
  // copy first 64 -bytes = 16 words ( sha256 word 32 -bit wide )
  // from input to output on-chip block RAM based memory allocation
  //
  // this loop execution can be fully parallelized
#pragma unroll 16 // 512 -bit burst coalesced access !
  for (size_t i = 0; i < 16; i++) {
    out[i] = in[i];
  }

  constexpr size_t offset = 16;

  // next 16 words are set to 0, due to padding requirement
  // and also to leverage 512 -bit burst coalesced memory access
#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    out[offset + i] = 0u;
  }

  // then specifically set 16 -th word
  out[offset] = 0b10000000u << 24;

  // note, intermediate 14 words are already set to 0 ( as per sha256 padding
  // requirement ) in above fully unrolled for loop

  // finally last word of 1024 -bit padded input set to original input length,
  // in bits ( = 512 )
  out[31] = 0u | 0b00000010u << 8;
}

// As input takes two padded, parsed input message blocks ( = 1024 -bit, total )
// and computes SHA2-256 digest ( = 256 -bit ) in two sequential rounds\
//
// Finally computed digest is placed on first 8 words of hash state
//
// See algorithm defined in section 6.2.2 of Secure Hash Standard
// http://dx.doi.org/10.6028/NIST.FIPS.180-4
void
hash(sycl::private_ptr<uint32_t> hash_state,
     sycl::private_ptr<uint32_t> msg_schld,
     sycl::private_ptr<uint32_t> in,
     sycl::private_ptr<uint32_t> out)
{
  // initial hash state of 256 -bit
#pragma unroll 8 // 256 -bit burst coalesced access
  for (size_t i = 0; i < 8; i++) {
    hash_state[i] = IV[i];
  }

  // padded input message is 1024 -bit wide, so two message blocks ( each of 512
  // -bit ) are to be mixed into hash state in two sequential rounds
  //
  // this loop will be pipelined, but mutliple iterations can't be parallelly
  // executed, due to sequential data dependency
  for (size_t i = 0; i < 2; i++) {
    // step 1 of algorithm defined in section 6.2.2 of Secure Hash Standard
    // http://dx.doi.org/10.6028/NIST.FIPS.180-4
    prepare_message_schedule(in + (i << 4), msg_schld);

    // step 2 of algorithm defined in section 6.2.2 of Secure Hash Standard
    // http://dx.doi.org/10.6028/NIST.FIPS.180-4
    uint32_t a = hash_state[0];
    uint32_t b = hash_state[1];
    uint32_t c = hash_state[2];
    uint32_t d = hash_state[3];
    uint32_t e = hash_state[4];
    uint32_t f = hash_state[5];
    uint32_t g = hash_state[6];
    uint32_t h = hash_state[7];

    // step 1 of algorithm defined in section 6.2.2 of Secure Hash Standard
    // http://dx.doi.org/10.6028/NIST.FIPS.180-4
    //
    // this inner loop will be pipelined, but multiple iterations can't be
    // parallelly executed, because 64 rounds are applied sequentially --- so
    // data dependency is in play !
    for (size_t t = 0; t < 64; t++) {
      const uint32_t tmp0 = h + Σ_1(e) + ch(e, f, g) + K[t] + msg_schld[t];
      const uint32_t tmp1 = Σ_0(a) + maj(a, b, c);

      h = g;
      g = f;
      f = e;
      e = d + tmp0;
      d = c;
      c = b;
      b = a;
      a = tmp0 + tmp1;
    }

    // see step 4 of algorithm defined in section  6.2.2 of Secure Hash Standard
    // http://dx.doi.org/10.6028/NIST.FIPS.180-4
    hash_state[0] += a;
    hash_state[1] += b;
    hash_state[2] += c;
    hash_state[3] += d;
    hash_state[4] += e;
    hash_state[5] += f;
    hash_state[6] += g;
    hash_state[7] += h;
  }

  // now 2-to-1 digest of originally 512 -bit input should be placed on first 8
  // words of hash state
}

}
