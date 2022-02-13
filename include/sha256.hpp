#pragma once
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace sha256 {

// Initial hash values for SHA2-256, as specified in section 5.3.3 of Secure
// Hash Standard http://dx.doi.org/10.6028/NIST.FIPS.180-4
//
// Copied from
// https://github.com/itzmeanjan/merklize-sha/blob/a209e74b91b5da8ce6dc360fc0b107ac9e693d12/include/sha2_256.hpp#L15-L20
constexpr uint32_t IV_0[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                               0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

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

}
