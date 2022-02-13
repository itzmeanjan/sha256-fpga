#include "sha256.hpp"
#include "utils.hpp"
#include <cassert>
#include <iostream>

class kernelSHA256Test;

// default accelerator choice for functional correctness check is FPGA
// emulation device
#if !(defined FPGA_EMU || defined FPGA_HW)
#define FPGA_EMU
#endif

int
main(int argc, char** argv)
{
#if defined FPGA_EMU
  sycl::ext::intel::fpga_emulator_selector s{};
#elif defined FPGA_HW
  sycl::ext::intel::fpga_selector s{};
#endif

  sycl::device d{ s };
  sycl::context c{ d };
  sycl::queue q{ c, d };

  bool* res_d = static_cast<bool*>(sycl::malloc_device(sizeof(bool), q));
  bool* res_h = static_cast<bool*>(std::malloc(sizeof(bool)));

  // Note, following test kernel is adapted from
  // https://github.com/itzmeanjan/merklize-sha/blob/5c9b80cbada54efa2a492e39d5868ac2b834c70a/include/test_sha2_256.hpp
  sycl::event evt =
    q.single_task<kernelSHA256Test>([=]() [[intel::kernel_args_restrict]] {
      // expected 256 -bit digest of 2-to-1 SHA256 hash routine where input ( =
      // 64 -bytes ) is
      //
      // $ python3
      // >>> in = [i for i in range(64)]
      constexpr uint8_t expected[32] = { 253, 234, 185, 172, 243, 113, 3,   98,
                                         189, 38,  88,  205, 201, 162, 158, 143,
                                         156, 117, 127, 207, 152, 17,  96,  58,
                                         140, 68,  124, 209, 217, 21,  17,  8 };

      // on-chip block RAM based allocation for preparing input message bytes
      // of length 64, where two SHA256 digests are concatenated to each other
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(1),
        intel::numbanks(64)]] uint8_t in[64];

#pragma unroll 64 // 512 -bit burst coalesced loading into on-chip block RAM
      for (size_t i = 0; i < 64; i++) {
        in[i] = i;
      }

      // 64 -bytes input is interpreted as 16 message words ( SHA256 word size
      // is 32 -bit )
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(4),
        intel::numbanks(16)]] uint32_t in_words[16];

      sycl::private_ptr<uint8_t> in_ptr{ in };
#pragma unroll 16 // 512 -bit burst coalesced loading between on-chip block RAMs
      for (size_t i = 0; i < 16; i++) {
        in_words[i] = from_be_bytes(in_ptr + (i << 2));
      }

      // padding 512 -bit input such that after padding output bit length is
      // also evenly divisible by 512 ( = SHA256 message block size )
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(4),
        intel::numbanks(32)]] uint32_t padded[32];

      sycl::private_ptr<uint32_t> in_words_ptr{ in_words };
      sycl::private_ptr<uint32_t> padded_ptr{ padded };

      // now we've 1024 -bit ( padded ) input
      sha256::pad_input_message(in_words_ptr, padded_ptr);

      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(4),
        intel::numbanks(8)]] uint32_t hash_state[8];
      [[intel::fpga_memory("BLOCK_RAM")]] uint32_t msg_schld[64];

      sycl::private_ptr<uint32_t> hash_state_ptr{ hash_state };
      sycl::private_ptr<uint32_t> msg_schld_ptr{ msg_schld };

      // compute digest of 1024 -bit padded input, producing 256 -bit digest ( =
      // 8 message words )
      sha256::hash(hash_state_ptr, msg_schld_ptr, padded_ptr);

      // finally convert 8 message word digest to 32 bytes of output
      [[intel::fpga_memory("BLOCK_RAM"),
        intel::bankwidth(1),
        intel::numbanks(32)]] uint8_t digest[32];
      sycl::private_ptr<uint8_t> digest_ptr{ digest };

#pragma unroll 8
      for (size_t i = 0; i < 8; i++) {
        to_be_bytes(hash_state_ptr[i], digest_ptr + (i << 2));
      }

      // comparing SHA256 digest, on accelerator itself
      bool _res = true;
      for (size_t i = 0; i < 32; i++) {
        _res &= (digest[i] == expected[i]);
      }

      // write back to global memory
      res_d[0] = _res;
    });

  // copy assertion result back to host
  q.memcpy(res_h, res_d, sizeof(bool)).wait();

  // assert that SHA256 2-to-1 hash computation did what it was supposed to do
  assert(res_h[0] == true);

  // deallocate resources
  //
  // note, memory allocated on device global memory is SYCL runtime managed
  sycl::free(res_d, q);
  std::free(res_h);

  return EXIT_SUCCESS;
}
