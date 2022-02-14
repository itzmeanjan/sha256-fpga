#pragma once
#include "sha256.hpp"
#include "utils.hpp"
#include <cassert>

namespace merklize {

// Kernels predeclared to avoid name mangling in optimization report
class kernelSHA256Hash0;
class kernelSHA256Hash1;
class kernelMerklizationOrchestrator;

// Pipes predeclared to avoid name mangling in optimization report
//
class SHA256MessageWords0; // input to hash compute kernel 0
class SHA256DigestWords0;  // output from hash compute kernel 0
class SHA256MessageWords1; // input to hash compute kernel 1
class SHA256DigestWords1;  // output from hash compute kernel 1

// Producer kernel(s) passing 1024 -bit (padded) input message ( = 32 words ) to
// compute kernel(s) over these pipe
using ipipe0 = sycl::ext::intel::pipe<SHA256MessageWords0, uint32_t, 0>;
using ipipe1 = sycl::ext::intel::pipe<SHA256MessageWords1, uint32_t, 0>;

// After computing SHA256 digest on 1024 -bit input message, 8 message words are
// passed to consumer kernel(s), over these pipe
using opipe0 = sycl::ext::intel::pipe<SHA256DigestWords0, uint32_t, 0>;
using opipe1 = sycl::ext::intel::pipe<SHA256DigestWords1, uint32_t, 0>;

// Computes binary logarithm of number `n`,
// where n = 2 ^ i | i = {1, 2, 3 ...}
const size_t
bin_log(size_t n)
{
  size_t cnt = 0;

  while (n > 1ul) {
    n >>= 1;
    cnt++;
  }

  return cnt;
}

// Computes all intermediate nodes of Binary Merkle Tree using SHA256 2-to-1
// hash function, where leaf node count is power of 2 value
//
// In this routine, two kernels will be communicating over SYCL pipes, where
// orchestrator kernel which is responsible for driving multiple phases of
// computation of all intermediates of binary merkle tree, sends padded input
// message words ( = 32 ) over blocking SYCL pipe ( though it has enough
// capacity to buffer ) and waits for completion of SHA256 computation on
// compute kernel, which finally sends back 32 -bytes digest to orchestrator for
// placing it in proper position in output memory allocation (on global memory),
// which will again be used in next level of intermediate node computation, if
// not root of tree
//
// Ensure that SYCL queue has profiling enabled, as at successful completion of
// this routine it returns time spent in computing all intermediate nodes of
// binary merkle tree
sycl::cl_ulong
merklize(sycl::queue& q,
         const size_t leaf_cnt,
         uint32_t* const __restrict leaves,
         const size_t i_size,
         uint32_t* const __restrict intermediates,
         const size_t o_size)
{
  assert(i_size == o_size);                 // ensure enough memory allocated
  assert((leaf_cnt & (leaf_cnt - 1)) == 0); // ensure power of 2

  sycl::event evt0 = q.single_task<kernelMerklizationOrchestrator>([=]() {
    sycl::device_ptr<uint32_t> leaves_ptr{ leaves };
    sycl::device_ptr<uint32_t> intermediates_ptr{ intermediates };

    // on-chip (kernel) private memory allocation with attributes
    // such that enough read/ write ports are available for facilitating
    // parallel (stall-free, preferrably) access
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(16)]] uint32_t msg[16];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(32)]] uint32_t padded[32];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(8)]] uint32_t digest[8];

    const size_t i_offset = 0;
    const size_t o_offset = (leaf_cnt >> 1) << 3;
    const size_t itr_cnt = leaf_cnt >> 1;

    [[intel::ivdep]] for (size_t i = 0; i < itr_cnt; i += 2)
    {
      const size_t i_offset_0 = i_offset + (i << 4);
      const size_t i_offset_1 = i_offset + ((i + 1) << 4);
      const size_t o_offset_0 = o_offset + (i << 3);
      const size_t o_offset_1 = o_offset + ((i + 1) << 3);

#pragma unroll 16 // 512 -bit burst coalesced global memory read
      for (size_t j = 0; j < 16; j++) {
        msg[j] = leaves_ptr[i_offset_0 + j];
      }

      sha256::pad_input_message(msg, padded);

      // send 32 padded input message words to compute kernel
      [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
      {
        ipipe0::write(padded[j]);
      }

      // ---

#pragma unroll 16 // 512 -bit burst coalesced global memory read
      for (size_t j = 0; j < 16; j++) {
        msg[j] = leaves_ptr[i_offset_1 + j];
      }

      sha256::pad_input_message(msg, padded);

      // send 32 padded input message words to compute kernel
      [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
      {
        ipipe1::write(padded[j]);
      }

      // ---

      [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
      {
        digest[j] = opipe0::read();
      }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
      for (size_t j = 0; j < 8; j++) {
        intermediates_ptr[o_offset_0 + j] = digest[j];
      }

      // ---

      [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
      {
        digest[j] = opipe1::read();
      }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
      for (size_t j = 0; j < 8; j++) {
        intermediates_ptr[o_offset_1 + j] = digest[j];
      }
    }

    // these many levels of intermediate nodes ( including root of tree )
    // remaining to be computed
    const size_t rounds = bin_log(leaf_cnt >> 1) - 1;

    for (size_t r = 0; r < rounds; r++) {
      const size_t i_offset = (leaf_cnt << 2) >> r;
      const size_t o_offset = i_offset >> 1;
      const size_t itr_cnt = leaf_cnt >> (r + 2);

      [[intel::ivdep]] for (size_t i = 0; i < itr_cnt; i += 2)
      {
        const size_t i_offset_0 = i_offset + (i << 4);
        const size_t i_offset_1 = i_offset + ((i + 1) << 4);
        const size_t o_offset_0 = o_offset + (i << 3);
        const size_t o_offset_1 = o_offset + ((i + 1) << 3);

#pragma unroll 16 // 512 -bit burst coalesced global memory read
        for (size_t j = 0; j < 16; j++) {
          msg[j] = intermediates_ptr[i_offset_0 + j];
        }

        sha256::pad_input_message(msg, padded);

        // send 32 padded input message words to compute kernel
        [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
        {
          ipipe0::write(padded[j]);
        }

        // ---

#pragma unroll 16 // 512 -bit burst coalesced global memory read
        for (size_t j = 0; j < 16; j++) {
          msg[j] = intermediates_ptr[i_offset_1 + j];
        }

        sha256::pad_input_message(msg, padded);

        // send 32 padded input message words to compute kernel
        [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
        {
          ipipe1::write(padded[j]);
        }

        // ---

        [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
        {
          digest[j] = opipe0::read();
        }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
        for (size_t j = 0; j < 8; j++) {
          intermediates_ptr[o_offset_0 + j] = digest[j];
        }

        // ---

        [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
        {
          digest[j] = opipe1::read();
        }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
        for (size_t j = 0; j < 8; j++) {
          intermediates_ptr[o_offset_1 + j] = digest[j];
        }
      }
    }

    // --- compute root of merkle tree ---

#pragma unroll 16 // 512 -bit burst coalesced global memory read
    for (size_t j = 0; j < 16; j++) {
      msg[j] = intermediates_ptr[16 + j];
    }

    sha256::pad_input_message(msg, padded);

    // send 32 padded input message words to compute kernel
    [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
    {
      ipipe0::write(padded[j]);
    }

    [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
    {
      digest[j] = opipe0::read();
    }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
    for (size_t j = 0; j < 8; j++) {
      intermediates_ptr[8 + j] = digest[j];
    }
  });

  sycl::event evt1 = q.single_task<kernelSHA256Hash0>([=]() {
    [[intel::fpga_register]] const size_t rounds = leaf_cnt >> 1;

    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(32)]] uint32_t padded[32];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(8)]] uint32_t hash_state[8];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(16)]] uint32_t msg_schld[64];

    for (size_t r = 0; r < rounds; r++) {

      // get padded 1024 -bit input message over SYCL pipe
      [[intel::ivdep]] for (size_t i = 0; i < 32; i++)
      {
        padded[i] = ipipe0::read();
      }

      // compute sha256 on that input, keep output
      // in hash state ( 8 message words )
      sha256::hash(hash_state, msg_schld, padded);

      // finally send 8 message words ( = 256 -bit )
      // input over SYCL pipe, back to orchestractor kernel
      [[intel::ivdep]] for (size_t i = 0; i < 8; i++)
      {
        opipe0::write(hash_state[i]);
      }
    }
  });

  sycl::event evt2 = q.single_task<kernelSHA256Hash1>([=]() {
    [[intel::fpga_register]] const size_t rounds = (leaf_cnt >> 1) - 1;

    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(32)]] uint32_t padded[32];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(8)]] uint32_t hash_state[8];
    [[intel::fpga_memory("BLOCK_RAM"),
      intel::bankwidth(4),
      intel::numbanks(16)]] uint32_t msg_schld[64];

    for (size_t r = 0; r < rounds; r++) {

      // get padded 1024 -bit input message over SYCL pipe
      [[intel::ivdep]] for (size_t i = 0; i < 32; i++)
      {
        padded[i] = ipipe1::read();
      }

      // compute sha256 on that input, keep output
      // in hash state ( 8 message words )
      sha256::hash(hash_state, msg_schld, padded);

      // finally send 8 message words ( = 256 -bit )
      // input over SYCL pipe, back to orchestractor kernel
      [[intel::ivdep]] for (size_t i = 0; i < 8; i++)
      {
        opipe1::write(hash_state[i]);
      }
    }
  });

  evt0.wait();

  return time_event(evt0);
}

}
