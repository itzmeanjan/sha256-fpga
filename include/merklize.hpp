#pragma once
#include "sha256.hpp"
#include <cassert>

namespace merklize {

class kernelSHA256Merklization;
class kernelMerklizationOrchestrator;

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

// 1024 -bit (padded) input message ( = 32 words ) passed to kernel computing
// SHA256 2-to-1 digest, over this pipe
using ipipe = sycl::ext::intel::pipe<class SHA256MessageWords, uint32_t, 32>;
// After computing SHA256 digest on 1024 -bit input message, 8 message words are
// passed back to orchestrator kernel, over this pipe
using opipe = sycl::ext::intel::pipe<class SHA256DigestWords, uint32_t, 8>;

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
void
merklize(sycl::queue& q,
         const size_t leaf_cnt,
         uint32_t* const __restrict leaves,
         const size_t i_size,
         uint32_t* const __restrict intermediates,
         const size_t o_size)
{
  assert((i_size << 1) == o_size);          // ensure enough memory allocated
  assert((leaf_cnt & (leaf_cnt - 1)) == 0); // ensure power of 2

  sycl::event evt0 = q.single_task<kernelMerklizationOrchestrator>([=]() {
    sycl::device_ptr<uint32_t> leaves_ptr{ leaves };
    sycl::device_ptr<uint32_t> intermediates_ptr{ intermediates };

    [[intel::fpga_register]] size_t i_offset = 0;
    [[intel::fpga_register]] size_t o_offset = (leaf_cnt >> 1) << 3;
    [[intel::fpga_register]] const size_t itr_cnt = leaf_cnt >> 1;

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

    // compute intermediate nodes which are living just above input leaf nodes
    [[intel::ivdep]] for (size_t i = 0; i < itr_cnt; i++)
    {
#pragma unroll 16 // 512 -bit burst coalesced global memory read
      for (size_t j = 0; j < 16; j++) {
        msg[j] = leaves_ptr[i_offset + j];
      }

      i_offset += 16;

      // padding input message such that padded message (bit) length
      // is evenly divisible by 512 ( = sha256 message block bit length )
      sha256::pad_input_message(msg, padded);

      // send 32 padded input message words to compute kernel
      [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
      {
        ipipe::write(padded[j]);
      }

      // wait for reception of sha256 digest of 512 -bit input ( imagine two
      // sha256 digests concatenated ), in form of 8 message words
      [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
      {
        digest[j] = opipe::read();
      }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
      for (size_t j = 0; j < 8; j++) {
        intermediates_ptr[o_offset + j] = digest[j];
      }

      o_offset += 8;
    }

    // these many levels of intermediate nodes ( including root of tree )
    // remaining to be computed
    [[intel::fpga_register]] const size_t rounds = bin_log(leaf_cnt >> 1);

    // each level of tree ( bottom up ) depends on completion of computation of
    // previous round --- data dependency !
    for (size_t r = 0; r < rounds; r++) {
      [[intel::fpga_register]] size_t rd_offset = ((leaf_cnt >> 1) << 3) >> r;
      [[intel::fpga_register]] size_t wr_offset = rd_offset >> 1;
      [[intel::fpga_register]] const size_t rnd_itr_cnt = leaf_cnt >> (r + 2);

      // these many intermediate nodes to be computed in this level of tree,
      // where each node computation is independent of any other node living on
      // same level of tree
      [[intel::ivdep]] for (size_t i = 0; i < rnd_itr_cnt; i++)
      {
#pragma unroll 16 // 512 -bit burst coalesced global memory read
        for (size_t j = 0; j < 16; j++) {
          msg[j] = intermediates_ptr[rd_offset + j];
        }

        rd_offset += 16;

        // padding input message ( = 64 -bytes, imagine two sha256 digests
        // concatenated ), such that padded input ( bit ) length is evenly
        // divisible by 512 ( = sha256 message block bit length )
        sha256::pad_input_message(msg, padded);

        [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
        {
          ipipe::write(padded[j]);
        }

        [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
        {
          digest[j] = opipe::read();
        }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
        for (size_t j = 0; j < 8; j++) {
          intermediates_ptr[wr_offset + j] = digest[j];
        }

        wr_offset += 8;
      }
    }
  });

  sycl::event evt1 = q.single_task<kernelSHA256Merklization>([=]() {
    // For computing all intermediate nodes of Binary Merkle Tree with N
    // leaf nodes, where N is power of 2, `rounds` -many sha256 intermediate
    // nodes to be computed, using sha256 2-to-1 hash function
    [[intel::fpga_register]] const size_t rounds = leaf_cnt - 1ul;

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
        padded[i] = ipipe::read();
      }

      // compute sha256 on that input, keep output
      // in hash state ( 8 message words )
      sha256::hash(hash_state, msg_schld, padded);

      // finally send 8 message words ( = 256 -bit )
      // input over SYCL pipe, back to orchestractor kernel
      [[intel::ivdep]] for (size_t i = 0; i < 8; i++)
      {
        opipe::write(hash_state[i]);
      }
    }
  });

  q.ext_oneapi_submit_barrier({ evt0, evt1 }).wait();
}

}
