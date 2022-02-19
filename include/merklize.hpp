#pragma once
#include "sha256.hpp"
#include "utils.hpp"
#include <cassert>

namespace merklize {

#define SHA256KernelDecl(idx) class kernelSHA256Hash##idx
#define MerklizeKernelDecl(idx) class kernelMerklizationOrchestrator##idx

#define IPipeDecl(idx) class SHA256MessageWords##idx
#define OPipeDecl(idx) class SHA256DigestWords##idx

// Compile-time template for sending padded input message words to SHA256 kernel
#define IPipe(idx)                                                             \
  using ipipe##idx =                                                           \
    sycl::ext::intel::pipe<SHA256MessageWords##idx, uint32_t, 0>

// Compile-time template for sending output digest words from SHA256 kernel
#define OPipe(idx)                                                             \
  using opipe##idx = sycl::ext::intel::pipe<SHA256DigestWords##idx, uint32_t, 0>

// Generic ( in terms of which pipes are for communication, kernel identifier )
// SHA256 hash calculator kernel
#define ComputeSHA256(idx)                                                     \
  q.single_task<kernelSHA256Hash##idx>([=]() [[intel::kernel_args_restrict]] { \
    [[intel::fpga_register]] uint32_t padded[32];                              \
    [[intel::fpga_register]] uint32_t hash_state[8];                           \
    [[intel::fpga_register]] uint32_t msg_schld[64];                           \
    while (true) {                                                             \
      [[intel::ivdep]] for (size_t i = 0; i < 32; i++)                         \
      {                                                                        \
        padded[i] = ipipe##idx::read();                                        \
      }                                                                        \
                                                                               \
      sha256::hash(hash_state, msg_schld, padded);                             \
                                                                               \
      [[intel::ivdep]] for (size_t i = 0; i < 8; i++)                          \
      {                                                                        \
        opipe##idx::write(hash_state[i]);                                      \
      }                                                                        \
    }                                                                          \
  })

// Kernels predeclared to avoid name mangling in optimization report
//
SHA256KernelDecl(0);
SHA256KernelDecl(1);
SHA256KernelDecl(2);
SHA256KernelDecl(3);

MerklizeKernelDecl(0);
MerklizeKernelDecl(1);
MerklizeKernelDecl(2);

// Pipes predeclared to avoid name mangling in optimization report
//
IPipeDecl(0);
OPipeDecl(0);
IPipeDecl(1);
OPipeDecl(1);
IPipeDecl(2);
OPipeDecl(2);
IPipeDecl(3);
OPipeDecl(3);

// Orchestrator kernel(s) passing 1024 -bit (padded) input message ( = 32 words
// ) to compute kernel(s) over these pipe
IPipe(0);
IPipe(1);
IPipe(2);
IPipe(3);

// After computing SHA256 digest on 1024 -bit input message, 8 message words ( =
// 32 -bytes ) are passed back to orchestrator kernel(s), over these pipes
OPipe(0);
OPipe(1);
OPipe(2);
OPipe(3);

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

// Computes all intermediate nodes of Binary Merkle Tree using SHA256
// 2-to-1 hash function, where leaf node count is power of 2 value
//
// In this routine, kernel pairs ( orchestrator <-> sha256hash ) will be
// communicating over SYCL pipes, where orchestrator kernel which is
// responsible for driving multiple phases ( dependent on previously
// completed one ) of computation of intermediates of binary merkle tree,
// sends padded input message words ( = 32 ) over blocking SYCL pipe and
// waits for completion of SHA256 computation on compute kernel, which
// finally sends back 32 -bytes digest to orchestrator for placing it in
// proper position in output memory allocation (on global memory), which
// will again be used in next level of intermediate node computation, if
// not in root level of tree
//
// Ensure that SYCL queue has profiling enabled, as at successful
// completion of this routine it returns time spent in computing all
// intermediate nodes of binary merkle tree
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

  sycl::event evt0 = q.single_task<kernelMerklizationOrchestrator0>([=]() {
    sycl::device_ptr<uint32_t> leaves_ptr{ leaves };
    sycl::device_ptr<uint32_t> intermediates_ptr{ intermediates };

    // on-chip (kernel) private memory allocation with attributes
    // such that enough read/ write ports are available for facilitating
    // parallel (stall-free, preferrably) access
    [[intel::fpga_register]] uint32_t msg[16];
    [[intel::fpga_register]] uint32_t padded[32];
    [[intel::fpga_register]] uint32_t digest[8];

    const size_t i_offset = 0;
    const size_t o_offset = (leaf_cnt >> 1) << 3;
    const size_t itr_cnt = leaf_cnt >> 2;

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

      if (itr_cnt > 1) {
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

      if (itr_cnt > 1) {
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

    // these many levels of intermediate nodes ( excluding root of tree
    // ) remaining to be computed, where (i+1)-th level is dependent on
    // i-th level, while indexing is done bottom up
    const size_t rounds = bin_log(leaf_cnt >> 2);

    for (size_t r = 0; r < rounds; r++) {
      const size_t i_offset = (leaf_cnt << 2) >> r;
      const size_t o_offset = i_offset >> 1;
      const size_t itr_cnt = leaf_cnt >> (r + 3);

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

        if (itr_cnt > 1) {
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

        if (itr_cnt > 1) {
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
    }
  });

  sycl::event evt1 = q.single_task<kernelMerklizationOrchestrator1>([=]() {
    sycl::device_ptr<uint32_t> leaves_ptr{ leaves };
    sycl::device_ptr<uint32_t> intermediates_ptr{ intermediates };

    // on-chip (kernel) private memory allocation with attributes
    // such that enough read/ write ports are available for facilitating
    // parallel (stall-free, preferrably) access
    [[intel::fpga_register]] uint32_t msg[16];
    [[intel::fpga_register]] uint32_t padded[32];
    [[intel::fpga_register]] uint32_t digest[8];

    const size_t i_offset = (leaf_cnt >> 1) << 3;
    const size_t o_offset = i_offset + (i_offset >> 1);
    const size_t itr_cnt = leaf_cnt >> 2;

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
        ipipe2::write(padded[j]);
      }

      // ---

      if (itr_cnt > 1) {
#pragma unroll 16 // 512 -bit burst coalesced global memory read
        for (size_t j = 0; j < 16; j++) {
          msg[j] = leaves_ptr[i_offset_1 + j];
        }

        sha256::pad_input_message(msg, padded);

        // send 32 padded input message words to compute kernel
        [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
        {
          ipipe3::write(padded[j]);
        }
      }

      // ---

      [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
      {
        digest[j] = opipe2::read();
      }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
      for (size_t j = 0; j < 8; j++) {
        intermediates_ptr[o_offset_0 + j] = digest[j];
      }

      // ---

      if (itr_cnt > 1) {
        [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
        {
          digest[j] = opipe3::read();
        }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
        for (size_t j = 0; j < 8; j++) {
          intermediates_ptr[o_offset_1 + j] = digest[j];
        }
      }
    }

    // these many levels of intermediate nodes ( excluding root of tree
    // ) remaining to be computed, where (i+1)-th level is dependent on
    // i-th level, while indexing is done bottom up
    const size_t rounds = bin_log(leaf_cnt >> 2);

    for (size_t r = 0; r < rounds; r++) {
      const size_t i_offset = ((leaf_cnt << 2) + (leaf_cnt << 1)) >> r;
      const size_t o_offset = i_offset >> 1;
      const size_t itr_cnt = leaf_cnt >> (r + 3);

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
          ipipe2::write(padded[j]);
        }

        // ---

        if (itr_cnt > 1) {
#pragma unroll 16 // 512 -bit burst coalesced global memory read
          for (size_t j = 0; j < 16; j++) {
            msg[j] = intermediates_ptr[i_offset_1 + j];
          }

          sha256::pad_input_message(msg, padded);

          // send 32 padded input message words to compute kernel
          [[intel::ivdep]] for (size_t j = 0; j < 32; j++)
          {
            ipipe3::write(padded[j]);
          }
        }

        // ---

        [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
        {
          digest[j] = opipe2::read();
        }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
        for (size_t j = 0; j < 8; j++) {
          intermediates_ptr[o_offset_0 + j] = digest[j];
        }

        // ---

        if (itr_cnt > 1) {
          [[intel::ivdep]] for (size_t j = 0; j < 8; j++)
          {
            digest[j] = opipe3::read();
          }

#pragma unroll 8 // 256 -bit burst coalesced global memory write
          for (size_t j = 0; j < 8; j++) {
            intermediates_ptr[o_offset_1 + j] = digest[j];
          }
        }
      }
    }
  });

  // --- compute root of merkle tree ---
  sycl::event evt2 = q.submit([&](sycl::handler& h) {
    h.depends_on({ evt0, evt1 });

    h.single_task<kernelMerklizationOrchestrator2>([=]() {
      sycl::device_ptr<uint32_t> intermediates_ptr{ intermediates };

      // on-chip (kernel) private memory allocation with attributes
      // such that enough read/ write ports are available for facilitating
      // parallel (stall-free, preferrably) access
      [[intel::fpga_register]] uint32_t msg[16];
      [[intel::fpga_register]] uint32_t padded[32];
      [[intel::fpga_register]] uint32_t hash_state[8];
      [[intel::fpga_register]] uint32_t msg_schld[64];

#pragma unroll 16 // 512 -bit burst coalesced global memory read
      for (size_t j = 0; j < 16; j++) {
        msg[j] = intermediates_ptr[16 + j];
      }

      sha256::pad_input_message(msg, padded);
      sha256::hash(hash_state, msg_schld, padded);

#pragma unroll 8 // 256 -bit burst coalesced global memory write
      for (size_t j = 0; j < 8; j++) {
        intermediates_ptr[8 + j] = hash_state[j];
      }
    });
  });

  sycl::event evt3 = ComputeSHA256(0);
  sycl::event evt4 = ComputeSHA256(1);
  sycl::event evt5 = ComputeSHA256(2);
  sycl::event evt6 = ComputeSHA256(3);

  evt2.wait();

  return ((time_event(evt0) + time_event(evt1)) >> 1) + time_event(evt2);
}
}
