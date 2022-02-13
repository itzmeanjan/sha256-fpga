#pragma once
#include <CL/sycl.hpp>

// Given four contiguous big endian bytes, this routine interprets those as 32
// -bit unsigned integer
static inline const uint32_t
from_be_bytes(sycl::private_ptr<uint8_t> in)
{
  return (static_cast<uint32_t>(in[0]) << 24) |
         (static_cast<uint32_t>(in[1]) << 16) |
         (static_cast<uint32_t>(in[2]) << 8) |
         (static_cast<uint32_t>(in[3]) << 0);
}

// Given 32 -bit unsigned integer, this routine interprets that as four
// contiguous big endian bytes
static inline void
to_be_bytes(const uint32_t word, sycl::private_ptr<uint8_t> out)
{
  out[0] = static_cast<uint32_t>((word >> 24) & 0xff);
  out[1] = static_cast<uint32_t>((word >> 16) & 0xff);
  out[2] = static_cast<uint32_t>((word >> 8) & 0xff);
  out[3] = static_cast<uint32_t>((word >> 0) & 0xff);
}

// Execution time ( in nanosecond level granularity ) of command, whose
// submission resulted into supplied event
//
// Note, ensure that profiling is enabled on SYCL queue !
static inline const sycl::cl_ulong
time_event(sycl::event evt)
{
  const size_t start =
    evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  const size_t end =
    evt.get_profiling_info<sycl::info::event_profiling::command_end>();

  return end - start;
}
