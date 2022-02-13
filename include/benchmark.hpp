#pragma once
#include "merklize.hpp"

// For given many leaf nodes of some binary merkle tree, computes all
// intermediates on accelerator, while input leaves are explicitly
// transferred from host to device over PCIe & after completion of
// computation of all intermediates, those are transferred back to host over
// PCIe interface
//
// Last parameter of this function will return execution time of three
// operations, in following order
//
// - host -> device data tx time
// - kernel exec time
// - device -> host data tx time
//
// Note, ensure that queue has profiling enabled
void
benchmark_merklize(sycl::queue& q,
                   const size_t leaf_cnt,
                   sycl::cl_ulong* const ts)
{
  const size_t i_size = leaf_cnt << 5;
  const size_t o_size = i_size;

  // acquire resources
  uint32_t* i_d = static_cast<uint32_t*>(sycl::malloc_device(i_size, q));
  uint32_t* o_d = static_cast<uint32_t*>(sycl::malloc_device(o_size, q));
  uint32_t* i_h = static_cast<uint32_t*>(std::malloc(i_size));
  uint32_t* o_h = static_cast<uint32_t*>(std::malloc(o_size));

  memset(i_h, 0xff, i_size);

  sycl::event evt0 = q.memcpy(i_d, i_h, i_size);
  evt0.wait();

  // waiting for completion of computation of all intermediates
  sycl::cl_ulong tm = merklize::merklize(q, leaf_cnt, i_d, i_size, o_d, o_size);

  sycl::event evt1 = q.memcpy(o_h, o_d, o_size);
  evt1.wait();

  // release resources
  sycl::free(i_d, q);
  sycl::free(o_d, q);
  std::free(i_h);
  std::free(o_h);

  ts[0] = time_event(evt0);
  ts[1] = tm;
  ts[2] = time_event(evt1);
}

// Executes SHA256 binary merklization kernels with same input size `itr_cnt`
// -many times and computes average execution time of following SYCL commands
//
// - host -> device input tx time
// - kernel execution time
// - device -> host output tx time
void
avg_kernel_exec_tm(sycl::queue& q,
                   const size_t leaf_cnt,
                   const size_t itr_cnt,
                   double* const ts)
{
  constexpr size_t ts_size = sizeof(sycl::cl_ulong) * 3;

  // allocate memory on host ( for keeping exec time of enqueued commands )
  sycl::cl_ulong* ts_sum = static_cast<sycl::cl_ulong*>(std::malloc(ts_size));
  sycl::cl_ulong* ts_rnd = static_cast<sycl::cl_ulong*>(std::malloc(ts_size));

  // so that average execution/ data transfer time can be safely computed !
  std::memset(ts_sum, 0, ts_size);

  for (size_t i = 0; i < itr_cnt; i++) {
    benchmark_merklize(q, leaf_cnt, ts_rnd);

    ts_sum[0] += ts_rnd[0];
    ts_sum[1] += ts_rnd[1];
    ts_sum[2] += ts_rnd[2];
  }

  for (size_t i = 0; i < 3; i++) {
    ts[i] = (double)ts_sum[i] / (double)itr_cnt;
  }

  // deallocate resources
  std::free(ts_sum);
  std::free(ts_rnd);
}

// Convert nanosecond granularity execution time to readable string i.e. in
// terms of seconds/ milliseconds/ microseconds/ nanoseconds
std::string
to_readable_timespan(double ts)
{
  return ts >= 1e9 ? std::to_string(ts * 1e-9) + " s"
                   : ts >= 1e6 ? std::to_string(ts * 1e-6) + " ms"
                               : ts >= 1e3 ? std::to_string(ts * 1e-3) + " us"
                                           : std::to_string(ts) + " ns";
}
