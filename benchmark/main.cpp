#include "benchmark.hpp"
#include <iomanip>
#include <iostream>

// default accelerator choice for benchmarking binary merklization
// implementation is FPGA h/w device
#if !(defined FPGA_EMU || defined FPGA_HW)
#define FPGA_HW
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
  sycl::queue q{ c, d, sycl::property::queue::enable_profiling{} };

  std::cout << "running on " << d.get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  constexpr size_t itr_cnt = 8;
  double* ts = static_cast<double*>(std::malloc(sizeof(double) * 3));

  std::cout << "Benchmarking SHA256 Binary Merklization FPGA implementation"
            << std::endl
            << std::endl;
  std::cout << std::setw(16) << std::right << "leaf count"
            << "\t\t" << std::setw(16) << std::right << "execution time"
            << "\t\t" << std::setw(16) << std::right << "host-to-device tx time"
            << "\t\t" << std::setw(16) << std::right << "device-to-host tx time"
            << std::endl;

  for (size_t i = 20; i <= 25; i++) {
    avg_kernel_exec_tm(q, 1ul << i, itr_cnt, ts);

    std::cout << std::setw(10) << std::right << 2 << " ^ " << i << "\t\t"
              << std::setw(22) << std::right << to_readable_timespan(ts[1])
              << "\t\t" << std::setw(22) << std::right
              << to_readable_timespan(ts[0]) << "\t\t" << std::setw(22)
              << std::right << to_readable_timespan(ts[2]) << std::endl;
  }

  std::free(ts);

  return EXIT_SUCCESS;
}
