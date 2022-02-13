# sha256-fpga
SHA256 based Binary Merklization on FPGA

## Job Submission

For easing FPGA h/w compilation/ execution job submissions on Intel Devcloud platform, I've prepared following scripts.

### Compilation Flow

Create job submission bash script

```bash
touch build_fpga_bench_hw.sh
```

And populate it with following content

```bash
#!/bin/bash

# file name: build_fpga_hw.sh

# env setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware compilation
time make fpga_hw_bench
```

Now submit compilation job targeting Intel Arria 10 board

```bash
qsub -l nodes=1:fpga_compile:ppn=2 -l walltime=24:00:00 -d . build_fpga_bench_hw.sh

# note down job id e.g. 1850154
```

**Note :** If you happen to be interested in targeting Intel Stratix 10 board, consider using following compilation command instead of above Make build recipe.

```bash
# hardware compilation
time dpcpp -Wall -std=c++20 -I./include -O3 -DFPGA_HW -fintelfpga -Xshardware -Xsboard=intel_s10sx_pac:pac_s10 -reuse-exe=benchmark/fpga_hw.out benchmark/main.cpp -o benchmark/fpga_hw.out
```

And finally submit job on `fpga_compile` enabled VM.

### Execution Flow

Create job submission shell script

```bash
touch run_fpga_bench_hw.sh
```

And populate it with environment setup and binary execution commands

```bash
#!/bin/bash

# file name: run_fpga_hw.sh

# env setup
export PATH=/glob/intel-python/python2/bin/:${PATH}
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

# hardware image execution
pushd benchmark; ./fpga_hw.out; popd
```

Now submit execution job on VM, enabled with `fpga_runtime` capability & Intel Arria 10 board

```bash
qsub -l nodes=1:fpga_runtime:arria10:ppn=2 -d . run_fpga_bench_hw.sh -W depend=afterok:1850154

# use compilation flow job id ( e.g. 1850154 ) to create dependency chain
```

**Note :** If you compiled h/w image targeting Intel Stratix 10 board, consider using following job submission command

```bash
qsub -l nodes=1:fpga_runtime:stratix10:ppn=2 -d . run_fpga_bench_hw.sh -W depend=afterok:1850157

# place proper compilation job id ( e.g. 1850157 ), to form dependency chain
```

After completion of compilation/ execution job submission, consider checking status using

```bash
watch -n 1 qstat -n -1

# or just `qstat -n -1`
```

When completed, following command(s) should reveal newly created files, having stdout/ stderr output of compilation/ execution flow in `{build|run}_fpga_bench_hw.sh.{o|e}1850157` files

```bash
ls -lhrt   # created files shown towards end of list
git status # untracked, newly created files
```
