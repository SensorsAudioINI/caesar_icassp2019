# Pytorch + STAN = <3

## Setup

### Prequisites
- Python --> 2.7 (conversion to 3.6 coming soon)
- [CUDA](https://developer.nvidia.com/cuda-downloads) --> 8.0 
- [cuDNN](https://developer.nvidia.com/cudnn) --> 6.0 recommended

### conda
1. Install [anaconda2](https://www.anaconda.com/what-is-anaconda/) --> 5.0.0.1 recommended
```
wget https://repo.continuum.io/archive/Anaconda2-5.0.0.1-Linux-x86_64.sh
chmod u+x Anaconda2-5.0.0.1-Linux-x86_64.sh
./Anaconda2-5.0.0.1-Linux-x86_64.sh
```

### pytorch

1. create a conda environment with python 2.7
```
conda create -n stan_ctc_27 python=2.7
```

2. activate the environment
```
source activate stan_ctc_27
```

3. installing pytorch from conda
```
conda install pytorch torchvision cuda80 -c soumith
```

4. required packages for compiling warp ctc
```
conda install pyyaml yaml cmake
```

```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
```

5. remove any previous build folders
```
mkdir build
cd build
cmake ..
make
```
6. build pytorch bindings now
```
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"

conda install libgcc

export CUDA_HOME=/path/to/cuda
cd pytorch_binding
python setup.py install
```
