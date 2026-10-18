cd 3rd/spmm_cnic
./build.sh
cd ../..

rm -rf build

#cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -B build
cmake -B build

cd build

make -j
