#!/bin/bash
grep -n 'reference\|readFeatures\|struct Matrix\|struct Error\|inline Error\|Buffer\|readMatrix\|loadCSR\|require' /home/shixingguo/work/GNN/tcc_spmm/tests/matrix_io.cuh | head -30
