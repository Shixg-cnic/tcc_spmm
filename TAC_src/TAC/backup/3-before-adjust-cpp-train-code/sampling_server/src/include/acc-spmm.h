#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#include "acc-spmm/include/config.h"
#include "acc-spmm/include/class.h"