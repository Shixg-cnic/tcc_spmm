#pragma once
#include "column_q4.cuh"
#include "column_test_utils.cuh"
namespace test {
struct GPUQ4 {
    Buffer<unsigned> cols,offset,rows,vo,order;
    Buffer<std::uint64_t> bits;Buffer<float> values;Buffer<Q4Task> tasks;
    Buffer<std::uint8_t> cw,token;DeviceQ4 view;
    explicit GPUQ4(const Q4Format& m):cols((validateQ4Cache(m,q4CacheSlots(m.peakSlots)),m.cwCols)),offset(m.cwOffset),rows(m.tileRows),vo(m.valueOffset),order(m.tileOrder),bits(m.bitmap),values(m.values),tasks(m.tasks),cw(m.orderCW),token(m.rowToken),
        view{m.rows,m.cols,unsigned(m.cwOffset.size()-1),unsigned(m.tasks.size()),cols.data,offset.data,rows.data,vo.data,order.data,bits.data,values.data,tasks.data,cw.data,token.data,m.peakSlots}{}
};
}
