#include "format/int8_features.cuh"
namespace {
using U=std::uint32_t;
__global__ void quantize(const float* x,std::int8_t* q,float* scale,U f) {
    __shared__ float maxima[128];
    U row=blockIdx.x,t=threadIdx.x;float m=0;
    for(U j=t;j<f;j+=blockDim.x)m=fmaxf(m,fabsf(x[std::size_t(row)*f+j]));
    maxima[t]=m;__syncthreads();
    for(U step=64;step;step>>=1){if(t<step)maxima[t]=fmaxf(maxima[t],maxima[t+step]);__syncthreads();}
    float s=maxima[0]==0?1:maxima[0]/127.f;
    if(t==0)scale[row]=s;
    for(U j=t;j<f;j+=blockDim.x){auto i=std::size_t(row)*f+j;int v=lrintf(x[i]/s);q[i]=std::int8_t(v<-127?-127:(v>127?127:v));}
}
__global__ void dequantize(const std::int8_t* q,const float* scale,float* x,std::size_t n,U f) {
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)x[i]=float(q[i])*scale[i/f];
}
}
cudaError_t quantizeFeaturesInt8(const float* x,std::int8_t* q,float* scale,U n,U f,cudaStream_t stream) {
    if(!x||!q||!scale||!n||!f)return cudaErrorInvalidValue;
    quantize<<<n,128,0,stream>>>(x,q,scale,f);return cudaGetLastError();
}
cudaError_t dequantizeFeaturesInt8(const std::int8_t* q,const float* scale,float* x,U n,U f,cudaStream_t stream) {
    if(!x||!q||!scale||!n||!f)return cudaErrorInvalidValue;
    auto size=std::size_t(n)*f;
    dequantize<<<unsigned((size+255)/256),256,0,stream>>>(q,scale,x,size,f);return cudaGetLastError();
}
