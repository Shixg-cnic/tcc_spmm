// Standalone output-stripe Column vs original Acc Basic, FP32 F=128/256.
// This is an experiment, not a claim of performance improvement.
// Usage: ./column_striped_test graph.csr 128 65536 [device=0] [X.bin]
#include "column_striped.cuh"
#include "column_test_utils.cuh"
#include "matrix_io.cuh"
#include "acc_reference.cuh"
#include <algorithm>
#include <iomanip>

using namespace test;
using U=std::uint32_t;

int main(int argc,char** argv) {
    try {
        if(argc<2) {
            std::cerr<<"Usage: column_striped_test matrix.csr [F=128|256] [stripeRows=65536] [device=0] [X.bin]\n";
            return 1;
        }
        U F=argc>2?U(std::stoul(argv[2])):128;
        U band=argc>3?U(std::stoul(argv[3])):65536;
        if((F!=128 && F!=256) || !band) throw std::invalid_argument("F must be 128/256 and stripeRows > 0");
        check(cudaSetDevice(argc>4?std::stoi(argv[4]):0));
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto mat=readMatrix(argv[1]);
        auto X=readFeatures(argc>5?argv[5]:nullptr,mat.cols,F);
        auto base=buildColumnBase(mat.rows,mat.cols,mat.csr);
        auto stripe=makeColumnStriped(base,band);
        Buffer<U> offsets(stripe.stripeCwOffset),rows(stripe.tileRows),vo(stripe.valueOffset);
        Buffer<std::uint64_t> bitmap(stripe.bitmap);
        Buffer<float> values(stripe.values),x(X),out(std::size_t(mat.rows)*F);
        DeviceColumnStriped device{stripe.rows,stripe.cols,stripe.stripeRows,stripe.stripes(),stripe.cws(),
            offsets.data,rows.data,vo.data,bitmap.data,values.data};
        U padded=U((std::uint64_t(mat.rows)+7)/8*8);
        auto acc=convertAccRow(mat.rows,mat.cols,mat.csr);
        Buffer<U> arw(acc.rowWindowOffset),ao(acc.tcOffset),am(acc.sparseA2B);
        Buffer<std::uint64_t> ab(acc.tcLocalBit);
        Buffer<float> av(acc.values),accY(std::size_t(padded)*F);
        auto runAcc=[&]() {
            check(cudaMemsetAsync(accY.data,0,std::size_t(padded)*F*sizeof(float),stream));
            check(launchAccRowSpMM(arw.data,ao.data,am.data,ab.data,av.data,x.data,accY.data,padded,mat.cols,F,stream));
        };
        auto runStripe=[&](bool wide) {
            check(launchColumnStriped(device,x.data,out.data,F,wide,stream,true));
        };
        runAcc();runStripe(false);check(cudaStreamSynchronize(stream));
        auto truth=accY.host(stream);
        auto checkOutput=[&](const char* label) {
            auto result=out.host(stream);
            std::vector<float> valid(truth.begin(),truth.begin()+std::size_t(mat.rows)*F);
            auto err=error(result,valid);
            std::cout<<label<<" rmse_vs_Acc="<<err.rmse<<" maxAbs="<<err.maxAbsolute<<"\n";
            require(err.rmse<2e-4 && err.maxAbsolute<2e-4+err.maxReference*5e-4,
                "striped column does not match Acc");
        };
        checkOutput("striped_scalar");
        runStripe(true);check(cudaStreamSynchronize(stream));checkOutput("striped_float4");
        auto measureRun=[&](const std::function<void()>& fn) {return measure(fn,5,stream);};
        double a=measureRun(runAcc),b=measureRun([&](){runStripe(false);});
        double c=measureRun([&](){runStripe(true);});
        std::cout<<std::setprecision(9)
            <<"dataset="<<argv[1]<<" F="<<F<<" stripeRows="<<band
            <<" stripes="<<stripe.stripes()<<" cws="<<stripe.cws()
            <<" original_tiles="<<base.tileLocalBit.size()
            <<" striped_tiles="<<stripe.bitmap.size()
            <<"\nmethod,complete_ms,ratio_to_Acc\n"
            <<"Acc,"<<a<<",1\n"
            <<"ColumnStripedScalar,"<<b<<","<<b/a<<"\n"
            <<"ColumnStripedFloat4,"<<c<<","<<c/a<<"\n";
        check(cudaStreamDestroy(stream));
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
    return 0;
}
