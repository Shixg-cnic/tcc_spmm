#include "column_test_utils.cuh"
using namespace test;

void run(const Matrix& matrix,unsigned f,int versions,cudaStream_t stream,unsigned swSize=COLUMN_SW_SIZE) {
    auto column=convertColumn(matrix.rows,matrix.cols,matrix.csr,swSize);
    decode(column,matrix.csr);
    GPUColumn gc(column); GPURow gr(convertRow(matrix.rows,matrix.cols,matrix.csr));
    auto x=features(matrix.cols,f);
    auto expected=reference(matrix,x,f,true),fp32=reference(matrix,x,f,false);
    Buffer<float> input(x),y(std::size_t(matrix.rows)*f),row(std::size_t(matrix.rows)*f);
    gr.launch(input.data,row.data,f,stream); auto rowOutput=row.host(stream);
    auto rowError=error(rowOutput,expected);
    if(rowError.rmse>=2e-4) {
        std::cerr<<"Row mismatch F="<<f<<" rmse="<<rowError.rmse<<" max="<<rowError.maxAbsolute<<'\n';
        int shown=0;
        for(std::size_t i=0;i<rowOutput.size() && shown<12;++i)
            if(std::abs(rowOutput[i]-expected[i])>.01) { std::cerr<<i/f<<','<<i%f<<": "<<rowOutput[i]<<" expected "<<expected[i]<<'\n'; ++shown; }
    }
    verify(rowError);
    for(int v=0;v<versions;++v) {
        auto launch=[&]() { check(launchColumnSpMM(gc.view,input.data,y.data,f,ColumnVersion(v),stream)); };
        launch(); auto output=y.host(stream); auto numeric=error(output,expected); verify(numeric);
        verify(error(output,rowOutput)); auto quant=error(output,fp32);
        double variation=0;
        for(int r=0;r<3;++r) { launch(); variation=std::max(variation,error(y.host(stream),output).maxAbsolute); }
        // Graph replays must overwrite Y, including empty rows, on each operation.
        measure(launch,3,stream); verify(error(y.host(stream),expected));
        std::cout<<"M="<<matrix.rows<<" K="<<matrix.cols<<" F="<<f<<" SW="<<swSize<<" C"<<v
            <<" relative_rmse_tf32="<<numeric.rmse<<" max_abs_tf32="<<numeric.maxAbsolute
            <<" relative_rmse_fp32="<<quant.rmse<<" max_abs_fp32="<<quant.maxAbsolute
            <<" repeat_max_abs="<<variation<<'\n';
    }
}
int main(int argc,char** argv) {
    try {
        int versions=argc>1?std::stoi(argv[1]):4;
        require(versions>=1 && versions<=4,"version count must be 1..4");
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto weighted=makeMatrix(19,269,[](unsigned r,unsigned c) {
            return ((r<8 && c<8) || ((r*31+c*7)%13<3)) ? float(int((r*17+c*3)%41)-20)/19.f:0.f;
        });
        for(unsigned f:{1u,17u,32u,64u,100u,128u,256u,512u,513u,544u,1024u}) run(weighted,f,versions,stream);
        run(makeMatrix(9,259,[](unsigned,unsigned){return 0.f;}),100,versions,stream);
        run(makeMatrix(17,273,[](unsigned r,unsigned c){return r==16 && c==272 ? -1.25f:0.f;}),17,versions,stream);
        // Empty SW between populated windows; all 64 bitmap bits including bit63.
        run(makeMatrix(65,401,[](unsigned r,unsigned c){return c<8 || c>=384 ? 1.f:0.f;}),64,versions,stream);
        for(unsigned rows:{7u,COLUMN_CACHE_SIZE,COLUMN_CACHE_SIZE+9}) {
            auto repeated=makeMatrix(rows,265,[](unsigned,unsigned c){return c%8==0?1.f:0.f;});
            auto format=convertColumn(rows,265,repeated.csr);
            require(format.superWindowCachedCount[0]==std::min(rows,COLUMN_CACHE_SIZE),"cache selection count mismatch");
            run(repeated,100,versions,stream);
        }
        run(weighted,32,versions,stream,1); run(weighted,128,versions,stream,3);
        std::uint32_t ptr[]{0,4,4},src[]{8,0,8,0};float weights[]{1.f,.25f,-1.f,.5f};
        auto merged=canonicalCSR(2,19,4,ptr,src,weights);
        require(merged.src.size()==1 && merged.values[0]==.75f,"coalescing failed");
        run({2,19,merged},17,versions,stream);
        bool rejected=false;
        try { auto raw=weighted.csr; raw.values[0]=0; convertColumn(weighted.rows,weighted.cols,raw); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"noncanonical input accepted");
        check(cudaStreamDestroy(stream));
        std::cout<<"All Column C0..C"<<versions-1<<" correctness and format decode tests passed.\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
