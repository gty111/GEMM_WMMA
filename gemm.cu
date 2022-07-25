// GEMM compare between bmma wmma cutlass_basic_gemm
// wmma warp level matrix multiply-accumulate
// bmma block level matrix multiply-accumulate
#include "cutlass/gemm/device/gemm.h"
#include "helper.h"
#include <mma.h>
#include <vector>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <functional>
using namespace nvcuda; 
#define PAD(X,Y) (X % Y ? (X/Y+1)*Y : X)

#define BLOCK_DIM_DEFAULT 512
#define WARP_SIZE 32
#define TIMES 5 // must greater than 1 to warm up for unify memory

template <typename TIN,typename TOUT,
      int M_TILE,int N_TILE,int K_TILE>
__global__ void wmma_kernel(TIN *a, TIN *b, TOUT *c,
      int M_PAD,int N_PAD,int K_PAD) {
   int idx,midx,nidx,ndim,kdim;
   ndim = N_PAD / N_TILE;
   kdim = K_PAD / K_TILE;
   idx = (blockIdx.x*blockDim.x+threadIdx.x)/WARP_SIZE;
   nidx = idx%ndim;
   midx = idx/ndim;
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, TOUT> c_frag;
   
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);
   
   TOUT *c_unique = c + nidx*N_TILE + midx*M_TILE*ndim*N_TILE;
   
   for(int kidx=0;kidx<kdim;kidx++){

      // Load the inputs
      TIN *a_unique = a + kidx*K_TILE + midx*M_TILE*kdim*K_TILE;
      TIN *b_unique = b + nidx*N_TILE + kidx*K_TILE*ndim*N_TILE;
      
      wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
      wmma::load_matrix_sync(b_frag, b_unique, N_PAD);
      
      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   // Store the output
   wmma::store_matrix_sync(c_unique, c_frag, N_PAD, wmma::mem_row_major);
}

template <typename TIN,typename TOUT,int M_TILE,int N_TILE,int K_TILE>
__global__ void bmma_kernel(TIN *a, TIN *b, TOUT *c,
      int M_PAD,int N_PAD,int K_PAD) {
   const int nwarp = BLOCK_DIM_DEFAULT/WARP_SIZE;
   const int C_TILE_SIZE = M_TILE * N_TILE;
   __shared__ TOUT shm[M_TILE][nwarp*N_TILE];
   const int ndim = N_PAD / N_TILE;
   const int kdim = K_PAD / K_TILE;
   const int warpidx = threadIdx.x/WARP_SIZE;
   const int nidx = blockIdx.x%ndim;
   const int midx = blockIdx.x/ndim;
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, TIN, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, TOUT> c_frag;
   
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);
   
   const int base = nidx*N_TILE + midx*ndim*C_TILE_SIZE;
   TOUT *c_unique = c + base;
   
   for(int kidx=0;kidx<kdim;kidx++){
      if(kidx % nwarp != warpidx)continue;
      // Load the inputs
      TIN *a_unique = a + kidx*K_TILE + midx*M_TILE*kdim*K_TILE;
      TIN *b_unique = b + nidx*N_TILE + kidx*K_TILE*ndim*N_TILE;
      
      wmma::load_matrix_sync(a_frag, a_unique, K_PAD);
      wmma::load_matrix_sync(b_frag, b_unique, N_PAD);
      
      // Perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
   }
   // Store the output
   wmma::store_matrix_sync(&shm[0][warpidx*N_TILE],
       c_frag, nwarp*N_TILE, wmma::mem_row_major);
   __syncthreads();
   for(int i=warpidx;i<C_TILE_SIZE;i+=nwarp){
         c_unique[i/N_TILE*ndim*N_TILE+i%N_TILE] = 0;
      for(int j=0;j<nwarp;j++){
         c_unique[i/N_TILE*ndim*N_TILE+i%N_TILE] += 
            shm[i/N_TILE][i%N_TILE+j*N_TILE];
      }
   }
}      


//unify memory
template <typename T> struct cuda_data {
  T *data;

  cuda_data(size_t n) {
    cudaMallocManaged(&data, sizeof(T) * n);
    //init to zero
    for(long i=0;i<n;i++){
      data[i] = 0;
    }
  }
  ~cuda_data() { cudaFree(data); }
};

enum DIR {ARR2CUARR,CUARR2ARR};

template <typename TARR,typename TCUARR,DIR dir>
void copy(int ARR_M,int ARR_N,TARR *arr,
      int CUARR_M,int CUARR_N,cuda_data<TCUARR> &cuarr){
   assert(CUARR_M>=ARR_M && CUARR_N>=ARR_N);
   if(dir==ARR2CUARR){
      for(int i=0;i<ARR_M;i++)
      for(int j=0;j<ARR_N;j++){
         cuarr.data[i*CUARR_N+j] = arr[i*ARR_N+j];
      }
   }else if(dir==CUARR2ARR){   
      for(int i=0;i<ARR_M;i++){
         for(int j=0;j<ARR_N;j++){
            arr[i*ARR_N+j] = cuarr.data[i*CUARR_N+j];
         }
      }
   }else assert(0);
}

void Timer(const char *tag, const std::function<void()> &kernel,
               int test_time = TIMES) {
  float min_time = 9e99;
  for (int i = 0; i < test_time; ++i) {
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    kernel();
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    min_time = std::min(min_time, elapsed_time);
    std::printf("[%s] iter %d: %f ms elapsed, %f ms min.\n", tag, i,
                elapsed_time, min_time);
  }
}

cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    printf("cutlass error\n");
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

template <typename TIN, typename TOUT,
         typename TGEMMIN=half, typename TGEMMOUT=float,
         int M_TILE=16,int N_TILE=16,int K_TILE=16>
void GEMM_wmma(int M,int N,int K,TIN *a_in,TIN *b_in,TOUT *c_out){
   assert(M!=0 && N!=0 && K!=0);
   
   const int M_PAD = PAD(M,M_TILE) ;
   const int N_PAD = PAD(N,N_TILE) ;
   const int K_PAD = PAD(K,K_TILE) ;

   cuda_data<TGEMMIN> a(M_PAD*K_PAD),b(K_PAD*N_PAD);
   cuda_data<TGEMMOUT> c(M_PAD*N_PAD);

   //init a b
   copy<TIN,TGEMMIN,ARR2CUARR>(M,K,a_in,M_PAD,K_PAD,a);
   copy<TIN,TGEMMIN,ARR2CUARR>(K,N,b_in,K_PAD,N_PAD,b);

   int GRID_DIM,BLOCK_DIM,nwarp;
   nwarp = (M_PAD/M_TILE) * (N_PAD/N_TILE);
   if(nwarp*WARP_SIZE < BLOCK_DIM_DEFAULT){
      GRID_DIM = 1;
      BLOCK_DIM = nwarp*WARP_SIZE;
   }else{
      GRID_DIM = (nwarp*WARP_SIZE)%BLOCK_DIM_DEFAULT ? 
         nwarp*WARP_SIZE/BLOCK_DIM_DEFAULT+1 : nwarp*WARP_SIZE/BLOCK_DIM_DEFAULT ;
      BLOCK_DIM = BLOCK_DIM_DEFAULT;
   }
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);
   Timer("gemm_gty_wmma", [&]{
   wmma_kernel<TGEMMIN,TGEMMOUT,M_TILE,N_TILE,K_TILE>
      <<<GRID_DIM,BLOCK_DIM>>>(a.data,b.data,c.data,
            M_PAD,N_PAD,K_PAD);});

   copy<TOUT,TGEMMOUT,CUARR2ARR>(M,N,c_out,M_PAD,N_PAD,c);
}

template <typename TIN, typename TOUT,
         typename TGEMMIN=half, typename TGEMMOUT=float,
         int M_TILE=16,int N_TILE=16,int K_TILE=16>
void GEMM_bmma(int M,int N,int K,TIN *a_in,TIN *b_in,TOUT *c_out){
   assert(M!=0 && N!=0 && K!=0);
   
   const int M_PAD = PAD(M,M_TILE) ;
   const int N_PAD = PAD(N,N_TILE) ;
   const int K_PAD = PAD(K,K_TILE) ;

   cuda_data<TGEMMIN> a(M_PAD*K_PAD),b(K_PAD*N_PAD);
   cuda_data<TGEMMOUT> c(M_PAD*N_PAD);

   //init a b
   copy<TIN,TGEMMIN,ARR2CUARR>(M,K,a_in,M_PAD,K_PAD,a);
   copy<TIN,TGEMMIN,ARR2CUARR>(K,N,b_in,K_PAD,N_PAD,b);

   int GRID_DIM,BLOCK_DIM;
   GRID_DIM = (M_PAD/M_TILE) * (N_PAD/N_TILE);
   BLOCK_DIM = BLOCK_DIM_DEFAULT;
   printf("GRID_DIM:%d BLOCK_DIM:%d\n",GRID_DIM,BLOCK_DIM);

   Timer("gemm_gty_bmma", [&]{
   bmma_kernel<TGEMMIN,TGEMMOUT,M_TILE,N_TILE,K_TILE>
      <<<GRID_DIM,BLOCK_DIM>>>(a.data,b.data,c.data,
         M_PAD,N_PAD,K_PAD);});

   copy<TOUT,TGEMMOUT,CUARR2ARR>(M,N,c_out,M_PAD,N_PAD,c);
}

template <typename TIN, typename TOUT>
void GEMM_cutlass(int M,int N,int K,TIN *a_in,TIN *b_in,TOUT *c_out){
   assert(M!=0 && N!=0 && K!=0);

   cuda_data<TIN> a(M*K),b(K*N);
   cuda_data<TOUT> c(M*N);

   //init a b
   copy<TIN,TIN,ARR2CUARR>(M,K,a_in,M,K,a);
   copy<TIN,TIN,ARR2CUARR>(K,N,b_in,K,N,b);

   Timer("gemm_cutlass", [&]{
      CutlassSgemmNN(M,N,K,1.0,a.data,K,b.data,N,0.0,c.data,N);});

   copy<TOUT,TOUT,CUARR2ARR>(M,N,c_out,M,N,c);
}


template<typename T>
bool valid(int M,int N,int K,T *ref,T *c,double threshold=1e-3){
   for(int i=0;i<M;i++)
   for(int j=0;j<N;j++){
      T abs = ref[i*N+j]>c[i*N+j] ? ref[i*N+j]-c[i*N+j] : c[i*N+j]-ref[i*N+j];
      if(abs/ref[i*N+j]>threshold){
         std::printf("at %d %d ",i,j);
         std::printf("expect:%f got:%f relative error:%f%%(>%f%%)\n",
            (float)ref[i*N+j],(float)c[i*N+j],(float)abs/ref[i*N+j]*100,threshold*100);
         return 0;
      }
   }
   return 1;
}

template<typename T>
void initVal(int size,T* arr){
   for(int i=0;i<size;i++){
      arr[i] = 1.0 * (rand()%1024) / (rand()%1024+1);
   }
}

template<typename TIN,typename TOUT>
void benchmark(int M,int N,int K,bool ifcheck){
   printf("----------------\n");
   printf("M:%d N:%d K:%d\n",M,N,K);

   srand(time(NULL));

   TIN *a,*b;
   TOUT *c_wmma,*c_bmma,*c_cutlass;

   a = (TIN*) malloc(M*K*sizeof(TIN));
   b = (TIN*) malloc(K*N*sizeof(TIN));
   c_wmma = (TOUT*)malloc(M*N*sizeof(TOUT));
   c_bmma = (TOUT*)malloc(M*N*sizeof(TOUT));
   c_cutlass = (TOUT*)malloc(M*N*sizeof(TOUT));

   initVal<TIN>(M*K,a);
   initVal<TIN>(K*N,b);

   GEMM_bmma<TIN,TOUT,half,float,16,16,16>(M,N,K,a,b,c_bmma);
   GEMM_wmma<TIN,TOUT,half,float,16,16,16>(M,N,K,a,b,c_wmma);
   GEMM_cutlass<TIN,TOUT>(M,N,K,a,b,c_cutlass);

   if(ifcheck && valid<TOUT>(M,N,K,c_wmma,c_bmma,1e-2) &&
      valid<TOUT>(M,N,K,c_bmma,c_cutlass,1e-2)){
      std::printf("check pass\n");
   }else if(ifcheck){
      std::printf("check fail\n");
   }else{
      std::printf("skip check\n");
   }
   
}

int main(){
   int args1[] = {512,1024,2048,4096,8192};
   for(auto arg:args1){
      benchmark<float,float>(arg,arg,arg,1);
   }
   
   int args2[] = {512,1024,2048,4096,8192};
   for(auto arg:args2){
      benchmark<float,float>(arg,arg,32,1);
   }

   int args3[] = {5120,10240,20480,40960,81920};
   for(auto arg:args3){
      benchmark<float,float>(16,16,arg,1);
   }
}