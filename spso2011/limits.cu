#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cuda.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#define CHK_CUDA(e) {if (e != cudaSuccess) {fprintf(stderr,"Error: %s\n", cudaGetErrorString(e)); exit(-1);}}

/* from wikipedia page, for machine epsilon calculation */
/* assumes mantissa in final bits */
__device__ double machine_eps_dbl() {
    typedef union {
        long long i64;
        double d64;
    } dbl_64;

    dbl_64 s;

    s.d64 = 1.;
    s.i64++;
    return (s.d64 - 1.);
}

__device__ float machine_eps_flt() {
    typedef union {
        int i32;
        float f32;
    } flt_32;

    flt_32 s;

    s.f32 = 1.;
    s.i32++;
    return (s.f32 - 1.);
}

#define EPS 0
#define MIN 1
#define MAX 2

__global__ void calc_consts(float *fvals, double *dvals) {

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i==0) {
        fvals[EPS] = machine_eps_flt();
        dvals[EPS]= machine_eps_dbl();

        float xf, oldxf;
        double xd, oldxd; 

        xf = 2.; oldxf = 1.;
        xd = 2.; oldxd = 1.;

        /* double until overflow */
        /* Note that real fmax is somewhere between xf and oldxf */
        while (!isinf(xf))  {
            oldxf *= 2.;
            xf *= 2.;
        }

        while (!isinf(xd))  {
            oldxd *= 2.;
            xd *= 2.;
        }

        dvals[MAX] = oldxd;
        fvals[MAX] = oldxf;

        /* half until overflow */
        /* Note that real fmin is somewhere between xf and oldxf */
        xf = 1.; oldxf = 2.;
        xd = 1.; oldxd = 2.;

        while (xf != 0.)  {
            oldxf /= 2.;
            xf /= 2.;
        }

        while (xd != 0.)  {
            oldxd /= 2.;
            xd /= 2.;
        }

        dvals[MIN] = oldxd;
        fvals[MIN] = oldxf;

    }
    return;
}

int main(int argc, char **argv) {
    float  fvals[3];
    double dvals[3];
    float  *fvals_d;
    double *dvals_d;

    CHK_CUDA( cudaMalloc(&fvals_d, 3*sizeof(float)) );
    CHK_CUDA( cudaMalloc(&dvals_d, 3*sizeof(double)) );

    calc_consts<<<1,32>>>(fvals_d, dvals_d);

    CHK_CUDA( cudaMemcpy(fvals, fvals_d, 3*sizeof(float), cudaMemcpyDeviceToHost) );
    CHK_CUDA( cudaMemcpy(dvals, dvals_d, 3*sizeof(double), cudaMemcpyDeviceToHost) );

    CHK_CUDA( cudaFree(fvals_d) );
    CHK_CUDA( cudaFree(dvals_d) );

    printf("Single machine epsilon:\n");
    printf("CUDA = %g, CPU = %g\n", fvals[EPS], FLT_EPSILON);
    printf("Single min value (CUDA - approx):\n");
    printf("CUDA = %g, CPU = %g\n", fvals[MIN], FLT_MIN);
    printf("Single max value (CUDA - approx):\n");
    printf("CUDA = %g, CPU = %g\n", fvals[MAX], FLT_MAX);

    printf("\nDouble machine epsilon:\n");
    printf("CUDA = %lg, CPU = %lg\n", dvals[EPS], DBL_EPSILON);
    printf("Double min value (CUDA - approx):\n");
    printf("CUDA = %lg, CPU = %lg\n", dvals[MIN], DBL_MIN);
    printf("Double max value (CUDA - approx):\n");
    printf("CUDA = %lg, CPU = %lg\n", dvals[MAX], DBL_MAX);

    return 0;
}