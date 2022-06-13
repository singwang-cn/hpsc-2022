/*Requires a gcc version below 9.0.0*/
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define M 1024

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
__constant__ double dx = 2.0 / ((double)nx - 1.0);
__constant__ double dy = 2.0 / ((double)ny - 1.0);
__constant__ double dt = .01;
__constant__ double rho = 1;
__constant__ double nu = .02;

thrust::device_vector<double> u(ny*nx);
thrust::device_vector<double> v(ny*nx);
thrust::device_vector<double> p(ny*nx);
thrust::device_vector<double> b(ny*nx);

__global__ void computing_b(double *u, double *v, double *b) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i > nx) && (i < nx*(ny-1)) && (i%nx != 0) && ((i-nx+1)%nx != 0)) {
        b[i] = rho * (1/dt
                * ((u[i+1]-u[i-1]) / (2*dx) + (v[i+nx]-v[i-nx]) / (2*dy))
                - (((u[i+1]-u[i-1]) / (2*dx)) * ((u[i+1]-u[i-1]) / (2*dx)))
                -2 * ((u[i+nx]-u[i-nx]) / (2*dy) * (v[i+1] - v[i-1]) / (2*dx))
                - (((v[i+nx]-v[i-nx]) / (2*dy))*((v[i+nx]-v[i-nx]) / (2*dy))));
        //printf("%lf, %d\n", b[i], i);
    }
   
}

__global__ void computing_p(double *pn, double *p, double *b) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i > nx) && (i < nx*(ny-1)) && (i%nx != 0) && ((i-nx+1)%nx != 0)) {
        p[i] = (dy*dy * (pn[i+1]+pn[i-1])
                + dx*dx * (pn[i+nx]+pn[i-nx])
                - b[i] * dx*dx * dy*dy) / (2 * (dx*dx + dy*dy));
        //printf("%lf, %d\n", p[i], i);
    }
    __syncthreads();
    if ((i-nx+1)%nx == 0) p[i] = p[i-1];
    __syncthreads();
    if (i < nx) p[i] = p[i+nx];
    __syncthreads();
    if (i%nx == 0) p[i] = p[i+1];
    __syncthreads();
    if (i >= nx*(ny-1) && i < nx*ny) p[i] = 0;
}

__global__ void computing_uv(double *u, double *v, double *un, double *vn, double *p) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __syncthreads();
    if ((i > nx) && (i < nx*(ny-1)) && (i%nx != 0) && ((i-nx+1)%nx != 0)) {
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1]) - un[i] * dt / dy * (un[i] - un[i-nx])
                    - dt / (2*rho*dx) * (p[i+1] - p[i-1])
                    + nu * dt / (dx*dx) * (un[i+1] - 2 * un[i] + un[i-1])
                    + nu * dt / (dy*dy) * (un[i+nx] - 2 * un[i] + un[i-nx]);
        v[i] = vn[i] - vn[i] * dt / dx * (vn[i] - vn[i-1]) - vn[i] * dt / dy * (vn[i] - vn[i-nx])
                    - dt / (2*rho*dx) * (p[i+nx] - p[i-nx])
                    + nu * dt / (dx*dx) * (vn[i+1] - 2 * vn[i] + vn[i-1])
                    + nu * dt / (dy*dy) * (vn[i+nx] - 2 * vn[i] + vn[i-nx]);
        //printf("%lf, %lf, %d\n", u[i], v[i], i);
    }
    __syncthreads();
    if (i >= nx*(ny-1) && i < nx*ny) {
        u[i] = 1;
        v[i] = 0;
    }
    __syncthreads();
    if (i < nx || i%nx == 0 || (i-nx+1)%nx == 0) {
        u[i] = 0;
        v[i] = 0;
    }   
}

int main(void)
{
    for (int i = 0; i < nt; i++) {
        computing_b<<<(ny*nx+M-1)/M, M>>>(thrust::raw_pointer_cast(u.data()),
                                            thrust::raw_pointer_cast(v.data()),
                                            thrust::raw_pointer_cast(b.data()));
        cudaDeviceSynchronize();
        for (int i = 0; i < nit; i++) {
            thrust::device_vector<double> pn(ny*nx);
            thrust::copy(p.begin(), p.end(), pn.begin());
            computing_p<<<(ny*nx+M-1)/M, M>>>(thrust::raw_pointer_cast(pn.data()),
                                            thrust::raw_pointer_cast(p.data()),
                                            thrust::raw_pointer_cast(b.data()));
            cudaDeviceSynchronize();
        }
        thrust::device_vector<double> un(ny*nx);
        thrust::device_vector<double> vn(ny*nx);
        thrust::copy(u.begin(), u.end(), un.begin());
        thrust::copy(v.begin(), v.end(), vn.begin());
        computing_uv<<<(ny*nx+M-1)/M, M>>>(thrust::raw_pointer_cast(u.data()),
                                            thrust::raw_pointer_cast(v.data()),
                                            thrust::raw_pointer_cast(un.data()),
                                            thrust::raw_pointer_cast(vn.data()),
                                            thrust::raw_pointer_cast(p.data()));
        cudaDeviceSynchronize();
        //This process behaves differently on GPU and CPU resulting in some undesired errors,
        //will continue to look for a solution after that.
        /*for(int i = 0; i < u.size(); i++) {
            if (i >= nx*(ny-1) && i < nx*ny) {
                u[i] = 1;
                v[i] = 0;
            }
            if (i < nx || i%nx == 0 || (i-nx+1)%nx == 0) {
                u[i] = 0;
                v[i] = 0;
            }   
        }*/
    }
    
    
    for(int i = 0; i < u.size(); i++)
        std::cout << "u[" << i << "] = " << u[i]<< " v[" << i << "] = " << v[i] << std::endl;
    
    //just a test -_-!
    /*
    thrust::device_vector<double> B(ny*nx, 1);
    thrust::device_vector<double> U(ny*nx, 1);
    thrust::device_vector<double> V(ny*nx, 1);
    thrust::sequence(U.begin(), U.end());
    thrust::sequence(V.begin(), V.end());
    computing_b<<<(ny*nx+M-1)/M, M>>>(thrust::raw_pointer_cast(U.data()),
                                            thrust::raw_pointer_cast(V.data()),
                                            thrust::raw_pointer_cast(B.data()));
    for(int i = 0; i < B.size(); i++)
        std::cout << "B[" << i << "] = " << B[i] << std::endl;
    thrust::device_vector<double> P(ny*nx, 1);
    thrust::device_vector<double> PN(ny*nx, 1);
    computing_p<<<(ny*nx+M-1)/M, M>>>(thrust::raw_pointer_cast(PN.data()),
                                            thrust::raw_pointer_cast(P.data()),
                                            thrust::raw_pointer_cast(B.data()));
    for(int i = 0; i < P.size(); i++)
        std::cout << "P[" << i << "] = " << P[i] << std::endl;
    thrust::device_vector<double> UN(ny*nx);
    thrust::device_vector<double> VN(ny*nx);
    thrust::copy(U.begin(), U.end(), UN.begin());
    thrust::copy(V.begin(), V.end(), VN.begin());
    computing_uv<<<(ny*nx+M-1)/M, M>>>(thrust::raw_pointer_cast(U.data()),
                                            thrust::raw_pointer_cast(V.data()),
                                            thrust::raw_pointer_cast(UN.data()),
                                            thrust::raw_pointer_cast(VN.data()),
                                            thrust::raw_pointer_cast(P.data()));
    for(int i = 0; i < U.size(); i++) {
        if (i >= nx*(ny-1) && i < nx*ny) {
            U[i] = 1;
            V[i] = 0;
        }
        if (i < nx || i%nx == 0 || (i-nx+1)%nx == 0) {
            U[i] = 0;
            V[i] = 0;
        }   
    }                                        
    for(int i = 0; i < U.size(); i++)
        std::cout << "U[" << i << "] = " << U[i] << " V[" << i << "] = " << V[i] << std::endl;
    */
    
    return 0;
}