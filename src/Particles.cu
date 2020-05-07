#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#define TPB 64
#define TOTAL 10000
#define K 4
#define N_BATCHES 4
#define MAX_GPU_PARTICLES 1024000
#define STREAM_SIZE 6000
#define N_STREAMS 4


/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is, bool pinMemory)
{

    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];

    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }

    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;

    // cast it to required precision
    part->qom = (FPpart) param->qom[is];

    long npmax = part->npmax;

    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];


    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    if(!pinMemory) {
        part->x = new FPpart[npmax];
        part->y = new FPpart[npmax];
        part->z = new FPpart[npmax];
        // allocate velocity
        part->u = new FPpart[npmax];
        part->v = new FPpart[npmax];
        part->w = new FPpart[npmax];

        // allocate charge = q * statistical weight
        part->q = new FPinterp[npmax];
    }
    else {
        cudaHostAlloc(&part->x, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->y, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->z, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->u, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->v, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->w, sizeof(FPpart) * npmax, cudaHostAllocDefault);
        cudaHostAlloc(&part->q, sizeof(FPinterp) * npmax, cudaHostAllocDefault);
    }
}
/** deallocate */
void particle_deallocate(struct particles* part, bool pinMemory) {
    if (!pinMemory) {
        // deallocate particle variables
        delete[] part->x;
        delete[] part->y;
        delete[] part->z;
        delete[] part->u;
        delete[] part->v;
        delete[] part->w;
        delete[] part->q;
    }
    else{
        cudaFreeHost(part->x);
        cudaFreeHost(part->y);
        cudaFreeHost(part->z);
        cudaFreeHost(part->u);
        cudaFreeHost(part->v);
        cudaFreeHost(part->w);
        cudaFreeHost(part->q);
    }
}

/** single cpu mover */
int cpu_mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);

                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }

                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;


            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;


            //////////
            //////////
            ////////// BC

            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }

            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }


            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }

            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }

            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }



        }  // end of subcycling
    } // end of one particle

    return(0); // exit succcesfully
} // end of the mover


/*
1D block. Number of blocks = number of particles / TPB
 */
__global__ void particle_kernel( FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w,
                            FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, int nxn, int nyn, int nzn,
                            double xStart, double yStart, double zStart, FPfield invdx, FPfield invdy, FPfield invdz,
                            double Lx, double Ly, double Lz, FPfield invVOL,
                            FPfield* Ex_flat, FPfield* Ey_flat, FPfield* Ez_flat,
                            FPfield* Bxn_flat, FPfield* Byn_flat, FPfield* Bzn_flat,
                            bool PERIODICX, bool PERIODICY, bool PERIODICZ,
                            FPpart dt_sub_cycling, FPpart dto2, FPpart qomdt2,
                            const int NiterMover, const int npmax, const int offset)
{
    //calculate global index and check boundary
    const int idx = blockIdx.x*blockDim.x + threadIdx.x + offset;
    if(idx - offset > npmax) return;

    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // replace all indicies i to idx. all part->* replace to *
    xptilde = x[idx];
    yptilde = y[idx];
    zptilde = z[idx];
    // calculate the average velocity iteratively
    for(int innter=0; innter < NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((x[idx] - xStart)*invdx);
        iy = 2 +  int((y[idx] - yStart)*invdy);
        iz = 2 +  int((z[idx] - zStart)*invdz);

        // calculate weights
        xi[0]   = x[idx] - XN_flat[get_idx(ix-1, iy, iz, nyn, nzn)];//grd->XN[ix - 1][iy][iz];
        eta[0]  = y[idx] - YN_flat[get_idx(ix, iy-1, iz, nyn, nzn)];//grd->YN[ix][iy - 1][iz];
        zeta[0] = z[idx] - ZN_flat[get_idx(ix, iy, iz-1, nyn, nzn)];//grd->ZN[ix][iy][iz - 1];
        xi[1]   = XN_flat[get_idx(ix, iy, iz, nyn, nzn)] - x[idx];//grd->XN[ix][iy][iz] - x[i];
        eta[1]  = YN_flat[get_idx(ix, iy, iz, nyn, nzn)] - y[idx];//;grd->YN[ix][iy][iz] - y[i];
        zeta[1] = ZN_flat[get_idx(ix, iy, iz, nyn, nzn)] - z[idx];//grd->ZN[ix][iy][iz] - z[i];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * invVOL;

        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    Exl += weight[ii][jj][kk]*Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn)];//field->Ex[ix- ii][iy -jj][iz- kk ];
                    Eyl += weight[ii][jj][kk]*Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn)];//field->Ey[ix- ii][iy -jj][iz- kk ];
                    Ezl += weight[ii][jj][kk]*Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn)];//field->Ez[ix- ii][iy -jj][iz -kk ];
                    Bxl += weight[ii][jj][kk]*Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn)];//field->Bxn[ix- ii][iy -jj][iz -kk ];
                    Byl += weight[ii][jj][kk]*Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn)];//field->Byn[ix- ii][iy -jj][iz -kk ];
                    Bzl += weight[ii][jj][kk]*Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, nyn, nzn)];//field->Bzn[ix- ii][iy -jj][iz -kk ];
                }

        // end interpolation
        omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= u[idx] + qomdt2*Exl;
        vt= v[idx] + qomdt2*Eyl;
        wt= w[idx] + qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
        wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
        // update position
        x[idx] = xptilde + uptilde*dto2;
        y[idx] = yptilde + vptilde*dto2;
        z[idx] = zptilde + wptilde*dto2;


    } // end of iteration
    // update the final position and velocity
    u[idx]= 2.0*uptilde - u[idx];
    v[idx]= 2.0*vptilde - v[idx];
    w[idx]= 2.0*wptilde - w[idx];
    x[idx] = xptilde + uptilde*dt_sub_cycling;
    y[idx] = yptilde + vptilde*dt_sub_cycling;
    z[idx] = zptilde + wptilde*dt_sub_cycling;


    //////////
    //////////
    ////////// BC

    // X-DIRECTION: BC particles
    if (x[idx] > Lx){
        if (PERIODICX==true){ // PERIODIC
            x[idx] = x[idx] - Lx;
        } else { // REFLECTING BC
            u[idx] = -u[idx];
            x[idx] = 2*Lx - x[idx];
        }
    }

    if (x[idx] < 0){
        if (PERIODICX==true){ // PERIODIC
            x[idx] = x[idx] + Lx;
        } else { // REFLECTING BC
            u[idx] = -u[idx];
            x[idx] = -x[idx];
        }
    }


    // Y-DIRECTION: BC particles
    if (y[idx] > Ly){
        if (PERIODICY==true){ // PERIODIC
            y[idx] = y[idx] - Ly;
        } else { // REFLECTING BC
            v[idx] = -v[idx];
            y[idx] = 2*Ly - y[idx];
        }
    }

    if (y[idx] < 0){
        if (PERIODICY==true){ // PERIODIC
            y[idx] = y[idx] + Ly;
        } else { // REFLECTING BC
            v[idx] = -v[idx];
            y[idx] = -y[idx];
        }
    }

    // Z-DIRECTION: BC particles
    if (z[idx] > Lz){
        if (PERIODICZ==true){ // PERIODIC
            z[idx] = z[idx] - Lz;
        } else { // REFLECTING BC
            w[idx] = -w[idx];
            z[idx] = 2*Lz - z[idx];
        }
    }

    if (z[idx] < 0){
        if (PERIODICZ==true){ // PERIODIC
            z[idx] = z[idx] + Lz;
        } else { // REFLECTING BC
            w[idx] = -w[idx];
            z[idx] = -z[idx];
        }
    }



}


/** particle mover kernel of GPU*/
int gpu_mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param, bool useStream)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;


    //grid
    FPfield *d_XN_flat, *d_YN_flat, *d_ZN_flat;
    cudaMalloc(&d_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    //field
    FPfield *d_Ex_flat, *d_Ey_flat, *d_Ez_flat, *d_Bxn_flat, *d_Byn_flat, *d_Bzn_flat;
    //E-nodes
    cudaMalloc(&d_Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_Ex_flat, field->Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_Ey_flat, field->Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_Ez_flat, field->Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    //B-nodes
    cudaMalloc(&d_Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_Bxn_flat, field->Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_Byn_flat, field->Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_Bzn_flat, field->Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    std::cout<<"Free:"<<free_memory<<" , Total:"<<total_memory<<std::endl;

    assert(part->npmax % N_BATCHES == 0);
    const int n_batches = N_BATCHES;//part->npmax / MAX_GPU_PARTICLES;
    const int batch_size = part->npmax / N_BATCHES;
    const int batch_bytes = batch_size * sizeof(FPpart);

    for (int batch_idx = 0; batch_idx < n_batches; batch_idx++)
    {
        const int batch_offset = batch_idx * batch_size;

        assert(part->npmax % N_STREAMS == 0);
        const int stream_size = batch_size / N_STREAMS;
        const int stream_bytes = stream_size * sizeof(FPpart);

        FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
        cudaMalloc(&d_x, batch_bytes);
        cudaMalloc(&d_y, batch_bytes);
        cudaMalloc(&d_z, batch_bytes);
        cudaMalloc(&d_u, batch_bytes);
        cudaMalloc(&d_v, batch_bytes);
        cudaMalloc(&d_w, batch_bytes);

        // initialize streams
        cudaStream_t stream[N_STREAMS];
        for (int i = 0; i < N_STREAMS; i++)
            cudaStreamCreate(&stream[i]);

        for (int stream_idx = 0; stream_idx < N_STREAMS; stream_idx++)
        {

            const int offset = stream_idx * stream_size;
            const int global_offset = offset + batch_offset;

            // Use async copy for particles
            cudaMemcpyAsync(&d_x[offset], &part->x[global_offset], stream_bytes, cudaMemcpyHostToDevice, stream[stream_idx]);
            cudaMemcpyAsync(&d_y[offset], &part->y[global_offset], stream_bytes, cudaMemcpyHostToDevice, stream[stream_idx]);
            cudaMemcpyAsync(&d_z[offset], &part->z[global_offset], stream_bytes, cudaMemcpyHostToDevice, stream[stream_idx]);
            cudaMemcpyAsync(&d_u[offset], &part->u[global_offset], stream_bytes, cudaMemcpyHostToDevice, stream[stream_idx]);
            cudaMemcpyAsync(&d_v[offset], &part->v[global_offset], stream_bytes, cudaMemcpyHostToDevice, stream[stream_idx]);
            cudaMemcpyAsync(&d_w[offset], &part->w[global_offset], stream_bytes, cudaMemcpyHostToDevice, stream[stream_idx]);

            std::cout << "Before loop" << ". Offset:" << offset << ". # of elems:" << stream_size
                      << " Stream index:" << stream_idx << std::endl;
            // start subcycling
            for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
                //call kernel
                particle_kernel <<< (stream_size + TPB - 1) / TPB, TPB, 0, stream[stream_idx] >>>
                        (
                        d_x, d_y, d_z, d_u, d_v, d_w,
                        d_XN_flat, d_YN_flat, d_ZN_flat, grd->nxn, grd->nyn, grd->nzn,
                        grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz,
                        grd->Lx, grd->Ly, grd->Lz, grd->invVOL,
                        d_Ex_flat, d_Ey_flat, d_Ez_flat, d_Bxn_flat, d_Byn_flat, d_Bzn_flat,
                        param->PERIODICX, param->PERIODICY, param->PERIODICZ,
                        dt_sub_cycling, dto2, qomdt2,
                        part->NiterMover, stream_size, offset
                        );


            } // end of one particle

            cudaMemcpyAsync(&part->x[global_offset], &d_x[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[stream_idx]);
            cudaMemcpyAsync(&part->y[global_offset], &d_y[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[stream_idx]);
            cudaMemcpyAsync(&part->z[global_offset], &d_z[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[stream_idx]);
            cudaMemcpyAsync(&part->u[global_offset], &d_u[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[stream_idx]);
            cudaMemcpyAsync(&part->v[global_offset], &d_v[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[stream_idx]);
            cudaMemcpyAsync(&part->w[global_offset], &d_w[offset], stream_bytes, cudaMemcpyDeviceToHost, stream[stream_idx]);

            cudaStreamSynchronize(stream[stream_idx]);

        }

        for(int i = 0; i < N_STREAMS; i++){
                cudaStreamDestroy(stream[i]);
        }

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_u);
        cudaFree(d_v);
        cudaFree(d_w);
    }
    /*
    while(true)
    {
        const long int to = split_index + MAX_GPU_PARTICILES - 1 < part->npmax - 1 ? split_index + MAX_GPU_PARTICILES - 1 : part->npmax - 1;


        if(!useStream) {
            const int n_particles = to - split_index + 1;
            size_t batch_size = (to - split_index + 1) * sizeof(FPpart);
            FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
            cudaMalloc(&d_x, batch_size);
            cudaMalloc(&d_y, batch_size);
            cudaMalloc(&d_z, batch_size);
            cudaMalloc(&d_u, batch_size);
            cudaMalloc(&d_v, batch_size);
            cudaMalloc(&d_w, batch_size);

            //particles
            cudaMemcpy(d_x, part->x + split_index, batch_size, cudaMemcpyHostToDevice);

            cudaMemcpy(d_y, part->y + split_index, batch_size, cudaMemcpyHostToDevice);

            cudaMemcpy(d_z, part->z + split_index, batch_size, cudaMemcpyHostToDevice);

            cudaMemcpy(d_u, part->u + split_index, batch_size, cudaMemcpyHostToDevice);

            cudaMemcpy(d_v, part->v + split_index, batch_size, cudaMemcpyHostToDevice);

            cudaMemcpy(d_w, part->w + split_index, batch_size, cudaMemcpyHostToDevice);

            std::cout << "Before loop" << ". Batch idxs:" << split_index << ":" << to << ". # of elems:" << n_particles
                      << std::endl;
            // start subcycling
            for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
                particle_kernel << < (n_particles + TPB - 1) / TPB, TPB >> > (d_x, d_y, d_z, d_u, d_v, d_w,
                        d_XN_flat, d_YN_flat, d_ZN_flat, grd->nxn, grd->nyn, grd->nzn,
                        grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz,
                        grd->Lx, grd->Ly, grd->Lz, grd->invVOL,
                        d_Ex_flat, d_Ey_flat, d_Ez_flat, d_Bxn_flat, d_Byn_flat, d_Bzn_flat,
                        param->PERIODICX, param->PERIODICY, param->PERIODICZ,
                        dt_sub_cycling, dto2, qomdt2,
                        part->NiterMover, n_particles);
			cudaDeviceSynchronize();

            } // end of one particle

            cudaMemcpy(part->x + split_index, d_x, batch_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(part->y + split_index, d_y, batch_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(part->z + split_index, d_z, batch_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(part->u + split_index, d_u, batch_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(part->v + split_index, d_v, batch_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(part->w + split_index, d_w, batch_size, cudaMemcpyDeviceToHost);

    	split_index += MAX_GPU_PARTICILES;

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_u);
        cudaFree(d_v);
        cudaFree(d_w);

        if (to == part->npmax - 1)
            break;


	 }

        else{
            //createStreams(&streams);
	    // If batch_size <= STREAM_SIZE, n_streams = 1, and whole batch is done in one stream
            	split_index += MAX_GPU_PARTICILES;
        if (to == part->npmax - 1)
            break;


        }



    }
    */
    //E-nodes
    cudaMemcpy(field->Ex_flat, d_Ex_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ey_flat, d_Ey_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Ez_flat, d_Ez_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);
    //B-nodes
    cudaMemcpy(field->Bxn_flat, d_Bxn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Byn_flat, d_Byn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);

    cudaMemcpy(field->Bzn_flat, d_Bzn_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyDeviceToHost);

    //free memory


    cudaFree(d_XN_flat);
    cudaFree(d_YN_flat);
    cudaFree(d_ZN_flat);

    cudaFree(d_Ex_flat);
    cudaFree(d_Ey_flat);
    cudaFree(d_Ez_flat);
    cudaFree(d_Bxn_flat);
    cudaFree(d_Byn_flat);
    cudaFree(d_Bzn_flat);

    return(0); // exit succcesfully
} // end of the mover



__global__ void interP2G_kernel(FPpart* x, FPpart* y, FPpart* z, FPpart* u, FPpart* v, FPpart* w, FPinterp* q,
                                FPfield* XN_flat, FPfield* YN_flat, FPfield* ZN_flat, int nxn, int nyn, int nzn,
                                double xStart, double yStart, double zStart, FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
                                FPinterp* Jx_flat, FPinterp* Jy_flat, FPinterp *Jz_flat, FPinterp *rhon_flat,
                                FPinterp* pxx_flat, FPinterp* pxy_flat, FPinterp* pxz_flat,
                                FPinterp* pyy_flat, FPinterp* pyz_flat, FPinterp* pzz_flat, const int npmax)
{
    //calculate global index and check boundary
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= npmax) return;

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // 3-D index of the cell
    int ix, iy, iz;

    ix = 2 + int (floor((x[idx] - xStart) * invdx));
    iy = 2 + int (floor((y[idx] - yStart) * invdy));
    iz = 2 + int (floor((z[idx] - zStart) * invdz));

    // distances from node
    xi[0]   = x[idx] - XN_flat[get_idx(ix-1, iy, iz, nyn, nzn)];//grd->XN[ix - 1][iy][iz];
    eta[0]  = y[idx] - YN_flat[get_idx(ix, iy-1, iz, nyn, nzn)];//grd->YN[ix][iy - 1][iz];
    zeta[0] = z[idx] - ZN_flat[get_idx(ix, iy, iz-1, nyn, nzn)];//grd->ZN[ix][iy][iz - 1];
    xi[1]   = XN_flat[get_idx(ix, iy, iz, nyn, nzn)] - x[idx];//grd->XN[ix][iy][iz] - x[i];
    eta[1]  = YN_flat[get_idx(ix, iy, iz, nyn, nzn)] - y[idx];//;grd->YN[ix][iy][iz] - y[i];
    zeta[1] = ZN_flat[get_idx(ix, iy, iz, nyn, nzn)] - z[idx];//grd->ZN[ix][iy][iz] - z[i];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                // calculate the weights for different nodes
                weight[ii][jj][kk] = q[idx] * xi[ii] * eta[jj] * zeta[kk] * invVOL;


    //////////////////////////
    // add charge density
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&rhon_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], weight[ii][jj][kk] * invVOL);


    ////////////////////////////
    // add current density - Jx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * weight[ii][jj][kk];

    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&Jx_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);


    ////////////////////////////
    // add current density - Jy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&Jy_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);



    ////////////////////////////
    // add current density - Jz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = w[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&Jz_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);


    ////////////////////////////
    // add pressure pxx
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * u[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pxx_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);


    ////////////////////////////
    // add pressure pxy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * v[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pxy_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);



    /////////////////////////////
    // add pressure pxz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = u[idx] * w[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pxz_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);


    /////////////////////////////
    // add pressure pyy
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * v[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pyy_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);


    /////////////////////////////
    // add pressure pyz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = v[idx] * w[idx] * weight[ii][jj][kk];
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                atomicAdd(&pyz_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);


    /////////////////////////////
    // add pressure pzz
    for (int ii = 0; ii < 2; ii++)
        for (int jj = 0; jj < 2; jj++)
            for (int kk = 0; kk < 2; kk++)
                temp[ii][jj][kk] = w[idx] * w[idx] * weight[ii][jj][kk];
    for (int ii=0; ii < 2; ii++)
        for (int jj=0; jj < 2; jj++)
            for(int kk=0; kk < 2; kk++)
                atomicAdd(&pzz_flat[get_idx(ix - ii, iy - jj, iz - kk, nyn, nzn)], temp[ii][jj][kk] * invVOL);

}



/** Interpolation kernel of GPU*/
void gpu_interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    //particles
    FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
    FPinterp* d_q;
    cudaMalloc(&d_x, part->npmax * sizeof(FPpart));
    cudaMemcpy(d_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&d_y, part->npmax * sizeof(FPpart));
    cudaMemcpy(d_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&d_z, part->npmax * sizeof(FPpart));
    cudaMemcpy(d_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&d_u, part->npmax * sizeof(FPpart));
    cudaMemcpy(d_u, part->u, part->npmax* sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&d_v, part->npmax * sizeof(FPpart));
    cudaMemcpy(d_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&d_w, part->npmax * sizeof(FPpart));
    cudaMemcpy(d_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    cudaMalloc(&d_q, part->npmax * sizeof(FPinterp));
    cudaMemcpy(d_q, part->q, part->npmax * sizeof(FPinterp), cudaMemcpyHostToDevice);

    //ids
    FPinterp *d_Jx_flat, *d_Jy_flat, *d_Jz_flat, *d_rhon_flat;
    FPinterp *d_pxx_flat, *d_pxy_flat, *d_pxz_flat, *d_pyy_flat, *d_pyz_flat, *d_pzz_flat;
    cudaMalloc(&d_Jx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_Jx_flat, ids->Jx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Jy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_Jy_flat, ids->Jy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_Jz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_Jz_flat, ids->Jz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_rhon_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_rhon_flat, ids->rhon_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pxx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_pxx_flat, ids->pxx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pxy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_pxy_flat, ids->pxy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pxz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_pxz_flat, ids->pxz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pyy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_pyy_flat, ids->pyy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pyz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_pyz_flat, ids->pyz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_pzz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp));
    cudaMemcpy(d_pzz_flat, ids->pzz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    //grid
    FPfield *d_XN_flat, *d_YN_flat, *d_ZN_flat;
    cudaMalloc(&d_XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMalloc(&d_ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMemcpy(d_ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);


    interP2G_kernel<<<(part->npmax + TPB - 1)/TPB, TPB>>>(  d_x, d_y, d_z, d_u, d_v, d_w, d_q,
            d_XN_flat, d_YN_flat, d_ZN_flat, grd->nxn, grd->nyn, grd->nzn,
            grd->xStart, grd->yStart, grd->zStart, grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
            d_Jx_flat, d_Jy_flat, d_Jz_flat, d_rhon_flat,
            d_pxx_flat, d_pxy_flat, d_pxz_flat, d_pyy_flat, d_pyz_flat, d_pzz_flat,
            part->nop);

    cudaDeviceSynchronize();

    //ids
    cudaMemcpy(ids->Jx_flat, d_Jx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jy_flat, d_Jy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->Jz_flat, d_Jz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->rhon_flat, d_rhon_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxx_flat, d_pxx_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxy_flat, d_pxy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pxz_flat, d_pxz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyy_flat, d_pyy_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pyz_flat, d_pyz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ids->pzz_flat, d_pzz_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPinterp), cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_q);
    cudaFree(d_XN_flat);
    cudaFree(d_YN_flat);
    cudaFree(d_ZN_flat);
    cudaFree(d_rhon_flat);
    cudaFree(d_pxx_flat);
    cudaFree(d_pxy_flat);
    cudaFree(d_pxz_flat);
    cudaFree(d_pyy_flat);
    cudaFree(d_pyz_flat);
    cudaFree(d_pzz_flat);

    return;
}


/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{

    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // index of the cell
    int ix, iy, iz;


    for (register long long i = 0; i < part->nop; i++) {

        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));

        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];

        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;



        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;



        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

    }

}
