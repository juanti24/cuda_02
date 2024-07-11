
#define PALETTE_SIZE 16

#define WIDTH 1920
#define HEIGHT 1080
__constant__ unsigned int c_Pallete[PALETTE_SIZE];

__device__
unsigned int divergente(double cx, double cy, int max_iteraciones) {
    int iter = 0;
    double vx = cx;
    double vy = cy;
    double tx,ty;
    while (iter < max_iteraciones && (vx * vx + vy * vy) <= 4) {
        //Zn+1=Zn^2+C
        tx=vx * vx - vy * vy + cx; //vx"2 - vy"2+cx
        ty = 2 * vx * vy + cy;  // 2vx vy +cy

        vx = tx;
        vy = ty;

        iter++;
    }
    if ((vx * vx + vy * vy) > 4) {
        int color_idx = iter % PALETTE_SIZE;
        return c_Pallete[color_idx];
    }

    if (iter > 0 && iter < max_iteraciones) {
        //diverge
        //return 0xFFFF00FF;
        int color_idx = iter % PALETTE_SIZE;
        return c_Pallete[color_idx];
    } else {
        //convergente
        return 0x000000FF;

    }
}
__global__
void mandelbrotKernel(unsigned int* buffer, double x_min, double x_max , double y_min , double y_max,double dx, double dy,int max_iteraciones) {

//    double dx = (x_max - x_min) / WIDTH;
//    double dy = (y_max - y_min) / HEIGHT;

    int id = blockDim.x*blockIdx.x+threadIdx.x;
    int i= id/WIDTH ;
    int j=id%WIDTH;


    double x = x_min + j*dx;
    double y = y_max - i * dy;
    //C=X+Yi
    unsigned int color = divergente(x, y,max_iteraciones);
    buffer[id] = color;

}
extern "C" void set_kernel_palette(unsigned int* h_palette){
    cudaMemcpyToSymbol(c_Pallete,h_palette,PALETTE_SIZE*sizeof(unsigned int));

}

extern "C" void invoke_mandelbrot_kernel(
        int blk_in_grid,
        int thr_per_block,
        unsigned int* buffer,
        double x_min, double x_max , double y_min , double y_max,
        double dx, double dy,
        int max_iteraciones){

    mandelbrotKernel<<<blk_in_grid,thr_per_block>>>(buffer,
                                                    x_min,x_max,x_min,y_max,
                                                    dx,dy,
                                                    max_iteraciones);


}