#include <cuda_runtime.h>
#include <SFML/Window.hpp>
#include <fmt/core.h>
#include <SFML/Graphics.hpp>

#define WIDTH 1920
#define HEIGHT 1080

int max_iterations = 100;
const double x_min = -2;
const double x_max = 1;

const double y_min = -1;
const double y_max = 1;
int num_cores = 0;

static unsigned int *pixel_buffer = nullptr;

#define PALETTE_SIZE 16

uint32_t _bswap32(uint32_t a) {
    return
            ((a & 0x000000FF) << 24) |
            ((a & 0x0000FF00) << 8) |
            ((a & 0x00FF0000) >> 8) |
            ((a & 0xFF000000) >> 24);
}

//creando vector. (16)
std::vector<unsigned int> color_ramp = {
        _bswap32(0xEF1019FF),
        _bswap32(0xE01123FF),
        _bswap32(0xD1112DFF),
        _bswap32(0xC11237FF),
        _bswap32(0xB21341FF),
        _bswap32(0xA3134BFF),
        _bswap32(0x931455FF),
        _bswap32(0x84145EFF),
        _bswap32(0x751568FF),
        _bswap32(0x651672FF),
        _bswap32(0x56167CFF),
        _bswap32(0x471786FF),
        _bswap32(0x371790FF),
        _bswap32(0x28189AFF),
        _bswap32(0x1919A4FF)
};
//class para que el compilador no dependa de algo externo
enum class runtime_type_enum {
    rtCUDA
};

static runtime_type_enum runtime_type = runtime_type_enum::rtCUDA;


static unsigned int* host_pixel_buffer = nullptr;
static unsigned int* device_pixel_buffer = nullptr;

#define CHECK(expr) {                       \
        auto err = (expr);                  \
        if (err != cudaSuccess) {           \
            printf("%d: %s in % s at line % d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);             \
        }                                   \
    }



extern "C" void invoke_mandelbrot_kernel(
        int blk_in_grid,
        int thr_per_block,
        unsigned int* buffer,
        double x_min, double x_max , double y_min , double y_max,
        double dx, double dy,
        int max_iteraciones);
extern "C" void set_kernel_palette(unsigned int* h_palette);

void mandelbrotCuda(){
    int thr_per_block=1024;
    //redondear hacia arriba
    int blk_in_grid= std::ceil( float(WIDTH*HEIGHT)/thr_per_block);
    double dx = (x_max - x_min) / WIDTH;
    double dy = (y_max - y_min) / HEIGHT;

    invoke_mandelbrot_kernel(
            blk_in_grid,thr_per_block,
            device_pixel_buffer,
            x_min,x_max,y_min,y_max,
            dx,dy,
            max_iterations
    );
    CHECK(cudaGetLastError());
    int size = WIDTH*HEIGHT*sizeof (unsigned  int);
    cudaMemcpy(host_pixel_buffer,device_pixel_buffer,size,cudaMemcpyDeviceToHost);

}
int main() {

    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    fmt::println("Device : {}", properties.name);
    fmt::println("Usignd  {} multiprocesors", properties.multiProcessorCount);
    fmt::println("Using  {} max threads per processor", properties.maxThreadsPerMultiProcessor);

    int buffer_size = WIDTH * HEIGHT * sizeof(unsigned int);

    CHECK(cudaMalloc(&device_pixel_buffer, buffer_size));
    host_pixel_buffer = (unsigned int *) malloc(buffer_size);

    set_kernel_palette(color_ramp.data());

    sf::Text text;
    sf::Font font;

    pixel_buffer = new unsigned int[WIDTH * HEIGHT];

    {
        font.loadFromFile("arial.ttf");
        text.setFont(font);
        text.setString("Ejemplo CUDA");
        text.setCharacterSize(24);
        text.setFillColor(sf::Color::Green);
        text.setStyle(sf::Text::Bold);
        text.setPosition(10, 10);
    }
    sf::Text textOptions;
    {
        font.loadFromFile("arial.ttf");
        textOptions.setFont(font);
        textOptions.setCharacterSize(24);
        textOptions.setFillColor(sf::Color::White);
        textOptions.setStyle(sf::Text::Bold);
        textOptions.setString("Option: [1] CUDA ");
        text.setPosition(10, 50);
    }


    mandelbrotCuda();
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    texture.update((const sf::Uint8 *) host_pixel_buffer);

    sf::Sprite sprite;
    sprite.setTexture(texture);

    //auto = var
    sf::RenderWindow window{{WIDTH, HEIGHT}, "Madelbrot set"};
    window.setFramerateLimit(144);

    //time -> frames and fps
    sf::Clock clock;
    int frames = 0;
    int fps = 0;

    //siempre va estar abierto
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event))
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::Resized) {
                textOptions.setPosition(10, window.getView().getSize().y - 40);
            }

        mandelbrotCuda();

//        else if (event.type == sf::Event::KeyReleased) {
//                switch (event.key.scancode) {
//                    case sf::Keyboard::Scan::Num1:
//                        //cpu
//#pragma omp master
//                    {
//                        num_cores = omp_get_num_threads();
//                    }
//                        runtime_type = runtime_type_enum::rtCPU;
//                        break;
//                    case sf::Keyboard::Scan::Num2:
//                        //openMp
//#pragma omp parallel
//                    {
//                        num_cores = omp_get_num_threads();
//                    }
//                        runtime_type = runtime_type_enum::rtOpenMp;
//                        break;
//                }
//            }

//        fmt::println("generando imagen");
//        if (runtime_type == runtime_type_enum::rtCPU) {
//            mandelbrotCpu();
//        } else {
//            mandelbrotOpenMp();
//        }
        //textura actualizada after condicion line 177
        texture.update((const sf::Uint8 *) host_pixel_buffer);

        textOptions.setPosition(40, 40);



        // formatear el texto
        fmt::format("Ejemplo CUDA, FPS: {}", fps);
//        text.setString(msg);
        std::string msg = fmt::format("MANDELBROT SET::Mode={}, FPS:{}, Cores: {}",
                                      runtime_type == runtime_type_enum::rtCUDA ? "CUDA" : "OpenMp", fps, num_cores);
        text.setString(msg);

        if (clock.getElapsedTime().asSeconds() >= 1.0) {
            fps = frames;
            frames = 0;
            clock.restart();
        }

        frames++;

        window.clear();
        {
            window.draw(sprite);
            window.draw(text);
            window.draw(textOptions);
        }
        window.display();
    }

    delete[] pixel_buffer;
    return 0;
}