// Bouncy ball reflections demo.
//
// It's not a very big program so the whole thing is written around
// this Ball class that does everything -- unsophisticated bouncy
// physics and OpenGL draw calls all come from there. A properly
// structured program, which this is not, would have some abstraction
// layer for OpenGL but we don't do that.
//
// Basically, what we do is keep an OpenGL cubemap texture handle and
// 6 framebuffer handles in each Ball object. Each frame, we draw the
// scene to the screen as usual, but we also draw the scene onto each
// ball's cubemap texture (from each ball's perspective). When we draw
// a Ball, we calculate a reflection vector for each fragment and
// sample from the Ball's cubemap to create reflection effects.
//
// This code is not even close to threadsafe.

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <string>
#include <vector>
#include <list>
#include <utility>
using std::swap;

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "gl_core_3_3.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

namespace {

constexpr float
    ball_radius = 0.4f,
    ball_core_radius_ratio = 0.707f,
    min_x = -12.0f,
    max_x = +12.0f,
    min_y = 0.0f,
    max_y = 1e+30f,
    min_z = -12.0f,
    max_z = +12.0f,
    ticks_per_second = 600.0f,
    gravity = 4.0f,
    fovy_radians = 1.0f,
    near_plane = 0.1f,
    far_plane = 40.0f;

constexpr int
    plus_x_index = 0,
    minus_x_index = 1,
    plus_y_index = 2,
    minus_y_index = 3,
    plus_z_index = 4,
    minus_z_index = 5;

int screen_x = 1280, screen_y = 960;
SDL_Window* window = nullptr;
std::string argv0;

static void panic(const char* message, const char* reason) {
        fprintf(stderr, "%s: %s %s\n", argv0.c_str(), message, reason);
    fflush(stderr);
    fflush(stdout);
    SDL_ShowSimpleMessageBox(
        SDL_MESSAGEBOX_ERROR, message, reason, nullptr
    );
    exit(1);
    abort();
}

#define PANIC_IF_GL_ERROR do { \
    if (auto PANIC_error = glGetError()) { \
        char PANIC_msg[160]; \
        snprintf(PANIC_msg, sizeof PANIC_msg, "line %i: code %i", __LINE__, (int)PANIC_error); \
        panic("OpenGL error", PANIC_msg); \
    } \
} while (0)

static GLuint make_program(const char* vs_code, const char* fs_code) {
    static GLchar log[1024];
    PANIC_IF_GL_ERROR;
    GLuint program_id = glCreateProgram();
    GLuint vs_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs_id = glCreateShader(GL_FRAGMENT_SHADER);
    
    const GLchar* string_array[1];
    string_array[0] = (GLchar*)vs_code;
    glShaderSource(vs_id, 1, string_array, nullptr);
    string_array[0] = (GLchar*)fs_code;
    glShaderSource(fs_id, 1, string_array, nullptr);
    
    glCompileShader(vs_id);
    glCompileShader(fs_id);
    
    printf("%i %i\n", (int)vs_id, (int)fs_id);
    
    PANIC_IF_GL_ERROR;
    
    GLint okay = 0;
    GLsizei length = 0;
    const GLuint shader_id_array[2] = { vs_id, fs_id };
    for (auto id : shader_id_array) {
        glGetShaderiv(id, GL_COMPILE_STATUS, &okay);
        if (okay) {
            glAttachShader(program_id, id);
        } else {
            glGetShaderInfoLog(id, sizeof log, &length, log);
            fprintf(stderr, "%s\n", id == vs_id ? vs_code : fs_code);
            panic("Shader compilation error", log);
        }
    }
    
    glLinkProgram(program_id);
    glGetProgramiv(program_id, GL_LINK_STATUS, &okay);
    if (!okay) {
        glGetProgramInfoLog(program_id, sizeof log, &length, log);
        panic("Shader link error", log);
    }
    
    PANIC_IF_GL_ERROR;
    return program_id;
}

// To do reflections on each ball, we will associate a framebuffer
// object and six 2d texture faces (+/- xyz) to each ball in the
// scene. We will render a "skybox" from the perspective of each ball
// and sample reflections from this skybox.
struct BallRender {
    GLuint framebuffers[6];
    GLuint cubemap;
};

class Ball;
using BallList = std::list<Ball>;

class Ball {
    glm::vec3 position;
    glm::vec3 velocity;
    float r, g, b, radius;
    bool bounced = false;
    BallRender render;
    
    static std::vector<BallRender> recycled_ball_render;
  public:
    Ball(glm::vec3 pos_arg, glm::vec3 vel_arg,
         float r, float g, float b, float radius)
    {
        position = pos_arg;
        velocity = vel_arg;
        r = r;
        g = g;
        b = b;
        bounced = false;
        radius = radius;
        
        if (!recycled_ball_render.empty()) {
            render = recycled_ball_render.back();
            recycled_ball_render.pop_back();
        } else {
            PANIC_IF_GL_ERROR;
            glGenFramebuffers(6, render.framebuffers);
            glGenTextures(1, &render.cubemap);
            PANIC_IF_GL_ERROR;
        }
    }
    
    ~Ball() {
        recycled_ball_render.push_back(render);
    }
    
    Ball(Ball&&) = delete;
    
    // Returns true (and modifies velocity) if we bounce with the
    // other ball.  The other ball is also affected. We bounce if the
    // two balls overlap and the two balls are moving towards each
    // other (so don't bounce if they're already moving away; that
    // would put them back on a collision course).
    //
    // Sets the bounce flag of both balls to true if we bounced.
    bool bounce_ball(Ball* other_ball_ptr) {
        // This isn't right physics.
        Ball& other = *other_ball_ptr;
        glm::vec3 displacement = other.position - position;
        float squared_distance = dot(displacement, displacement);
        float squared_radii = (radius + other.radius)
                            * (radius + other.radius);
        
        bool collision_course = dot(displacement, velocity-other.velocity) > 0;
        
        if (collision_course && squared_distance < squared_radii) {
            swap(velocity, other.velocity);
            bounced = true;
            other.bounced = true;
            return true;
        } else {
            return false;
        }
    }
    
    // Returns true (and modifies velocity) if we are beyond the edge
    // of the bounding box (min/max x y z) and our speed is such that
    // we're moving further out.
    //
    // Sets the bounce flag to true if we bounced.
    bool bounce_bounds() {
        bool flag = false;
        
        if (position[0] > max_x && velocity[0] > 0) {
            velocity[0] *= -1.0f;
            flag = true;
        }
        if (position[1] > max_y && velocity[1] > 0) {
            velocity[1] *= -1.0f;
            flag = true;
        }
        if (position[2] > max_z && velocity[2] > 0) {
            velocity[2] *= -1.0f;
            flag = true;
        }
        if (position[0] < min_x && velocity[0] < 0) {
            velocity[0] *= -1.0f;
            flag = true;
        }
        if (position[1] < max_y && velocity[1] < 0) {
            velocity[1] *= -1.0f;
            flag = true;
        }
        if (position[2] < max_z && velocity[2] < 0) {
            velocity[2] *= -1.0f;
            flag = true;
        }
        bounced |= flag;
        return flag;
    }
    
    bool bounce_flag() const {
        return bounced;
    }
    
    void reset_bounce_flag() {
        bounced = false;
    }
    
    // Euler method tick: update position using velocity and velocity
    // using gravity acceleration.
    void tick(float dt) {
        velocity -= glm::vec3(0.0f, dt*gravity, 0.0f);
        position += velocity * dt;
    }
    
    // Draw a list of Balls onto the current framebuffer, skipping the
    // ball pointed-to by the skip pointer (if any). The provided view
    // and projection matrices are used in the ordinary way.
    static void draw_list(
        glm::mat4 view_matrix,
        glm::mat4 proj_matrix,
        BallList const& list,
        Ball* skip=nullptr
    ) {
        static bool buffers_initialized = false;
        static GLuint vertex_buffer_id;
        static int vertex_count;
        
        // Initialize a vertex buffer with vertices of sphere with
        // radius 1 suitable for use with GL_TRIANGLES draw mode.
        if (!buffers_initialized) {
            std::vector<glm::vec3> coord_vector;
            
            const float magic = 3;
            
            auto add_face = [&coord_vector, magic]
            (glm::vec3 a_vec, glm::vec3 b_vec, glm::vec3 face_vec) {
                for (float a = -magic; a < magic; ++a) {
                    for (float b = -magic; b < magic; ++b) {
                        glm::vec3 coord0 = a*a_vec + b*b_vec + face_vec;
                        coord0 = normalize(coord0);
                        glm::vec3 coord1 = (a+1)*a_vec + b*b_vec + face_vec;
                        coord1 = normalize(coord1);
                        glm::vec3 coord2 = a*a_vec + (b+1)*b_vec + face_vec;
                        coord2 = normalize(coord2);
                        glm::vec3 coord3 = (a+1)*a_vec + (b+1)*b_vec + face_vec;
                        coord3 = normalize(coord3);
                        
                        coord_vector.push_back(coord0);
                        coord_vector.push_back(coord1);
                        coord_vector.push_back(coord3);
                        coord_vector.push_back(coord0);
                        coord_vector.push_back(coord3);
                        coord_vector.push_back(coord2);
                    }
                }
            };
            
            add_face( {0,1,0}, {0,0,1}, {+magic,0,0} ); // +x face
            add_face( {0,0,1}, {0,1,0}, {-magic,0,0} ); // -x face
            add_face( {0,0,1}, {1,0,0}, {0,+magic,0} ); // +y face
            add_face( {1,0,0}, {0,0,1}, {0,-magic,0} ); // -y face
            add_face( {1,0,0}, {0,1,0}, {0,0,+magic} ); // +z face
            add_face( {0,1,0}, {1,0,0}, {0,0,-magic} ); // -z face
            
            vertex_count = int(coord_vector.size());
            
            PANIC_IF_GL_ERROR;
            glGenBuffers(1, &vertex_buffer_id);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
            glBufferData(
                GL_ARRAY_BUFFER,
                3 * sizeof(float) * coord_vector.size(),
                coord_vector.data(),
                GL_STATIC_DRAW
            );
            buffers_initialized = true;
        }
        
        static bool initialized0 = false;
        static GLuint vao0;
        static GLuint program0_id;
        
        static GLint view_matrix_idx0;
        static GLint proj_matrix_idx0;
        static GLint color_idx0;
        static GLint sphere_origin_idx0;
        static GLint radius_idx0;
        static GLint sphere_coord_idx0 = 0;
        
        static const char vs0_source[] =
            "#version 330\n"
            "precision mediump float;\n"
            "uniform mat4 view_matrix;\n"
            "uniform mat4 proj_matrix;\n"
            "uniform vec3 color;\n"
            "uniform float radius;\n"
            "uniform vec3 sphere_origin;\n"
            
            "layout(location=0) in vec3 sphere_coord;\n"
            
            "out vec4 varying_color;\n"
            
            "void main() {\n"
                "vec4 coord = vec4(radius*sphere_coord + sphere_origin, 1.0);\n"
                "gl_Position = proj_matrix * view_matrix * coord;\n"
                "varying_color = vec4(color, 1.0);\n"
            "}\n"
        ;
        static const char fs0_source[] =
            "#version 330\n"
            "precision mediump float;\n"
            "in vec4 varying_color;\n"
            "layout(location=0) out vec4 fragment_color;\n"
            "void main() { fragment_color = varying_color; }\n"
        ;
        if (!initialized0) {
            PANIC_IF_GL_ERROR;
            program0_id = make_program(vs0_source, fs0_source);
            
            PANIC_IF_GL_ERROR;
            glGenVertexArrays(1, &vao0);
            glBindVertexArray(vao0);
            
            view_matrix_idx0 = glGetUniformLocation(program0_id, "view_matrix");
            proj_matrix_idx0 = glGetUniformLocation(program0_id, "proj_matrix");
            color_idx0 = glGetUniformLocation(program0_id, "color");
            sphere_origin_idx0 = glGetUniformLocation(program0_id, "sphere_origin");
            radius_idx0 = glGetUniformLocation(program0_id, "radius");
            
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
            
            glVertexAttribPointer(
                sphere_coord_idx0,
                3,
                GL_FLOAT,
                false,
                3*sizeof(float),
                (void*)0
            );
            glEnableVertexAttribArray(sphere_coord_idx0);
            PANIC_IF_GL_ERROR;
            
            initialized0 = true;
        }
        
        glUseProgram(program0_id);
        glBindVertexArray(vao0);
        
        glUniformMatrix4fv(view_matrix_idx0, 1, false, &view_matrix[0][0]);
        glUniformMatrix4fv(proj_matrix_idx0, 1, false, &proj_matrix[0][0]);
        
        for (Ball const& ball : list) {
            if (&ball == skip) continue;
            glUniform3f(color_idx0, ball.r, ball.g, ball.b);
            glUniform3fv(sphere_origin_idx0, 1, &ball.position[0]);
            glUniform1f(radius_idx0, ball.radius);
            
            printf("%i\n", (int)vertex_count);
            glDrawArrays(GL_TRIANGLES, 0, vertex_count);
        }
        
        glBindVertexArray(0);
    }
};

std::vector<BallRender> Ball::recycled_ball_render;

static bool handle_controls(glm::mat4* view_ptr, glm::mat4* proj_ptr) {
    glm::mat4& view = *view_ptr;
    glm::mat4& projection = *proj_ptr;
    static bool w, a, s, d, q, e, space;
    static float theta = 1.5707f, phi = 1.8f, radius = 2.0f;
    static float mouse_x, mouse_y;
    static glm::vec3 eye;
    
    bool no_quit = true;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
          default:
          break; case SDL_KEYDOWN:
            switch (event.key.keysym.scancode) {
              default:
              break; case SDL_SCANCODE_W: case SDL_SCANCODE_I: w = true;
              break; case SDL_SCANCODE_A: case SDL_SCANCODE_J: a = true;
              break; case SDL_SCANCODE_S: case SDL_SCANCODE_K: s = true;
              break; case SDL_SCANCODE_D: case SDL_SCANCODE_L: d = true;
              break; case SDL_SCANCODE_Q: case SDL_SCANCODE_U: q = true;
              break; case SDL_SCANCODE_E: case SDL_SCANCODE_O: e = true;
              break; case SDL_SCANCODE_SPACE:  space = true;
              }
            
          break; case SDL_KEYUP:
            switch (event.key.keysym.scancode) {
              default:
              break; case SDL_SCANCODE_W: case SDL_SCANCODE_I: w = false;
              break; case SDL_SCANCODE_A: case SDL_SCANCODE_J: a = false;
              break; case SDL_SCANCODE_S: case SDL_SCANCODE_K: s = false;
              break; case SDL_SCANCODE_D: case SDL_SCANCODE_L: d = false;
              break; case SDL_SCANCODE_Q: case SDL_SCANCODE_U: q = false;
              break; case SDL_SCANCODE_E: case SDL_SCANCODE_O: e = false;
              break; case SDL_SCANCODE_SPACE: space = false;
            }
          break; case SDL_MOUSEWHEEL:
            phi -= event.wheel.y * 0.04f;
            theta -= event.wheel.x * 0.04f;
          break; case SDL_MOUSEBUTTONDOWN: case SDL_MOUSEBUTTONUP:
            mouse_x = event.button.x;
            mouse_y = event.button.y;
          break; case SDL_MOUSEMOTION:
            mouse_x = event.motion.x;
            mouse_y = event.motion.y;
          break; case SDL_WINDOWEVENT:
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                event.window.event == SDL_WINDOWEVENT_RESIZED) {
                
                screen_x = event.window.data1;
                screen_y = event.window.data2;
            }
          break; case SDL_QUIT:
            no_quit = false;
        }
    }
    
    glm::vec3 forward_normal_vector(
        sinf(phi) * cosf(theta),
        cosf(phi),
        sinf(phi) * sinf(theta)
    );
    
    // Free-camera mode.
    auto right_vector = glm::cross(forward_normal_vector, glm::vec3(0,1,0));
    right_vector = glm::normalize(right_vector);
    auto up_vector = glm::cross(right_vector, forward_normal_vector);
    
    eye += 5e-2f * right_vector * (float)(d - a);
    eye += 5e-2f * forward_normal_vector * (float)(w - s);
    eye += 5e-2f * up_vector * (float)(e - q);
    
    if (space) {
        theta += 1e-4 * (mouse_x - screen_x*0.5f);
        phi +=   1e-4 * (mouse_y - screen_y*0.5f);
    }
    
    view = glm::lookAt(eye, eye+forward_normal_vector, glm::vec3(0,1,0));
    
    printf("%f %f %f : %f %f %f\n", eye[0], eye[1], eye[2], forward_normal_vector[0], forward_normal_vector[1], forward_normal_vector[2]);
    
    projection = glm::perspective(
        fovy_radians,
        float(screen_x)/screen_y,
        near_plane,
        far_plane
    );
    
    float y_plane_radius = tanf(fovy_radians / 2.0f);
    float x_plane_radius = y_plane_radius * screen_x / screen_y;
    float mouse_vcs_x = x_plane_radius * (2.0f * mouse_x / screen_x - 1.0f);
    float mouse_vcs_y = y_plane_radius * (1.0f - 2.0f * mouse_y / screen_y);
    glm::vec4 mouse_vcs(mouse_vcs_x, mouse_vcs_y, -1.0f, 1.0f);
    glm::vec4 mouse_wcs = glm::inverse(view) * mouse_vcs;
    
    return no_quit;
}

int Main(int, char** argv) {
    argv0 = argv[0];
    
    window = SDL_CreateWindow(
        "Bouncy",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        screen_x, screen_y,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (window == nullptr) {
        panic("Could not initialize window", SDL_GetError());
    }
    auto context = SDL_GL_CreateContext(window);
    if (context == nullptr) {
        panic("Could not create OpenGL context", SDL_GetError());
    }
    if (SDL_GL_MakeCurrent(window, context) < 0) {
        panic("SDL OpenGL context error", SDL_GetError());
    }
    
    ogl_LoadFunctions();
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0, 1, 1, 1);
    
    bool no_quit = true;
    uint32_t last_fps_print_time = 0;
    
    glm::mat4 view_matrix, proj_matrix;
    BallList list;
    
    list.emplace_back(glm::vec3(0,0,3), glm::vec3(0,20,0), 1, 0, 1, 0.5);
        
    while (no_quit) {
        no_quit = handle_controls(&view_matrix, &proj_matrix);
        
        glViewport(0, 0, screen_x, screen_y);
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        Ball::draw_list(view_matrix, proj_matrix, list);
        
        SDL_GL_SwapWindow(window);
        PANIC_IF_GL_ERROR;
    }
    
    return 0;
}

} // end anonymous namespace

int main(int argc, char** argv) {
    return Main(argc, argv);
}
