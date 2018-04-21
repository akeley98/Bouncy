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

#include <string>
#include <vector>
#include <list>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"
#include "glew.h"

namespace {

static void panic(const char* message, const char* reason) {
    fprintf(stderr, "JellyMcJelloFace: %s %s\n", message, reason);
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
    gravity = 4.0f;

constexpr int
    plus_x_index = 0,
    minus_x_index = 1,
    plus_y_index = 2,
    minus_y_index = 3,
    plus_z_index = 4,
    minus_z_index = 5;

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
            render = *recycled_ball_render.back();
            recycled_ball_render.pop();
        } else {
            PANIC_IF_OPENGL_ERROR;
            glGenFramebuffers(6, render.framebuffers);
            glGenTextures(1, &render.cubemap);
            PANIC_IF_OPENGL_ERROR;
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
        Ball& other_ball = *other_ball_ptr;
        glm::vec3 displacement = other_ball.position - position;
        float squared_distance = dot(displacement, displacement);
        float squared_radii = (radius + other_ball.radius)
                            * (radius + other_ball.radius);
        
        bool collision_course = dot(displacement, velocity-other.velocity) > 0;
        
        if (collision_course && squared_distance < squared_radii) {
            swap(velocity, other_ball.velocity);
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
        static GLuint element_buffer_id;
        static int element_count;
        
        // Initialize a vertex buffer with vertices of sphere with
        // radius 1 and an element buffer with indicies suitable for
        // drawing a sphere using the earlier vertex buffer and
        // GL_TRIANGLES draw mode.
        if (!buffers_initialized) {
            const int W = 36, H = 18;
            std::vector<glm::vec3> coord_vector;
            std::vector<uint16_t> element_vector;
            
            // Axis is along y-axis; phi is angle above xz-plane,
            // theta is angle around y-axis.
            for (int h = 1; h < H; ++h) {
                float phi = (M_PI/H)*h - M_PI/2;
                float cos_phi = cosf(phi);
                float sin_phi = sinf(phi);
                
                for (int w = 0; w < W; ++w) {
                    float theta = (2*M_PI/W)*w;
                    float cos_theta = cosf(theta);
                    float sin_theta = sinf(theta);
                    
                    coord_vector.emplace_back(
                        cos_theta*cos_phi, sin_phi, sin_theta*cos_phi
                    );
                }
                
                int top_element = int(coord_vector.size());
                coord_vector.emplace_back(0, 1, 0);
                int bottom_element = int(coord_vector.size());
                coord_vector.emplace_back(0, -1, 0);
                
                
            }
            
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
        static GLint sphere_coord_idx0;
        
        static const char vs0_source[] =
            "#version 330\n"
            
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
                "varying_color = color;\n"
            "}\n"
        ;
        static const char fs0_source[] =
            "#version 330\n"
            "in vec4 varying_color;\n"
            "layout(location=0) out vec4 fragment_color;\n"
            "void main() { fragment_color = varying_color;\n"
        ;
        if (!initialized0) {
            PANIC_IF_GL_ERROR;
            program0_id = make_program(vs0_source, fs0_source);
            
            PANIC_IF_GL_ERROR;
            glGenVertexArrays(1, &vao0);
            glBindVertexArray(vao0);
            
            view_matrix_idx0 = glGetUniformLocation(program0_id, "view_matrix");
            proj_matrix_idx0 = glGetUniformLocation(program0_id, "proj_matrix");
            
            initialized0 = true;
        }
    }
};

} // end anonymous namespace

int main() {
    
}
