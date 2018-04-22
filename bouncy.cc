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
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

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
    ball_radius = 0.1f,
    ball_core_radius_ratio = 0.707f,
    min_x = -0.8f,
    max_x = +0.8f,
    min_y = 0.0f,
    max_y = 2.0f,
    min_z = -0.8f,
    max_z = +0.8f,
    ticks_per_second = 600.0f,
    gravity = 4.0f,
    fovy_radians = 1.4f,
    near_plane = 0.01f,
    far_plane = 20.0f,
    camera_speed = 8e-2;

constexpr int
    plus_x_index = 0,
    minus_x_index = 1,
    plus_y_index = 2,
    minus_y_index = 3,
    plus_z_index = 4,
    minus_z_index = 5,
    ball_cubemap_dim = 512,
    ball_count = 25;

GLenum cubemap_face_enums[] = {
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
};

int screen_x = 1280, screen_y = 960;
bool paused = false, do_one_tick = false;
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

static void draw_skybox(
    glm::mat4 view_matrix, glm::mat4 proj_matrix
);

static void draw_scene(
    glm::mat4 view_matrix, glm::mat4 proj_matrix,
    BallList const& list, Ball const* skip=nullptr
);

class Ball {
    glm::vec3 position;
    glm::vec3 velocity;
    float r, g, b, radius;
    bool bounced = false;
    BallRender render;
    
    static std::vector<BallRender> recycled_ball_render;
  public:
    Ball(glm::vec3 pos_arg, glm::vec3 vel_arg,
         float r_arg, float g_arg, float b_arg, float radius_arg)
    {
        position = pos_arg;
        velocity = vel_arg;
        r = r_arg;
        g = g_arg;
        b = b_arg;
        bounced = false;
        radius = radius_arg;
        
        if (!recycled_ball_render.empty()) {
            render = recycled_ball_render.back();
            recycled_ball_render.pop_back();
        } else {
            PANIC_IF_GL_ERROR;
            GLuint depth_buffers[6];
            glGenFramebuffers(6, render.framebuffers);
            glGenRenderbuffers(6, depth_buffers);
            glGenTextures(1, &render.cubemap);
            glBindTexture(GL_TEXTURE_CUBE_MAP, render.cubemap);
            
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);   
            
            PANIC_IF_GL_ERROR;
            
            for (int i = 0; i < 6; ++i) {
                glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[i]);
                
                PANIC_IF_GL_ERROR;
                
                auto face = cubemap_face_enums[i];
                
                PANIC_IF_GL_ERROR;
                glTexImage2D(
                    face, 0, GL_RGB, ball_cubemap_dim, ball_cubemap_dim, 0,
                    GL_RGB, GL_UNSIGNED_BYTE, 0
                );
                
                PANIC_IF_GL_ERROR;
                
                glBindRenderbuffer(GL_RENDERBUFFER, depth_buffers[i]);
                
                PANIC_IF_GL_ERROR;
                glRenderbufferStorage(
                    GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
                    ball_cubemap_dim, ball_cubemap_dim
                );
                PANIC_IF_GL_ERROR;
                glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                    GL_RENDERBUFFER, depth_buffers[i]
                );
                PANIC_IF_GL_ERROR;
                glFramebufferTexture2D(
                    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    cubemap_face_enums[i], render.cubemap, 0
                );
                GLenum tmp = GL_COLOR_ATTACHMENT0;
                glDrawBuffers(1, &tmp);
                PANIC_IF_GL_ERROR;
                
/*                if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    printf("%i\n", (int)glCheckFramebufferStatus(GL_FRAMEBUFFER));
                    panic("Apocalypse", "Framebuffer Frobnication Error");
                }*/
            }
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
        
        if (position[0] + radius > max_x && velocity[0] > 0) {
            velocity[0] *= -1.0f;
            position[0] = max_x - radius;
            flag = true;
        }
        if (position[1] + radius > max_y && velocity[1] > 0) {
            velocity[1] *= -1.0f;
            position[1] = max_y - radius;
            flag = true;
        }
        if (position[2] + radius > max_z && velocity[2] > 0) {
            velocity[2] *= -1.0f;
            position[2] = max_z - radius;
            flag = true;
        }
        if (position[0] - radius < min_x && velocity[0] < 0) {
            velocity[0] *= -1.0f;
            position[0] = min_x + radius;
            flag = true;
        }
        if (position[1] - radius < min_y && velocity[1] < 0) {
            velocity[1] *= -1.0f;
            position[1] = min_y + radius;
            flag = true;
        }
        if (position[2] - radius < min_z && velocity[2] < 0) {
            velocity[2] *= -1.0f;
            position[2] = min_z + radius;
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
        Ball const* skip=nullptr
    ) {
        static bool buffers_initialized = false;
        static GLuint vertex_buffer_id;
        static int vertex_count;
        
        // Initialize a vertex buffer with vertices of sphere with
        // radius 1 suitable for use with GL_TRIANGLES draw mode.
        if (!buffers_initialized) {
            std::vector<glm::vec3> coord_vector;
            
            const float magic = 5;
            
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
        static GLint reflection_cubemap_idx0;
        static GLint eye_idx0;
        static GLint sphere_coord_idx0 = 0;
        
        static const char vs0_source[] =
            "#version 330\n"
            "precision mediump float;\n"
            "uniform mat4 view_matrix;\n"
            "uniform mat4 proj_matrix;\n"
            "uniform vec3 color;\n"
            "uniform float radius;\n"
            "uniform vec3 sphere_origin;\n"
            "uniform vec3 eye;\n"
            
            "layout(location=0) in vec3 sphere_coord;\n"
            
            "out vec3 surface_color;\n"
            "out vec3 reflected_vector;\n"
            "void main() {\n"
                "vec4 coord = vec4(radius*sphere_coord + sphere_origin, 1.0);\n"
                "gl_Position = proj_matrix * view_matrix * coord;\n"
                "surface_color = color;\n"
                "reflected_vector = reflect(coord.xyz - eye, sphere_coord);\n"
            "}\n"
        ;
        static const char fs0_source[] =
            "#version 330\n"
            "precision mediump float;\n"
            "uniform samplerCube reflection_cubemap;\n"
            "in vec3 reflected_vector;\n"
            "in vec3 surface_color;\n"
            "layout(location=0) out vec4 fragment_color;\n"
            "void main() { \n"
                "vec3 c = 0.2*surface_color + 0.8*texture(reflection_cubemap,reflected_vector).rgb;\n"
                "fragment_color = vec4(c,1.0);\n"
            "}\n"
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
            eye_idx0 = glGetUniformLocation(program0_id, "eye");
            reflection_cubemap_idx0 = glGetUniformLocation(
                program0_id, "reflection_cubemap"
            );
            
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
        glm::vec3 eye = inverse(view_matrix) * glm::vec4(0,0,0,1);
        glUniform3fv(eye_idx0, 1, &eye[0]);
        
        for (Ball const& ball : list) {
            if (&ball == skip) continue;
            
            glUniform3f(color_idx0, ball.r, ball.g, ball.b);
            glUniform3fv(sphere_origin_idx0, 1, &ball.position[0]);
            glUniform1f(radius_idx0, ball.radius * ball_core_radius_ratio);
            
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, ball.render.cubemap);
            glUniform1i(reflection_cubemap_idx0, 0);
            
            glDrawArrays(GL_TRIANGLES, 0, vertex_count);
        }
        
        static bool initialized1 = false;
        static GLuint vao1;
        static GLuint program1_id;
        
        static GLint view_matrix_idx1;
        static GLint proj_matrix_idx1;
        static GLint sphere_origin_idx1;
        static GLint radius_idx1;
        static GLint eye_idx1;
        static GLint sphere_coord_idx1 = 0;
        
        static const char vs1_source[] =
            "#version 330\n"
            "precision mediump float;\n"
            "uniform mat4 view_matrix;\n"
            "uniform mat4 proj_matrix;\n"
            "uniform vec3 sphere_origin;\n"
            "uniform float radius;\n"
            
            "layout(location=0) in vec3 sphere_coord;\n"
            "out vec3 varying_normal;\n"
            "out vec3 varying_pos;\n"
            
            "void main() { \n"
                "vec3 coord3 = radius*sphere_coord + sphere_origin;\n"
                "vec4 coord = vec4(coord3, 1.0);\n"
                "gl_Position = proj_matrix * view_matrix * coord;\n"
                "varying_normal = sphere_coord;\n"
                "varying_pos = coord3;\n"
            "}\n"
        ;
        static const char fs1_source[] =
            "#version 330\n"
            "precision mediump float;\n"
            "in vec3 varying_normal;\n"
            "in vec3 varying_pos;\n"
            "out vec4 frag_color;\n"
            "uniform mat4 view_matrix;\n"
            "uniform vec3 eye;\n"
            "void main() { \n"
                "float Dot = dot(normalize(eye-varying_pos),\n"
                                "normalize(varying_normal));\n"
                "float f = Dot*Dot*0.6;\n"
                "frag_color = vec4(f,f,f,0.4-Dot*0.15);\n"
            "}\n"
        ;
        
        if (!initialized1) {
            PANIC_IF_GL_ERROR;
            program1_id = make_program(vs1_source, fs1_source);
            
            PANIC_IF_GL_ERROR;
            glGenVertexArrays(1, &vao1);
            glBindVertexArray(vao1);
            
            view_matrix_idx1 = glGetUniformLocation(program1_id, "view_matrix");
            proj_matrix_idx1 = glGetUniformLocation(program1_id, "proj_matrix");
            sphere_origin_idx1 = glGetUniformLocation(
                program1_id, "sphere_origin");
            radius_idx1 = glGetUniformLocation(program1_id, "radius");
            eye_idx1 = glGetUniformLocation(program1_id, "eye");
            
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
            glVertexAttribPointer(
                sphere_coord_idx1,
                3,
                GL_FLOAT,
                false,
                3*sizeof(float),
                (void*)0
            );
            glEnableVertexAttribArray(sphere_coord_idx1);
            PANIC_IF_GL_ERROR;
            
            initialized1 = true;
        }
        
        glUseProgram(program1_id);
        glBindVertexArray(vao1);
        
        glUniformMatrix4fv(view_matrix_idx1, 1, false, &view_matrix[0][0]);
        glUniformMatrix4fv(proj_matrix_idx1, 1, false, &proj_matrix[0][0]);
        glUniform3fv(eye_idx1, 1, &eye[0]);
        
        glDepthMask(GL_FALSE);
        
        for (Ball const& ball : list) {
            if (&ball == skip) continue;
            
            glUniform3fv(sphere_origin_idx1, 1, &ball.position[0]);
            glUniform1f(radius_idx1, ball.radius);
            
            glDrawArrays(GL_TRIANGLES, 0, vertex_count);
        }
        glDepthMask(GL_TRUE);
        glBindVertexArray(0);
    }
    
    void update_reflection_texture(BallList const& list) {
        glm::mat4 view_matrix;
        glm::mat4 proj_matrix = glm::perspective(
            1.5707963267948966f, 1.0f, radius*0.1f, far_plane
        );
        glm::vec3 const& v = position;
        
        glViewport(0, 0, ball_cubemap_dim, ball_cubemap_dim);
        
        glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[plus_x_index]);
        view_matrix = glm::lookAt(v, v+glm::vec3(1,0,0), glm::vec3(0,-1,0));
        draw_scene(view_matrix, proj_matrix, list, this);
        
        glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[minus_x_index]);
        view_matrix = glm::lookAt(v, v+glm::vec3(-1,0,0), glm::vec3(0,-1,0));
        draw_scene(view_matrix, proj_matrix, list, this);
        
        glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[plus_y_index]);
            view_matrix = glm::lookAt(v, v+glm::vec3(0,1,0), glm::vec3(0,0,1));
        draw_scene(view_matrix, proj_matrix, list, this);
        
        glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[minus_y_index]);
        view_matrix = glm::lookAt(v, v+glm::vec3(0,-1,0), glm::vec3(0,0,-1));
        draw_scene(view_matrix, proj_matrix, list, this);
        
        glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[plus_z_index]);
        view_matrix = glm::lookAt(v, v+glm::vec3(0,0,1), glm::vec3(0,-1,0));
        draw_scene(view_matrix, proj_matrix, list, this);
        
        glBindFramebuffer(GL_FRAMEBUFFER, render.framebuffers[minus_z_index]);
        view_matrix = glm::lookAt(v, v+glm::vec3(0,0,-1), glm::vec3(0,-1,0));
        draw_scene(view_matrix, proj_matrix, list, this);
    }
};

std::vector<BallRender> Ball::recycled_ball_render;

static void load_cubemap_face(GLenum face, const char* filename) {
    std::string full_filename = argv0 + "Tex/" + filename;
    SDL_Surface* surface = SDL_LoadBMP(full_filename.c_str());
    if (surface == nullptr) {
        panic(SDL_GetError(), full_filename.c_str()); 
    }
    if (surface->w != 512 || surface->h != 512) {
        panic("Expected 512x512 texture", full_filename.c_str());
    }
    if (surface->format->format != SDL_PIXELFORMAT_BGR24) {
        fprintf(stderr, "%i\n", (int)surface->format->format);
        panic("Expected 24-bit BGR bitmap", full_filename.c_str());
    }
    
    glTexImage2D(face, 0, GL_RGB, 512, 512, 0,
                  GL_BGR, GL_UNSIGNED_BYTE, surface->pixels);
    
    SDL_FreeSurface(surface);
}

static GLuint load_cubemap() {
    GLuint id = 0;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_CUBE_MAP, id);
    
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_LOD, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LOD, 8);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0); 
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 8); 
    
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, "left.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_POSITIVE_X, "right.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, "bottom.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, "top.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, "back.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, "front.bmp");
    
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    PANIC_IF_GL_ERROR;
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    
    return id;
}

static const char skybox_vs_source[] =
"#version 330\n"
"layout(location=0) in vec3 position;\n"
"out vec3 texture_coordinate;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"void main() {\n"
    "vec4 v = view_matrix * vec4(10*position, 0.0);\n"
    "gl_Position = proj_matrix * vec4(v.xyz, 1);\n"
    "texture_coordinate = position;\n"
"}\n";

static const char skybox_fs_source[] =
"#version 330\n"
"in vec3 texture_coordinate;\n"
"out vec4 color;\n"
"uniform samplerCube cubemap;\n"
"void main() {\n"
    "vec4 c = texture(cubemap, texture_coordinate);\n"
    "c.a = 1.0;\n"
    "color = c;\n"
"}\n";

static const float skybox_vertices[24] = {
    -1, 1, 1,
    -1, -1, 1,
    1, -1, 1,
    1, 1, 1,
    -1, 1, -1,
    -1, -1, -1,
    1, -1, -1,
    1, 1, -1,
};

static const GLushort skybox_elements[36] = {
    7, 4, 5, 7, 5, 6,
    1, 0, 3, 1, 3, 2,
    5, 1, 2, 5, 2, 6,
    4, 7, 3, 4, 3, 0,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6
};

static void draw_skybox(
    glm::mat4 view_matrix, glm::mat4 proj_matrix
) {
    static bool cubemap_loaded = false;
    static GLuint cubemap_texture_id;
    if (!cubemap_loaded) {
        cubemap_texture_id = load_cubemap();
        cubemap_loaded = true;
    }
    
    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLint view_matrix_id;
    static GLint proj_matrix_id;
    static GLint cubemap_uniform_id;
    
    if (vao == 0) {
        program_id = make_program(skybox_vs_source, skybox_fs_source);
        view_matrix_id = glGetUniformLocation(program_id, "view_matrix");
        proj_matrix_id = glGetUniformLocation(program_id, "proj_matrix");
        cubemap_uniform_id = glGetUniformLocation(program_id, "cubemap");
        
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        
        glGenBuffers(1, &vertex_buffer_id);
        glGenBuffers(1, &element_buffer_id);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, sizeof skybox_elements,
            skybox_elements, GL_STATIC_DRAW
        );
        
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        glBufferData(
            GL_ARRAY_BUFFER, sizeof skybox_vertices,
            skybox_vertices, GL_STATIC_DRAW
        ); 
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            false,
            sizeof(float) * 3,
            (void*)0
        );
        glEnableVertexAttribArray(0);
        PANIC_IF_GL_ERROR;
    }
    
    glUseProgram(program_id);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id);
    glUniform1i(cubemap_uniform_id, 0);
    
    glUniformMatrix4fv(view_matrix_id, 1, 0, &view_matrix[0][0]);
    glUniformMatrix4fv(proj_matrix_id, 1, 0, &proj_matrix[0][0]);
    
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, (void*)0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    
    PANIC_IF_GL_ERROR;
}

static void draw_scene(
    glm::mat4 view_matrix,
    glm::mat4 proj_matrix,
    BallList const& list,
    Ball const* skip
) {
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    draw_skybox(view_matrix, proj_matrix);
    Ball::draw_list(view_matrix, proj_matrix, list, skip);
}

static bool handle_controls(glm::mat4* view_ptr, glm::mat4* proj_ptr) {
    glm::mat4& view = *view_ptr;
    glm::mat4& projection = *proj_ptr;
    static bool w, a, s, d, q, e, space, shift;
    static float theta = 1.5707f, phi = 1.8f;
    static float mouse_x, mouse_y;
    static glm::vec3 eye(0, 0, 2*min_z);
    
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
              break; case SDL_SCANCODE_LSHIFT: case SDL_SCANCODE_RSHIFT:
                shift = true;
              break; case SDL_SCANCODE_TAB: paused = !paused;
              break; case SDL_SCANCODE_RETURN: do_one_tick = true;
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
              break; case SDL_SCANCODE_LSHIFT: case SDL_SCANCODE_RSHIFT:
                shift = false;
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
    
    if (phi < 0.01f) phi = 0.01f;
    if (phi > 3.14f) phi = 3.14f;
    
    glm::vec3 forward_normal_vector(
        sinf(phi) * cosf(theta),
        cosf(phi),
        sinf(phi) * sinf(theta)
    );
    
    // Free-camera mode.
    auto right_vector = glm::cross(forward_normal_vector, glm::vec3(0,1,0));
    right_vector = glm::normalize(right_vector);
    auto up_vector = glm::cross(right_vector, forward_normal_vector);
    
    float V = shift ? camera_speed * 0.2f : camera_speed;
    eye += V * right_vector * (float)(d - a);
    eye += V * forward_normal_vector * (float)(w - s);
    eye += V * up_vector * (float)(e - q);
    
    if (space) {
        theta += 1e-4 * (mouse_x - screen_x*0.5f);
        phi +=   1e-4 * (mouse_y - screen_y*0.5f);
    }
    
    view = glm::lookAt(eye, eye+forward_normal_vector, glm::vec3(0,1,0));
    
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
    glEnable(GL_TEXTURE_CUBE_MAP);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glClearColor(0, 1, 1, 1);
    
    bool no_quit = true;
    
    glm::mat4 view_matrix, proj_matrix;
    BallList list;
    
    srandom(static_cast<unsigned int>(time(nullptr)));
    auto rnd = [](float min, float max) {
        float result = uint16_t(random()) * ((max-min)/65535.f) + min;
        return result;
    };
    
    for (int i = 0; i < ball_count; ++i) {
        list.emplace_back(
            glm::vec3(rnd(min_x, max_x), rnd(min_y, max_y), rnd(min_z, max_z)),
            glm::vec3(rnd(-3, 3), rnd(-3, 3), rnd(-3, 3)),
            rnd(0, 1), rnd(0, 1), rnd(0, 1), ball_radius
        );
    }
    
    auto previous_update = SDL_GetTicks();
    auto previous_fps_print = SDL_GetTicks();
    int frames = 0;
    
    while (no_quit) {
        auto current_tick = SDL_GetTicks();
        if (current_tick >= previous_update + 16) {
            no_quit = handle_controls(&view_matrix, &proj_matrix);
            previous_update += 16;
            if (current_tick - previous_update > 100) {
                previous_update = current_tick;
            }
            
            if (!paused || do_one_tick) {
                do_one_tick = false;
                for (Ball& ball : list) {
                    ball.reset_bounce_flag();
                }
                
                for (Ball& ball : list) {
                    ball.bounce_bounds();
                    ball.tick(0.01f);
                }
                
                for (Ball& ball : list) {
                    for (Ball& other : list) {
                        if (&ball != &other
                            && !ball.bounce_flag()
                            && !other.bounce_flag()
                        ) {
                            ball.bounce_ball(&other);
                        }
                    }
                }
            }
            ++frames;
            if (current_tick >= previous_fps_print + 2000) {
                float fps = 1000.0 * frames / (current_tick-previous_fps_print);
                printf("%4.1f FPS\n", fps);
                previous_fps_print = current_tick;
                frames = 0;
            }
        }
        
        for (Ball& ball : list) {
            ball.update_reflection_texture(list);
        }
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, screen_x, screen_y);
        draw_scene(view_matrix, proj_matrix, list);
        SDL_GL_SwapWindow(window);
        PANIC_IF_GL_ERROR;
    }
    
    return 0;
}

} // end anonymous namespace

int main(int argc, char** argv) {
    return Main(argc, argv);
}
