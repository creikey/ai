#pragma once

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>

#define Kb (1024)
#define Mb (1024 * Kb)

typedef struct Arena {
    uint8_t *mem;
    size_t next;
    size_t size;
} Arena;

Arena *arena_new(size_t size) {
    Arena *arena = (Arena *)malloc(sizeof(Arena));
    arena->mem = malloc(size);
    arena->next = 0;
    arena->size = size;
    return arena;
}

void arena_destroy(Arena **arena) {
    free((*arena)->mem);
    free(*arena);
    *arena = NULL;
}

uint8_t *arena_alloc(Arena *arena, size_t size) {
    if(size == 0) {
        return 0;
    }
    // Align next to 8 bytes for 64-bit alignment
    arena->next = (arena->next + 7) & ~7;
    uint8_t *mem = arena->mem + arena->next;
    arena->next += size;
    assert(arena->next <= arena->size);
    memset(mem, 0, size);
    return mem;
}

Arena *arena_subarena(Arena *arena, size_t size) {
    Arena *subarena = (Arena *)arena_alloc(arena, sizeof(Arena));
    uint8_t *data = arena_alloc(arena, size);
    subarena->mem = (uint8_t *)arena->mem + arena->next;
    subarena->next = 0;
    subarena->size = size;
    return subarena;
}

void arena_reset(Arena *arena) {
    arena->next = 0;
}

Arena *temp_arena = NULL;


typedef struct String {
    uint8_t *data;
    size_t size;
} String;

#define STRING_LITERAL(str) (String){.data = str, .size = sizeof(str) - 1}
// Use %.*s to print a String or (data,size) pair. The .* means read width from arg
#define STRINGF(str) (int)(str).size, (str).data

String file_load(Arena *arena, const char *path) {
    FILE *file;
#ifdef _WIN32
    if (fopen_s(&file, path, "rb") != 0) {
        String empty = {NULL, 0};
        return empty;
    }
#else
    file = fopen(path, "rb");
    if (file == NULL) {
        String empty = {NULL, 0};
        return empty;
    }
#endif
    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void *mem = arena_alloc(arena, size);
    fread(mem, 1, size, file);
    fclose(file);
    String result = {mem, size};
    return result;
}

String tprint(const char *format, ...) {
    va_list args;
    va_start(args, format);
    va_list args2;
    va_copy(args2, args);
    int size = vsnprintf(NULL, 0, format, args) + 1;
    va_end(args);
    char *buffer = (char *)arena_alloc(temp_arena, size);
    vsnprintf(buffer, size, format, args2);
    buffer[size - 1] = '\0';
    va_end(args2);
    return (String){.data = (uint8_t*)buffer, .size = size - 1};
}

bool file_exists(const char *path) {
    FILE *file;
#ifdef _WIN32
    if (fopen_s(&file, path, "rb") != 0) {
        return false;
    }
#else
    file = fopen(path, "rb");
    if (file == NULL) {
        return false;
    }
#endif
    fclose(file);
    return true;
}

typedef struct TensorShape {
    size_t *shape;
    size_t rank;
} TensorShape;

typedef struct Tensor {
    float *data;
    TensorShape shape;
} Tensor;


TensorShape ten_shape_impl(size_t *sizes, size_t rank) {
    size_t *shape = (size_t *)arena_alloc(temp_arena, sizeof(size_t) * rank);
    memcpy(shape, sizes, sizeof(size_t) * rank);
    TensorShape result = {shape, rank};
    return result;
}

#define ten_shape(first_dim, ...) ten_shape_impl((size_t[]){first_dim, __VA_ARGS__}, sizeof((size_t[]){first_dim, __VA_ARGS__}) / sizeof(size_t))

Tensor ten_new(Arena *arena, TensorShape shape) {
    Tensor tensor = {NULL, {NULL, 0}};
    size_t size = 1;
    for (size_t i = 0; i < shape.rank; i++) {
        size *= shape.shape[i];
    }
    tensor.data = (float *)arena_alloc(arena, sizeof(float) * size);
    tensor.shape.rank = shape.rank;
    tensor.shape.shape = (size_t*)arena_alloc(arena, sizeof(size_t) * shape.rank);
    memcpy(tensor.shape.shape, shape.shape, sizeof(size_t) * shape.rank);
    return tensor;
}

Tensor ten_index(Arena *arena, Tensor tensor, size_t index) {
    Tensor result = tensor;
    int stride = 1;
    for(int i = 1; i < tensor.shape.rank; i++) {
        stride *= tensor.shape.shape[i];
    }
    result.data = tensor.data + index * stride;
    result.shape = (TensorShape){(size_t*)arena_alloc(arena, sizeof(size_t) * (tensor.shape.rank - 1)), tensor.shape.rank - 1};
    for (size_t i = 1; i < tensor.shape.rank; i++) {
        result.shape.shape[i - 1] = tensor.shape.shape[i];
    }
    return result;
}

Tensor ten_matmul(Arena *arena, Tensor a, Tensor b) {
    assert(a.shape.shape[a.shape.rank - 1] == b.shape.shape[0]);
    Tensor result = ten_new(arena, ten_shape(a.shape.shape[0], b.shape.shape[1]));
    for(int i = 0; i < a.shape.shape[0]; i++) {
        for(int j = 0; j < b.shape.shape[1]; j++) {
            for(int k = 0; k < a.shape.shape[1]; k++) {
                result.data[i * b.shape.shape[1] + j] += a.data[i * a.shape.shape[1] + k] * b.data[k * b.shape.shape[1] + j];
            }
        }
    }
    return result;
}

typedef struct ByteStream {
    void *stream;
    bool corrupted;
    void (*write)(void *stream, uint8_t *data, size_t size);
    void (*read)(void *stream, uint8_t *data, size_t size);
} ByteStream;

void stream_read(ByteStream *stream, void *data, size_t size) {
    if (stream->corrupted) {
        return;
    }
    stream->read(stream->stream, (uint8_t *)data, size);
}

void stream_write(ByteStream *stream, void *data, size_t size) {
    if (stream->corrupted) {
        return;
    }
    stream->write(stream->stream, (uint8_t *)data, size);
}

uint32_t stream_read_uint32(ByteStream *stream) {
    uint32_t result = 0;
    stream_read(stream, (uint8_t *)&result, sizeof(uint32_t));
    return result;
}

uint8_t stream_read_uint8(ByteStream *stream) {
    uint8_t result = 0;
    stream_read(stream, (uint8_t *)&result, sizeof(uint8_t));
    return result;
}

uint32_t stream_read_uint32_bigendian(ByteStream *stream) {
    uint32_t result = 0;
    for(int i = 0; i < 4; i += 1) {
        uint8_t byte = stream_read_uint8(stream);
        result |= ((uint32_t)byte) << ((3 - i) * 8);
    }
    return result;
}

typedef struct StringStream {
    String string;
    size_t cursor;
    String corrupted_message;
} StringStream;

void stream_string_read(void *stream, uint8_t *data, size_t size) {
    StringStream *string_stream = (StringStream *)stream;
    if (string_stream->cursor + size > string_stream->string.size) {
        string_stream->corrupted_message = tprint("%zu bytes requested, but only %zu bytes available, cursor %zu", size, string_stream->string.size - string_stream->cursor, string_stream->cursor);
        return;
    }
    memcpy(data, string_stream->string.data + string_stream->cursor, size);
    string_stream->cursor += size;
    assert(string_stream->cursor <= string_stream->string.size);
}

ByteStream stream_string(Arena *arena_for_stream, String from_string) {
    StringStream *stream = (StringStream *)arena_alloc(arena_for_stream, sizeof(StringStream));
    stream->string = from_string;
    stream->cursor = 0;
    ByteStream result = {
        .corrupted = false,
        .write = NULL,
        .stream = stream,
        .read = stream_string_read,
    };
    return result;
}

// Tensor utility functions for perceptron
int ten_argmax(Tensor tensor) {
    int idx = 0;
    for (int i = 1; i < tensor.shape.shape[0]; i++) {
        if (tensor.data[i] > tensor.data[idx]) idx = i;
    }
    return idx;
}

float ten_softmax_loss(Tensor logits, int correct_label) {
    float max = logits.data[0];
    for (int i = 1; i < logits.shape.shape[0]; i++) {
        if (logits.data[i] > max) max = logits.data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < logits.shape.shape[0]; i++) {
        sum += expf(logits.data[i] - max);
    }
    float log_sum = max + logf(sum);
    return logits.data[correct_label] - log_sum;
}

Tensor ten_softmax_gradients(Arena *arena, Tensor logits, int correct_label) {
    Tensor gradients = ten_new(arena, logits.shape);
    
    // Find max for numerical stability
    float maxl = logits.data[0];
    for (int j = 1; j < logits.shape.shape[0]; j++) {
        if (logits.data[j] > maxl) maxl = logits.data[j];
    }
    
    // Compute softmax probabilities
    float sum_exp = 0.0f;
    for (int j = 0; j < logits.shape.shape[0]; j++) {
        sum_exp += expf(logits.data[j] - maxl);
    }
    
    // Compute gradients: softmax - one_hot
    for (int j = 0; j < logits.shape.shape[0]; j++) {
        gradients.data[j] = expf(logits.data[j] - maxl) / sum_exp;
    }
    gradients.data[correct_label] -= 1.0f;
    
    return gradients;
}

size_t ten_size(Tensor tensor) {
    size_t size = 1;
    for (size_t i = 0; i < tensor.shape.rank; i++) {
        size *= tensor.shape.shape[i];
    }
    return size;
}

Tensor ten_reshape(Arena *arena, Tensor tensor, TensorShape new_shape) {
    Tensor result = tensor;
    result.shape = new_shape;
    assert(ten_size(tensor) == ten_size(result));
    return result;
}

Tensor ten_flatten(Arena *arena, Tensor tensor) {
    TensorShape flat_shape = ten_shape(ten_size(tensor));
    return ten_reshape(arena, tensor, flat_shape);
}

// Initialize tensor with zeros
void ten_zero(Tensor tensor) {
    size_t size = ten_size(tensor);
    for (size_t i = 0; i < size; i++) {
        tensor.data[i] = 0.0f;
    }
}

// Initialize tensor with random values from min to max
void ten_rand(Tensor tensor, float min, float max) {
    size_t size = ten_size(tensor);
    for (size_t i = 0; i < size; i++) {
        tensor.data[i] = min + ((float)rand() / RAND_MAX) * (max - min);
    }
}

// ReLU activation function
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Apply ReLU activation to a tensor in-place
void ten_relu(Tensor tensor) {
    size_t size = ten_size(tensor);
    for (size_t i = 0; i < size; i++) {
        tensor.data[i] = relu(tensor.data[i]);
    }
}

// ReLU gradient (derivative)
float relu_gradient(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Apply ReLU gradient to a tensor in-place
void ten_relu_gradient(Tensor tensor) {
    size_t size = ten_size(tensor);
    for (size_t i = 0; i < size; i++) {
        tensor.data[i] = relu_gradient(tensor.data[i]);
    }
}

// Xavier/Glorot weight initialization for better training stability
void ten_xavier_init(Tensor tensor, int input_size) {
    float scale = sqrtf(2.0f / input_size);
    size_t size = ten_size(tensor);
    for (size_t i = 0; i < size; i++) {
        tensor.data[i] = (2.0f * ((float)rand() / RAND_MAX) - 1.0f) * scale;
    }
}

// Add bias to activations (works for any rank tensor)
void ten_add_bias(Tensor activations, Tensor bias) {
    // For 2D tensor: (batch_size, output_dim) + (output_dim,)
    // For 1D tensor: (output_dim,) + (output_dim,)
    size_t activations_size = ten_size(activations);
    size_t bias_size = ten_size(bias);
    
    // Handle different tensor shapes
    if (activations.shape.rank == 2 && bias.shape.rank == 1) {
        // (batch_size, output_dim) + (output_dim,)
        int batch_size = activations.shape.shape[0];
        int output_dim = activations.shape.shape[1];
        assert(output_dim == bias.shape.shape[0]);
        
        for (int b = 0; b < batch_size; b++) {
            for (int j = 0; j < output_dim; j++) {
                activations.data[b * output_dim + j] += bias.data[j];
            }
        }
    } else if (activations.shape.rank == 1 && bias.shape.rank == 1) {
        // (output_dim,) + (output_dim,)
        assert(activations_size == bias_size);
        for (size_t i = 0; i < activations_size; i++) {
            activations.data[i] += bias.data[i];
        }
    } else {
        assert(false && "Unsupported tensor shapes for bias addition");
    }
}
