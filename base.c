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
#define Gb (1024 * Mb)

typedef float float32_t;

typedef struct Arena {
    uint8_t *mem;
    uint64_t next;
    uint64_t size;
} Arena;

Arena *arena_new(uint64_t size) {
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

void *arena_alloc(Arena *arena, uint64_t size) {
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

Arena *arena_subarena(Arena *arena, uint64_t size) {
    Arena *subarena = (Arena *)arena_alloc(arena, sizeof(Arena));
    uint8_t *data = arena_alloc(arena, size);
    subarena->mem = data;
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
    uint64_t size;
} String;

#define STRL(str) (String){.data = str, .size = sizeof(str) - 1}
// Use %.*s to print a String or (data,size) pair. The .* means read width from arg
#define STRF(str) (int)(str).size, (str).data

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
    uint64_t size = ftell(file);
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

typedef struct Array {
    void *data;
    uint64_t element_size;
    uint64_t count;
    uint64_t cap;
} Array;

Array array_new(Arena *arena, uint64_t element_size, uint64_t cap) {
    Array result = {
        .data = arena_alloc(arena, element_size * cap),
        .element_size = element_size,
        .count = 0,
        .cap = cap,
    };
    return result;
}

void array_append(Arena *arena, Array *array, void *element) {
    if(array->count == array->cap) {
        array->cap = array->cap ? array->cap * 2 : 1;
        void *old_data = array->data;
        array->data = arena_alloc(arena, array->element_size * array->cap);
        if (old_data && array->count > 0) {
            memcpy(array->data, old_data, array->count * array->element_size);
        }
    }
    memcpy(array->data + array->count * array->element_size, element, array->element_size);
    array->count++;
}

#define array_appendval(arena, array, element_type, element_value) array_append(arena, array, &(element_type){element_value})

Array array_clone(Arena *arena, Array array) {
    Array result = array_new(arena, array.element_size, array.count);
    memcpy(result.data, array.data, array.count * array.element_size);
    result.count = array.count;
    return result;
}

void *array_index(Array array, uint64_t index) {
    assert(index < array.count);
    return array.data + index * array.element_size;
}

void *array_popfront(Array *array) {
    assert(array->count > 0);
    void *result = array->data;
    array->data = array->data + array->element_size;
    array->count--;
    array->cap--;
    return result;
}

void *array_popback(Array *array) {
    assert(array->count > 0);
    void *result = array->data + (array->count - 1) * array->element_size;
    array->count--;
    return result;
}

typedef struct ByteStream {
    void *stream;
    bool corrupted;
    void (*write)(void *stream, uint8_t *data, uint64_t size);
    void (*read)(void *stream, uint8_t *data, uint64_t size);
} ByteStream;

void stream_read(ByteStream *stream, void *data, uint64_t size) {
    if (stream->corrupted) {
        return;
    }
    stream->read(stream->stream, (uint8_t *)data, size);
}

void stream_write(ByteStream *stream, void *data, uint64_t size) {
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
    uint64_t cursor;
    String corrupted_message;
} StringStream;

void stream_string_read(void *stream, uint8_t *data, uint64_t size) {
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
