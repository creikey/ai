#include <stdio.h>
#include <time.h>

#include "base.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char **argv) {
    temp_arena = arena_new(1024 * Kb);

    if (!file_exists("mnist/train-labels-idx1-ubyte")) {
        printf("Please download mnist dataset\n");
        exit(1);
    }

    // load mnist into tensors
    Arena *train_arena = arena_new(256 * Mb); // training data set is around 180 megabytes in floats
    Tensor train_inputs; // (train count, 28, 28) each image to predict
    Tensor train_labels; // one hot encoded (train count, 10 digits) labeling
    {
        Arena *data_temp = arena_new(64 * Mb); // for loading the files

        uint32_t item_count;
        {
            String labels_data = file_load(data_temp, "mnist/train-labels-idx1-ubyte");
            ByteStream labels_stream = stream_string(data_temp, labels_data);
            uint32_t magic = stream_read_uint32_bigendian(&labels_stream);
            assert(magic == 2049);
            item_count = stream_read_uint32_bigendian(&labels_stream);

            train_labels = ten_new(train_arena, ten_shape(item_count, 10));
            for(int i = 0; i < item_count; i++) {
                uint8_t label = stream_read_uint8(&labels_stream);
                ten_index(data_temp, train_labels, i).data[label] = 1.0f;
            }
        }

        arena_reset(data_temp);
        {
            String inputs_data = file_load(data_temp, "mnist/train-images-idx3-ubyte");
            ByteStream stream = stream_string(data_temp, inputs_data);
            uint32_t magic = stream_read_uint32_bigendian(&stream);
            assert(magic == 0x00000803);
            uint32_t item_count = stream_read_uint32_bigendian(&stream);
            assert(item_count == train_labels.shape.shape[0]);
            uint32_t rows = stream_read_uint32_bigendian(&stream);
            assert(rows == 28);
            uint32_t cols = stream_read_uint32_bigendian(&stream);
            assert(cols == 28);

            train_inputs = ten_new(train_arena, ten_shape(item_count, 28, 28));
            Arena *imgdims = arena_new(128 * 28 * Kb);
            for(int i = 0; i < item_count; i++) {
                Tensor image = ten_index(imgdims, train_inputs, i);
                for(int row = 0; row < 28; row++) {
                    for(int col = 0; col < 28; col++) {
                        ten_index(imgdims, image, row).data[col] = (float)stream_read_uint8(&stream) / 255.0f;
                    }
                }
                arena_reset(imgdims);
            }
            arena_destroy(&imgdims);
        }

        arena_destroy(&data_temp);
    }

    // write a random image
    /*
    {
        srand(time(NULL));
        Arena *img_temp = arena_new(64 * Mb);
        uint8_t *as_grayscale_bytes = (uint8_t*)arena_alloc(img_temp, 28 * 28);
        int index = rand() % 100;
        Tensor image = ten_index(temp_arena, train_inputs, index);
        for(int i = 0; i < 28 * 28; i++) {
            as_grayscale_bytes[i] = (uint8_t)(image.data[i] * 255.0f);
        }
        stbi_write_png((const char *)tprint("input_%d.png", index).data, 28, 28, 1, as_grayscale_bytes, 28);
        arena_destroy(&img_temp);
    }
    */
}