// test with clang perceptron.c && ./a.out and making sure the accuracy is above 90%

#include <stdio.h>
#include <time.h>
#include <math.h>

#include "base.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char **argv) {
    temp_arena = arena_new(1000 * Mb);

    Tensor a = ten_new(temp_arena, ten_shape(10, 10));
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            ten_index(temp_arena, a, i).data[j] = i * 10 + j;
        }
    }

    if (!file_exists("mnist/train-labels.idx1-ubyte")) {
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
            String labels_data = file_load(data_temp, "mnist/train-labels.idx1-ubyte");
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
            String inputs_data = file_load(data_temp, "mnist/train-images.idx3-ubyte");
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
    arena_reset(temp_arena);

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

    // After MNIST loading, add perceptron implementation
    // --- Perceptron parameters ---
    int input_size = 28 * 28;
    int num_classes = 10;
    int train_count = train_inputs.shape.shape[0];
    int test_count = train_count / 10; // 10% for test
    int actual_train_count = train_count - test_count;

    // Flatten train_inputs to (train_count, 784)
    Arena *flat_arena = arena_new(256 * Mb);
    Tensor flat_inputs = ten_new(flat_arena, ten_shape(train_count, input_size));
    for (int i = 0; i < train_count; i++) {
        Tensor img = ten_index(temp_arena, train_inputs, i);
        Tensor flat_row = ten_index(temp_arena, flat_inputs, i);
        for (int row = 0; row < 28; row++) {
            Tensor img_row = ten_index(temp_arena, img, row);
            for (int col = 0; col < 28; col++) {
                flat_row.data[row * 28 + col] = img_row.data[col];
            }
        }
    }

    // Split into train and test sets
    Tensor X_train = flat_inputs;
    Tensor X_test = ten_new(temp_arena, ten_shape(test_count, input_size));
    Tensor y_train = train_labels;
    Tensor y_test = ten_new(temp_arena, ten_shape(test_count, num_classes));
    
    // Copy test data to separate tensors
    for (int i = 0; i < test_count; i++) {
        Tensor src_row = ten_index(temp_arena, flat_inputs, actual_train_count + i);
        Tensor dst_row = ten_index(temp_arena, X_test, i);
        for (int j = 0; j < input_size; j++) {
            dst_row.data[j] = src_row.data[j];
        }
        
        Tensor src_label = ten_index(temp_arena, train_labels, actual_train_count + i);
        Tensor dst_label = ten_index(temp_arena, y_test, i);
        for (int j = 0; j < num_classes; j++) {
            dst_label.data[j] = src_label.data[j];
        }
    }

    // --- Perceptron weights and bias as Tensors ---
    Tensor W = ten_new(temp_arena, ten_shape(input_size, num_classes));
    Tensor b = ten_new(temp_arena, ten_shape(num_classes));
    // Initialize weights and bias
    srand(42);
    ten_rand(W, -0.05f, 0.05f);
    ten_zero(b);

    // --- Training loop ---
    int epochs = 5;
    float lr = 0.01f;
    float best_acc = 0.0f;
    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0.0f;
        int correct = 0;
        for (int i = 0; i < actual_train_count; i++) {
            Tensor x = ten_index(temp_arena, X_train, i); // (784,)
            Tensor y = ten_index(temp_arena, y_train, i); // (10,)
            // Forward: logits = xW + b
            Tensor logits_tensor = ten_new(temp_arena, ten_shape(10));
            for (int j = 0; j < num_classes; j++) {
                for (int k = 0; k < input_size; k++) {
                    logits_tensor.data[j] += x.data[k] * W.data[k * num_classes + j];
                }
                logits_tensor.data[j] += b.data[j];
            }
            int label = ten_argmax(y);
            // Loss (cross-entropy) using Tensor-based softmax
            float logprob = ten_softmax_loss(logits_tensor, label);
            loss -= logprob;
            // Prediction using Tensor-based argmax
            int pred = ten_argmax(logits_tensor);
            if (pred == label) correct++;
            // Backward: gradient wrt logits using Tensor-based softmax gradients
            Tensor grad_logits_tensor = ten_softmax_gradients(temp_arena, logits_tensor, label);
            // Update weights and bias using Tensor gradients
            for (int j = 0; j < num_classes; j++) {
                for (int k = 0; k < input_size; k++) {
                    W.data[k * num_classes + j] -= lr * grad_logits_tensor.data[j] * x.data[k];
                }
                b.data[j] -= lr * grad_logits_tensor.data[j];
            }
        }
        float acc = (float)correct / actual_train_count;
        printf("Epoch %d: loss=%.4f, acc=%.4f\n", epoch+1, loss/actual_train_count, acc);
        // Evaluate on test set
        int test_correct = 0;
        for (int i = 0; i < test_count; i++) {
            Tensor x = ten_index(temp_arena, X_test, i);
            Tensor y = ten_index(temp_arena, y_test, i);
            Tensor logits_tensor = ten_new(temp_arena, ten_shape(10));
            for (int j = 0; j < num_classes; j++) {
                for (int k = 0; k < input_size; k++) {
                    logits_tensor.data[j] += x.data[k] * W.data[k * num_classes + j];
                }
                logits_tensor.data[j] += b.data[j];
            }
            int label = ten_argmax(y);
            int pred = ten_argmax(logits_tensor);
            if (pred == label) test_correct++;
        }
        float test_acc = (float)test_correct / test_count;
        printf("Test accuracy: %.4f\n", test_acc);
        if (test_acc > best_acc) best_acc = test_acc;
    }
    printf("Best test accuracy: %.4f\n", best_acc);
    arena_destroy(&flat_arena);
    return 0;
}