// test with clang perceptron.c && ./a.out and making sure the accuracy is above 90%

#include <stdio.h>
#include <time.h>
#include <math.h>

#include "base.c"
#include "ai.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// x -> W * x -> + b -> softmax -> loss

int main(int argc, char **argv) {
    temp_arena = arena_new(1 * Gb);

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

            train_labels = ten_new(train_arena, tenshape(item_count, 10));
            for(int i = 0; i < item_count; i++) {
                uint8_t label = stream_read_uint8(&labels_stream);
                ten_index(train_labels, i).data[label] = 1.0f;
            }
        }

        arena_reset(temp_arena);
        arena_reset(data_temp);

        {
            String inputs_data = file_load(data_temp, "mnist/train-images.idx3-ubyte");
            ByteStream stream = stream_string(data_temp, inputs_data);
            uint32_t magic = stream_read_uint32_bigendian(&stream);
            assert(magic == 0x00000803);
            uint32_t item_count = stream_read_uint32_bigendian(&stream);
            assert(item_count == train_labels.shape.dims[0]);
            uint32_t rows = stream_read_uint32_bigendian(&stream);
            assert(rows == 28);
            uint32_t cols = stream_read_uint32_bigendian(&stream);
            assert(cols == 28);

            train_inputs = ten_new(train_arena, tenshape(item_count, 28, 28));
            for(int i = 0; i < item_count; i++) {
                Tensor image = ten_index(train_inputs, i);
                for(int row = 0; row < 28; row++) {
                    for(int col = 0; col < 28; col++) {
                        ten_index(image, row).data[col] = (float)stream_read_uint8(&stream) / 255.0f;
                    }
                }
            }
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

    // clip some mnist training data
    train_inputs = ten_clip_upto(train_arena, train_inputs, (int)(0.5f * train_inputs.shape.dims[0]));

    // After MNIST loading, add perceptron implementation
    // --- Neural Network parameters ---
    int input_size = 28 * 28;
    int hidden_size = 128;  // Hidden layer size
    int num_classes = 10;
    int train_count = train_inputs.shape.dims[0];
    int test_count = train_count / 10; // 10% for test
    int actual_train_count = train_count - test_count;

    // Flatten train_inputs to (train_count, 784)
    Arena *flat_arena = arena_new(256 * Mb);
    Tensor flat_inputs = ten_new(flat_arena, tenshape(train_count, input_size));
    for (int i = 0; i < train_count; i++) {
        Tensor img = ten_index(train_inputs, i);
        Tensor flat_row = ten_index(flat_inputs, i);
        for (int row = 0; row < 28; row++) {
            Tensor img_row = ten_index(img, row);
            for (int col = 0; col < 28; col++) {
                flat_row.data[row * 28 + col] = img_row.data[col];
            }
        }
    }

    // Split into train and test sets
    Tensor X_train = flat_inputs;
    Tensor X_test = ten_new(temp_arena, tenshape(test_count, input_size));
    Tensor y_train = train_labels;
    Tensor y_test = ten_new(temp_arena, tenshape(test_count, num_classes));
    
    // Copy test data to separate tensors
    for (int i = 0; i < test_count; i++) {
        Tensor src_row = ten_index(flat_inputs, actual_train_count + i);
        Tensor dst_row = ten_index(X_test, i);
        for (int j = 0; j < input_size; j++) {
            dst_row.data[j] = src_row.data[j];
        }
        
        Tensor src_label = ten_index(train_labels, actual_train_count + i);
        Tensor dst_label = ten_index(y_test, i);
        for (int j = 0; j < num_classes; j++) {
            dst_label.data[j] = src_label.data[j];
        }
    }

    // --- Neural Network weights and biases ---
    // Layer 1: input -> hidden
    Tensor W1 = ten_new(temp_arena, tenshape(input_size, hidden_size));
    Tensor b1 = ten_new(temp_arena, tenshape(hidden_size));
    // Layer 2: hidden -> output
    Tensor W2 = ten_new(temp_arena, tenshape(hidden_size, num_classes));
    Tensor b2 = ten_new(temp_arena, tenshape(num_classes));
    
    // Initialize weights and biases
    srand(42);
    ten_xavier_init(W1, input_size);
    ten_zero(b1);
    ten_xavier_init(W2, hidden_size);
    ten_zero(b2);

    // --- Training loop with batching ---
    int epochs = 3;
    float lr = 0.001f;
    int batch_size = 32; // Batch size for training
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float loss = 0.0f;
        int correct = 0;
        
        // Process training data in batches
        for (int batch_start = 0; batch_start < actual_train_count; batch_start += batch_size) {
            int current_batch_size = (batch_start + batch_size <= actual_train_count) ? 
                                   batch_size : actual_train_count - batch_start;
            
            if(rand() % 100 == 0) {
                printf("Progress through epoch: %.1f%%\n", 100.0f * (float)batch_start / actual_train_count);
            }
            
            // Create batch tensors
            Tensor batch_inputs = ten_new(temp_arena, tenshape(current_batch_size, input_size));
            Tensor batch_labels = ten_new(temp_arena, tenshape(current_batch_size, num_classes));
            
            // Fill batch with data
            for (int b = 0; b < current_batch_size; b++) {
                int idx = batch_start + b;
                Tensor x = ten_index(X_train, idx);
                Tensor y = ten_index(y_train, idx);
                Tensor batch_x = ten_index(batch_inputs, b);
                Tensor batch_y = ten_index(batch_labels, b);
                
                // Copy input data
                for (int j = 0; j < input_size; j++) {
                    batch_x.data[j] = x.data[j];
                }
                
                // Copy label data
                for (int j = 0; j < num_classes; j++) {
                    batch_y.data[j] = y.data[j];
                }
            }
            
            // Forward pass - Layer 1: hidden = relu(batch_inputs * W1 + b1)
            Tensor batch_hidden = ten_matmul(temp_arena, batch_inputs, W1);
            ten_add_bias(batch_hidden, b1);
            ten_relu(batch_hidden);
            
            // Forward pass - Layer 2: logits = batch_hidden * W2 + b2
            Tensor batch_logits = ten_matmul(temp_arena, batch_hidden, W2);
            ten_add_bias(batch_logits, b2);
            
            // Compute loss and accuracy using existing tensor operations
            float batch_loss = 0.0f;
            int batch_correct = 0;
            for (int b = 0; b < current_batch_size; b++) {
                Tensor sample_logits = ten_index(batch_logits, b);
                Tensor sample_label = ten_index(batch_labels, b);
                int label = (int)ten_argmax(temp_arena, sample_label).data[0];
                
                float sample_loss = ten_softmax_loss(sample_logits, label);
                batch_loss += sample_loss;
                
                int pred = (int)ten_argmax(temp_arena, sample_logits).data[0];
                if (pred == label) batch_correct++;
            }
            loss += batch_loss;
            correct += batch_correct;
            
            // Backward pass - Gradient wrt logits
            Tensor grad_logits = ten_new(temp_arena, batch_logits.shape);
            for (int b = 0; b < current_batch_size; b++) {
                Tensor sample_logits = ten_index(batch_logits, b);
                Tensor sample_label = ten_index(batch_labels, b);
                int label = (int)ten_argmax(temp_arena, sample_label).data[0];
                
                Tensor sample_grad = ten_softmax_gradients(temp_arena, sample_logits, label);
                Tensor grad_slice = ten_index(grad_logits, b);
                
                for (int j = 0; j < num_classes; j++) {
                    grad_slice.data[j] = sample_grad.data[j];
                }
            }
            
            // Gradient wrt W2 and b2
            for (int j = 0; j < num_classes; j++) {
                for (int k = 0; k < hidden_size; k++) {
                    float grad_w2 = 0.0f;
                    for (int b = 0; b < current_batch_size; b++) {
                        grad_w2 += grad_logits.data[b * num_classes + j] * batch_hidden.data[b * hidden_size + k];
                    }
                    W2.data[k * num_classes + j] -= lr * grad_w2;
                }
                
                float grad_b2 = 0.0f;
                for (int b = 0; b < current_batch_size; b++) {
                    grad_b2 += grad_logits.data[b * num_classes + j];
                }
                b2.data[j] -= lr * grad_b2;
            }
            
            // Gradient wrt hidden layer (before ReLU)
            Tensor grad_hidden = ten_new(temp_arena, tenshape(current_batch_size, hidden_size));
            for (int b = 0; b < current_batch_size; b++) {
                for (int j = 0; j < hidden_size; j++) {
                    for (int k = 0; k < num_classes; k++) {
                        grad_hidden.data[b * hidden_size + j] += 
                            grad_logits.data[b * num_classes + k] * W2.data[j * num_classes + k];
                    }
                }
            }
            
            // Apply ReLU gradient to hidden layer gradients
            Tensor batch_hidden_before_relu = ten_matmul(temp_arena, batch_inputs, W1);
            ten_add_bias(batch_hidden_before_relu, b1);
            
            for (int b = 0; b < current_batch_size; b++) {
                Tensor hidden_slice = ten_index(batch_hidden_before_relu, b);
                Tensor grad_slice = ten_index(grad_hidden, b);
                
                for (int j = 0; j < hidden_size; j++) {
                    grad_slice.data[j] *= relu_gradient(hidden_slice.data[j]);
                }
            }
            
            // Gradient wrt W1 and b1
            for (int j = 0; j < hidden_size; j++) {
                for (int k = 0; k < input_size; k++) {
                    float grad_w1 = 0.0f;
                    for (int b = 0; b < current_batch_size; b++) {
                        grad_w1 += grad_hidden.data[b * hidden_size + j] * batch_inputs.data[b * input_size + k];
                    }
                    W1.data[k * hidden_size + j] -= lr * grad_w1;
                }
                
                float grad_b1 = 0.0f;
                for (int b = 0; b < current_batch_size; b++) {
                    grad_b1 += grad_hidden.data[b * hidden_size + j];
                }
                b1.data[j] -= lr * grad_b1;
            }
            arena_reset(temp_arena);
        }
        
        float acc = (float)correct / actual_train_count;
        printf("Epoch %d: loss=%.4f, acc=%.4f\n", epoch+1, loss/actual_train_count, acc);
        
        int test_correct = 0;
        {
            arena_reset(temp_arena);
            Tensor y_test_indices = ten_argmax(temp_arena, y_test);
            Tensor h1 = ten_matmul(temp_arena, X_test, W1);
            ten_add_bias(h1, b1);
            ten_relu(h1);
            Tensor h2 = ten_matmul(temp_arena, h1, W2);
            ten_add_bias(h2, b2);
            Tensor preds = ten_argmax(temp_arena, h2);
            Tensor eq = ten_equal(temp_arena, preds, y_test_indices);
            test_correct = (int)ten_sum(eq);
        }
        float test_acc = (float)test_correct / test_count;
        printf("Test accuracy: %.4f\n", test_acc);

        arena_reset(temp_arena);
    }
    arena_destroy(&flat_arena);
    return 0;
}