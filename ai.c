#pragma once

#include "base.c"

typedef struct TensorShape {
    uint64_t *dims;
    uint64_t rank; // how many dimensions the tensor has
} TensorShape;

TensorShape tenshape_new(uint64_t *sizes, uint64_t rank) {
    uint64_t *shape = (uint64_t *)arena_alloc(temp_arena, sizeof(uint64_t) * rank);
    memcpy(shape, sizes, sizeof(uint64_t) * rank);
    TensorShape result = {shape, rank};
    return result;
}

#define tenshape(first_dim, ...) tenshape_new((uint64_t[]){first_dim, __VA_ARGS__}, sizeof((uint64_t[]){first_dim, __VA_ARGS__}) / sizeof(uint64_t))

TensorShape tenshape_clone(Arena *arena, TensorShape shape) {
    TensorShape result = {
        .dims = (uint64_t*)arena_alloc(arena, sizeof(uint64_t) * shape.rank),
        .rank = shape.rank,
    };
    memcpy(result.dims, shape.dims, sizeof(uint64_t) * shape.rank);
    return result;
}

uint64_t tenshape_count(TensorShape shape) {
    uint64_t result = 1;
    for(uint64_t i = 0; i < shape.rank; i++) {
        result *= shape.dims[i];
    }
    return result;
}

typedef struct Tensor {
    float32_t *data;
    TensorShape shape;
} Tensor;


Tensor ten_new(Arena *arena, TensorShape shape) {
    float32_t *data = arena_alloc(arena, sizeof(float32_t) * tenshape_count(shape));
    TensorShape shape_clone = tenshape_clone(arena, shape);
    return (Tensor){data, shape_clone};
}

uint64_t ten_stride(Tensor tensor, uint64_t dim) {
    // for dims [3, 3, 4]
    // stride of dim 2 is 1
    // stride of dim 1 is 4
    // stride of dim 0 is 12

    assert(dim >= 0);
    assert(dim < tensor.shape.rank);

    if(tensor.shape.rank == 0) {
        return 1;
    }

    uint64_t result = 1;
    for(uint64_t i = tensor.shape.rank - 1; i > dim; i--) {
        result *= tensor.shape.dims[i];
    }
    return result;
}

// indexes into the first dimension of the tensor to create a "sub-tensor"
// with a temporarily allocated shape
Tensor ten_index(Tensor tensor, uint64_t index) {
    assert(tensor.shape.rank > 0);
    
    // make a shifted over shape
    TensorShape shape = tenshape_clone(temp_arena, tensor.shape);
    for(uint64_t i = 0; i < shape.rank - 1; i++) {
        shape.dims[i] = tensor.shape.dims[i + 1];
    }
    shape.rank -= 1;

    // find the start of the new data
    uint64_t stride = ten_stride(tensor, 0);    
    float32_t *data = tensor.data + index * stride;

    // sanity check
    assert(stride == tenshape_count(shape));

    // create the new tensor
    Tensor result = {data, shape};
    return result;
}

Tensor ten_index_from_array(Tensor tensor, Array address) {
    assert(address.count <= tensor.shape.rank);

    if(address.count == 0) {
        return tensor;
    }

    Tensor to_return = tensor;
    for(int i = 0; i < address.count; i++) {
        to_return = ten_index(to_return, *((int*)array_index(address, i)));
    }
    return to_return;
}

Tensor ten_reshape(Tensor tensor, TensorShape new_shape) {
    assert(tenshape_count(tensor.shape) == tenshape_count(new_shape));
    tensor.shape = new_shape;
    return tensor;
}

Tensor ten_clone(Arena *arena, Tensor tensor) {
    Tensor result = ten_new(arena, tensor.shape);
    memcpy(result.data, tensor.data, sizeof(float) * tenshape_count(tensor.shape));
    result.shape = tenshape_clone(arena, tensor.shape);
    return result;
}

// if you matmul tensor with (10,2,3) and (1, 3, 4) you get a (10,2,4) tensor
// all dimensions of '1' are telescoped (this is useful for batch matrix multiplication)
Tensor ten_matmul(Arena *arena, Tensor a, Tensor b) {
    assert(a.shape.rank == b.shape.rank);
    assert(a.shape.rank >= 2);

    assert(a.shape.dims[a.shape.rank - 1] == b.shape.dims[b.shape.rank - 2]);

    TensorShape result_shape = {0};
    result_shape.rank = a.shape.rank;
    result_shape.dims = arena_alloc(arena, sizeof(uint64_t) * result_shape.rank);

    // fill in the dims while respecting the telescoping
    for(uint64_t i = 0; i < result_shape.rank - 2; i++) {
        if(a.shape.dims[i] == 1) {
            result_shape.dims[i] = b.shape.dims[i];
        } else if(b.shape.dims[i] == 1) {
            result_shape.dims[i] = a.shape.dims[i];
        } else {
            assert(a.shape.dims[i] == b.shape.dims[i]);
            result_shape.dims[i] = a.shape.dims[i];
        }
    }
    result_shape.dims[result_shape.rank - 2] = a.shape.dims[a.shape.rank - 2];
    result_shape.dims[result_shape.rank - 1] = b.shape.dims[b.shape.rank - 1];

    Tensor result = ten_new(arena, result_shape);
    Array current_result_address = array_new(temp_arena, sizeof(int), result_shape.rank);
    for(int i = 0; i < result_shape.rank - 2; i++) {
        array_appendval(temp_arena, &current_result_address, int, 0);
    }

    while(true) {
        // index into the a and b tensors according to the telescoping rules and the current address
        Array a_address = array_clone(temp_arena, current_result_address);
        for(int dimi = 0; dimi < a_address.count; dimi++) {
            if(a.shape.dims[dimi] == 1) {
                *(int*)array_index(a_address, dimi) = 0;
            }
        }
        Array b_address = array_clone(temp_arena, current_result_address);
        for(int dimi = 0; dimi < b_address.count; dimi++) {
            if(b.shape.dims[dimi] == 1) {
                *(int*)array_index(b_address, dimi) = 0;
            }
        }

        Tensor a_mat = ten_index_from_array(a, a_address);
        Tensor b_mat = ten_index_from_array(b, b_address);
        assert(a_mat.shape.rank == 2);
        assert(b_mat.shape.rank == 2);
        assert(a_mat.shape.dims[1] == b_mat.shape.dims[0]);

        // do the matmul and set the result into the indexed tensor into the result tensor
        Tensor result_mat = ten_index_from_array(result, current_result_address);

        // Ensure the result_mat, a_mat, and b_mat have correct shapes
        int a_rows = a_mat.shape.dims[0];
        int a_cols = a_mat.shape.dims[1];
        int b_rows = b_mat.shape.dims[0];
        int b_cols = b_mat.shape.dims[1];
        int res_rows = result_mat.shape.dims[0];
        int res_cols = result_mat.shape.dims[1];

        assert(a_cols == b_rows);
        assert(a_rows == res_rows);
        assert(b_cols == res_cols);

        for(int row = 0; row < res_rows; row++) {
            for(int col = 0; col < res_cols; col++) {
                float32_t sum = 0.0f;
                for(int k = 0; k < a_cols; k++) {
                    float32_t a_val = a_mat.data[row * a_cols + k];
                    float32_t b_val = b_mat.data[k * b_cols + col];
                    sum += a_val * b_val;
                }
                result_mat.data[row * res_cols + col] = sum;
            }
        }


        // see if there's anything left to count up to in the telescoped dimensions, break if not
        if(current_result_address.count == 0) break;

        *(int*)array_index(current_result_address, current_result_address.count - 1) += 1;

        bool too_big = false;
        for(int i = current_result_address.count - 1; i >= 0; i--) {
            if(*(int*)array_index(current_result_address, i) == result.shape.dims[i]) {
                *(int*)array_index(current_result_address, i) = 0;
                if(i == 0) {
                    too_big = true;
                    break;
                }
                *(int*)array_index(current_result_address, i - 1) += 1;
            } else {
                break;
            }
        }
        if(too_big) break;
    }

    return result;
}

// supports tensor of shape (N) or (Batch, N)
Tensor ten_argmax(Arena *arena, Tensor input_tensor) {
    assert(input_tensor.shape.rank == 1 || input_tensor.shape.rank == 2);
    
    Tensor tensor = input_tensor;
    if(tensor.shape.rank == 1) {
        tensor = ten_reshape(tensor, tenshape(1, tensor.shape.dims[0]));
    }

    Tensor ret = ten_new(arena, tenshape(tensor.shape.dims[0], 1));

    for(int batch_i = 0; batch_i < tensor.shape.dims[0]; batch_i++) {
        Tensor to_argmax = ten_index(tensor, batch_i);
        int idx = 0;
        for (int i = 1; i < to_argmax.shape.dims[0]; i++) {
            if (to_argmax.data[i] > to_argmax.data[idx]) idx = i;
        }
        ten_index(ret, batch_i).data[0] = idx;
    }

    if(input_tensor.shape.rank == 1) {
        return ten_reshape(ret, tenshape_clone(arena, tenshape(1)));
    }
    return ret;
}

Tensor ten_softmax(Arena *arena, Tensor logits) {
    Tensor softmax = ten_new(arena, logits.shape);
    assert(logits.shape.rank == 1);
    float sum = 0.0f;
    for(int i = 0; i < logits.shape.dims[0]; i++) {
        softmax.data[i] = expf(logits.data[i]);
        sum += softmax.data[i];
    }
    for(int i = 0; i < logits.shape.dims[0]; i++) {
        softmax.data[i] = expf(logits.data[i]) / sum;
    }
    return softmax;
}

float ten_softmax_loss(Tensor logits, int correct_label) {
    float max = logits.data[0];
    for (int i = 1; i < logits.shape.dims[0]; i++) {
        if (logits.data[i] > max) max = logits.data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < logits.shape.dims[0]; i++) {
        sum += expf(logits.data[i] - max);
    }
    float log_sum = max + logf(sum);
    return logits.data[correct_label] - log_sum;
}

Tensor ten_softmax_gradients(Arena *arena, Tensor logits, int correct_label) {
    Tensor gradients = ten_new(arena, logits.shape);
    
    // Find max for numerical stability
    float maxl = logits.data[0];
    for (int j = 1; j < logits.shape.dims[0]; j++) {
        if (logits.data[j] > maxl) maxl = logits.data[j];
    }
    
    // Compute softmax probabilities
    float sum_exp = 0.0f;
    for (int j = 0; j < logits.shape.dims[0]; j++) {
        sum_exp += expf(logits.data[j] - maxl);
    }
    
    // Compute gradients: softmax - one_hot
    for (int j = 0; j < logits.shape.dims[0]; j++) {
        gradients.data[j] = expf(logits.data[j] - maxl) / sum_exp;
    }
    gradients.data[correct_label] -= 1.0f;
    
    return gradients;
}


// Initialize tensor with zeros
void ten_zero(Tensor tensor) {
    uint64_t size = tenshape_count(tensor.shape);
    for (uint64_t i = 0; i < size; i++) {
        tensor.data[i] = 0.0f;
    }
}

// Initialize tensor with random values from min to max
void ten_rand(Tensor tensor, float min, float max) {
    uint64_t size = tenshape_count(tensor.shape);
    for (uint64_t i = 0; i < size; i++) {
        tensor.data[i] = min + ((float)rand() / RAND_MAX) * (max - min);
    }
}

// ReLU activation function
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Apply ReLU activation to a tensor in-place
void ten_relu(Tensor tensor) {
    uint64_t size = tenshape_count(tensor.shape);
    for (uint64_t i = 0; i < size; i++) {
        tensor.data[i] = relu(tensor.data[i]);
    }
}

// ReLU gradient (derivative)
float relu_gradient(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Apply ReLU gradient to a tensor in-place
void ten_relu_gradient(Tensor tensor) {
    uint64_t size = tenshape_count(tensor.shape);
    for (uint64_t i = 0; i < size; i++) {
        tensor.data[i] = relu_gradient(tensor.data[i]);
    }
}

// Xavier/Glorot weight initialization for better training stability
void ten_xavier_init(Tensor tensor, int input_size) {
    float scale = sqrtf(2.0f / input_size);
    uint64_t size = tenshape_count(tensor.shape);
    for (uint64_t i = 0; i < size; i++) {
        tensor.data[i] = (2.0f * ((float)rand() / RAND_MAX) - 1.0f) * scale;
    }
}

Tensor ten_add(Arena *arena, Tensor a, Tensor b) {
    assert(a.shape.rank == b.shape.rank);
    assert(tenshape_count(a.shape) == tenshape_count(b.shape));
    Tensor result = ten_new(arena, a.shape);
    for(int i = 0; i < tenshape_count(a.shape); i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

void ten_add_bias(Tensor activations, Tensor bias) {
    // For 2D tensor: (batch_size, output_dim) + (output_dim,)
    // For 1D tensor: (output_dim,) + (output_dim,)
    uint64_t activations_size = tenshape_count(activations.shape);
    uint64_t bias_size = tenshape_count(bias.shape);

    if (activations.shape.rank == 2 && bias.shape.rank == 1) {
        // (batch_size, output_dim) + (output_dim,)
        int batch_size = activations.shape.dims[0];
        int output_dim = activations.shape.dims[1];
        assert(output_dim == bias.shape.dims[0]);

        for (int b = 0; b < batch_size; b++) {
            Tensor row = ten_index(activations, b);
            for (int j = 0; j < output_dim; j++) {
                row.data[j] += bias.data[j];
            }
        }
    } else if (activations.shape.rank == 1 && bias.shape.rank == 1) {
        // (output_dim,) + (output_dim,)
        assert(activations_size == bias_size);
        for (uint64_t i = 0; i < activations_size; i++) {
            activations.data[i] += bias.data[i];
        }
    } else {
        assert(false && "Unsupported tensor shapes for bias addition");
    }
}