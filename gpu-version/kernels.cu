#include "kernels.cuh"

//
// This file contains all the Cuda kernels which I used.
//

__global__ void calc_gradient_mb_sum_gpu(float *weight_grad_inp, float *weight_grad_temp,
                                         float *neuron_value,
                                         unsigned long int mini_batch_len,
                                         unsigned long int *neighbour_number,
                                         unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias,
                                         unsigned long int *graph_n, unsigned long int *graph_i,
                                         unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int all_neighbour_num)
{
    //
    // Collecting the gradients from the minibatch
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {

        unsigned long int startind_neighbour = first_ind_neighbour[neuron_id];
        for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
        {
            unsigned long int neighbour_ind_n = graph_n[startind_neighbour + neighbour_counter];
            unsigned long int neighbour_ind_i = graph_i[startind_neighbour + neighbour_counter];

            unsigned long int startind_neighbour_input = first_ind_bias[neighbour_ind_n];
            weight_grad_temp[data_index * all_neighbour_num + startind_neighbour + neighbour_counter] =
                weight_grad_inp[data_index * all_input_num + startind_neighbour_input + neighbour_ind_i] *
                neuron_value[data_index * neuron_num + neuron_id];
        }
    }
}

__global__ void calc_gradient_mb_sum_gpu_w(float *weight_grad_inp, float *weight_grad, float *neuron_value,
                                           unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias, unsigned long int *neighbour_number, unsigned long int *graph_n, unsigned long int *graph_i,
                                           unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int all_neighbour_num, unsigned long int mini_batch_len)
{
    //
    // Calculating the gradients for the weights from the minibatch
    //

    unsigned long int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id < neuron_num)
    {
        float temp_value = 0.0;
        unsigned long int startind_neighbour = first_ind_neighbour[neuron_id];
        for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
            {
                unsigned long int neighbour_ind_n = graph_n[startind_neighbour + neighbour_counter];
                unsigned long int neighbour_ind_i = graph_i[startind_neighbour + neighbour_counter];

                unsigned long int startind_neighbour_input = first_ind_bias[neighbour_ind_n];
                weight_grad[startind_neighbour + neighbour_counter] +=
                    weight_grad_inp[data_index * all_input_num + startind_neighbour_input + neighbour_ind_i] *
                    neuron_value[data_index * neuron_num + neuron_id];
            }
        }
    }
}

__global__ void calc_gradient_mb_sum_gpu_b(float *weight_grad_inp, float *bias_grad,
                                           unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int all_neighbour_num, unsigned long int mini_batch_len)
{
    //
    // Calculating the gradients for the bias from the minibatch
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_input_num)
    {
        bias_grad[ind] = 0.0;
        float temp_value = 0.0;
        for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            temp_value += weight_grad_inp[data_index * all_input_num + ind];
        }
        bias_grad[ind] = temp_value;
    }
}

__global__ void weight_transpose_gpu(unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent,
                                     unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int *parent_number,
                                     unsigned long int *graph_p, unsigned long int *graph_p_ind_n,
                                     float *weight, float *weight_trans,
                                     unsigned long int neuron_num)
{
    //
    // Transposing the weight matrix (for calculating the input values in the network)
    //

    unsigned long int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (neuron_id < neuron_num)
    {
        unsigned long int input_ind_1 = first_ind_bias[neuron_id];
        for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
        {
            unsigned long int input_ind_2 = input_ind_1 + bias_id;

            unsigned long int input_ind_p_start = first_ind_parent[input_ind_2];
            for (unsigned long int parent_ind = 0; parent_ind < parent_number[input_ind_2]; parent_ind++)
            {
                unsigned long int start_neuron_id = graph_p[input_ind_p_start + parent_ind];
                unsigned long int start_neuron_neighbour_counter = graph_p_ind_n[input_ind_p_start + parent_ind];

                float temp_value = weight[first_ind_neighbour[start_neuron_id] + start_neuron_neighbour_counter];

                unsigned long int input_ind = input_ind_p_start + parent_ind;

                weight_trans[input_ind] = temp_value;
            }
        }
    }
}

__global__ void add_bias_bcast(unsigned long int mini_batch_len, unsigned long int all_input_num,
                               float *bias, float *input_value)
{
    //
    // Adding bias to the inputs
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int input_ind = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && input_ind < all_input_num)
    {
        input_value[data_index * all_input_num + input_ind] += bias[input_ind];
    }
}

__global__ void calc_grad_help_0_gpu(unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias, unsigned long int *bias_number,
                                     float *input_value, float *weight_grad_help_temp, unsigned long int *activation_type,
                                     unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int mini_batch_len)
{
    //
    // Calculating the help vectors needed by the gradient calculations (f') --- 1st part
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {
        unsigned long int startind_input = first_ind_bias[neuron_id];
        unsigned long int startind_neighbour = first_ind_neighbour[neuron_id];
        for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                float input_val = input_value[data_index * all_input_num + startind + j];
                unsigned long int act_type = activation_type[startind + j];
                weight_grad_help_temp[data_index * all_input_num + startind + j] = act_fun_diff_gpu(input_val, act_type);
            }
        }
    }
}

__global__ void calc_grad_help_gpu(unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias, unsigned long int *bias_number,
                                   float *input_value, float *weight_grad_help, float *weight_grad_help_temp, unsigned long int *activation_type,
                                   unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int mini_batch_len)
{
    //
    // Calculating the help vectors needed by the gradient calculations (f') --- 2nd part
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {
        unsigned long int startind_input = first_ind_bias[neuron_id];
        unsigned long int startind_neighbour = first_ind_neighbour[neuron_id];
        for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                weight_grad_help[data_index * all_input_num + startind + j] = weight_grad_help_temp[data_index * all_input_num + startind + j];
                for (unsigned long int k = 0; k < bias_number[neuron_id]; k++)
                {
                    if (j != k)
                    {
                        weight_grad_help[data_index * all_input_num + startind + j] *= act_fun_gpu(input_value[data_index * all_input_num + startind + k], activation_type[startind + k]);
                    }
                }
            }
        }
    }
}

__global__ void calc_neuron_mb_gpu(unsigned long int *bias_number,
                                   unsigned long int *first_ind_bias, unsigned long int *activation_type, float *bias,
                                   float *neuron_value, float *input_value,
                                   unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int mini_batch_len)
{
    //
    // Calculating the activation values
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {
        float x = 1.0;
        unsigned long int startind_input = first_ind_bias[neuron_id];
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {

            float y = input_value[data_index * all_input_num + startind_input + j];
            unsigned long int act_type = activation_type[startind_input + j];
            float act_val = act_fun_gpu(y, act_type);
            x = x * act_val;
        }
        neuron_value[data_index * neuron_num + neuron_id] = x;
    }
}

__global__ void update_weight_gd_gpu(float *weight, float *bias,
                                     float *weight_grad, float *bias_grad,
                                     float *vt_weight, float *vt_bias,
                                     unsigned long int *graph_logic,
                                     unsigned long int *bias_logic,
                                     unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                     float grad_alpha, float adam_beta1)
{
    //
    // Update the weights by Gradient descent with momentum
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_neighbour_num)
    {

        vt_weight[ind] = adam_beta1 * vt_weight[ind] + (1.0 - adam_beta1) * weight_grad[ind];
        if (graph_logic[ind] == 1)
        {
            weight[ind] -= grad_alpha * vt_weight[ind];
        }
    }
}

__global__ void update_bias_gd_gpu(float *weight, float *bias,
                                   float *weight_grad, float *bias_grad,
                                   float *vt_weight, float *vt_bias,
                                   unsigned long int *graph_logic,
                                   unsigned long int *bias_logic,
                                   unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                   float grad_alpha, float adam_beta1)
{
    //
    // Update the bias by Gradient descent with momentum
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_input_num)
    {

        vt_bias[ind] = adam_beta1 * vt_bias[ind] + (1.0 - adam_beta1) * bias_grad[ind];
        if (bias_logic[ind] == 1)
        {
            bias[ind] -= grad_alpha * vt_bias[ind];
        }
    }
}

__global__ void update_weight_adam_gpu(float *weight, float *bias,
                                       float *weight_grad, float *bias_grad,
                                       float *mt_weight, float *mt_bias,
                                       float *vt_weight, float *vt_bias,
                                       float *mth_weight, float *mth_bias,
                                       float *vth_weight, float *vth_bias,
                                       unsigned long int *graph_logic,
                                       unsigned long int *bias_logic,
                                       unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                       float adam_alpha, float adam_beta1, float adam_beta2,
                                       float adam_beta1t, float adam_beta2t, float adam_eps)
{
    //
    // Update the weights by Adam
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_neighbour_num)
    {

        float adam_alpha_temp = adam_alpha * sqrt(1 - adam_beta2t) / (1 - adam_beta1t);

        mt_weight[ind] = adam_beta1 * mt_weight[ind] + (1.0 - adam_beta1) * weight_grad[ind];
        vt_weight[ind] = adam_beta2 * vt_weight[ind] + (1.0 - adam_beta2) * weight_grad[ind] * weight_grad[ind];
        // mth_weight[ind] = mt_weight[ind] / (1.0 - adam_beta1t);
        // vth_weight[ind] = vt_weight[ind] / (1.0 - adam_beta2t);
        if (graph_logic[ind] == 1)
        {
            weight[ind] -= adam_alpha_temp * mt_weight[ind] / (sqrt(fabs(vt_weight[ind])) + adam_eps);
        }
    }
}

__global__ void update_bias_adam_gpu(float *weight, float *bias,
                                     float *weight_grad, float *bias_grad,
                                     float *mt_weight, float *mt_bias,
                                     float *vt_weight, float *vt_bias,
                                     float *mth_weight, float *mth_bias,
                                     float *vth_weight, float *vth_bias,
                                     unsigned long int *graph_logic,
                                     unsigned long int *bias_logic,
                                     unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                     float adam_alpha, float adam_beta1, float adam_beta2,
                                     float adam_beta1t, float adam_beta2t, float adam_eps)
{
    //
    // Update the bias by Adam
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_input_num)
    {
        float adam_alpha_temp = adam_alpha * sqrt(1 - adam_beta2t) / (1 - adam_beta1t);

        mt_bias[ind] = adam_beta1 * mt_bias[ind] + (1.0 - adam_beta1) * bias_grad[ind];
        vt_bias[ind] = adam_beta2 * vt_bias[ind] + (1.0 - adam_beta2) * bias_grad[ind] * bias_grad[ind];
        // mth_bias[ind] = mt_bias[ind] / (1.0 - adam_beta1t);
        // vth_bias[ind] = vt_bias[ind] / (1.0 - adam_beta2t);
        if (bias_logic[ind] == 1)
        {
            bias[ind] -= adam_alpha_temp * mt_bias[ind] / (sqrt(fabs(vt_bias[ind])) + adam_eps);
        }
    }
}

__global__ void update_weight_adamax_gpu(float *weight, float *bias,
                                         float *weight_grad, float *bias_grad,
                                         float *mt_weight, float *mt_bias,
                                         float *ut_weight, float *ut_bias,
                                         unsigned long int *graph_logic,
                                         unsigned long int *bias_logic,
                                         unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                         float adam_alpha, float adam_beta1, float adam_beta2,
                                         float adam_beta1t, float adam_beta2t, float adam_eps)
{
    //
    // Update the weights by Adamax
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_neighbour_num)
    {

        mt_weight[ind] = adam_beta1 * mt_weight[ind] + (1.0 - adam_beta1) * weight_grad[ind];
        ut_weight[ind] = fmax(adam_beta2 * ut_weight[ind], fabs(weight_grad[ind]));

        if (graph_logic[ind] == 1)
        {
            weight[ind] -= adam_alpha / (1.0 - adam_beta1t) *
                           mt_weight[ind] / (ut_weight[ind] + adam_eps);
        }
    }
}

__global__ void update_bias_adamax_gpu(float *weight, float *bias,
                                       float *weight_grad, float *bias_grad,
                                       float *mt_weight, float *mt_bias,
                                       float *ut_weight, float *ut_bias,
                                       unsigned long int *graph_logic,
                                       unsigned long int *bias_logic,
                                       unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                       float adam_alpha, float adam_beta1, float adam_beta2,
                                       float adam_beta1t, float adam_beta2t, float adam_eps)
{
    //
    // Update the bias by Adamax
    //

    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < all_input_num)
    {

        mt_bias[ind] = adam_beta1 * mt_bias[ind] + (1.0 - adam_beta1) * bias_grad[ind];
        ut_bias[ind] = fmax(adam_beta2 * ut_bias[ind], fabs(bias_grad[ind]));

        if (bias_logic[ind] == 1)
        {
            bias[ind] -= adam_alpha / (1.0 - adam_beta1t) *
                         mt_bias[ind] / (ut_bias[ind] + adam_eps);
        }
    }
}

__global__ void copy_input_gpu(float *datas, float *input_value,
                               unsigned long int *first_ind_bias,
                               unsigned long int mini_batch_len,
                               unsigned long int all_input_num, unsigned long int neuron_num,
                               unsigned long int input_num, unsigned long int output_num)
{
    //
    // Copy the input data to the inputs of the input neurons. Wow, so many input words.
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {
        unsigned long int startind_input = first_ind_bias[neuron_id];

        if (neuron_id < input_num)
        {
            input_value[data_index * all_input_num + startind_input] = datas[data_index * (input_num + output_num) + neuron_id];
        }
    }
}

__global__ void set_zero_gpu(float *v, unsigned long int N)
{
    //
    // Setting all components of a vector to zero
    //
    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < N)
    {
        v[ind] = 0.0;
    }
}

__device__ float act_fun_gpu(float y, unsigned long int act_type)
{
    //
    // Activation functions
    //
    float act_val = 0.0;
    switch (act_type)
    {
    case 0:
        act_val = y;
        break;
    case 1:
        act_val = 1.0 / (1.0 + exp(-y));
        break;
    case 2:
        act_val = tanh(y);
        // if (y>0){
        //     act_val = y / (1.0+y);
        // }
        // else
        //{
        //     act_val = y / (1.0 -y);
        // }
        break;
    case 3:
        if (y > 0)
        {
            act_val = y;
        }
        else
        {
            act_val = 0.1 * y;
        }
        break;
    case 4:
        act_val = y / (1.0 + exp(-y));
        break;
    case 6:
        act_val = 1.0 - y;
        break;
    case 7:
        act_val = 1.0 / y;
        break;
    case 8:
        act_val = cos(y);
        break;
    case 9:
        act_val = atanf(y);

        break;
    default:
        act_val = 0.0;
        break;
    }
    return act_val;
}

__device__ float act_fun_diff_gpu(float x, unsigned long int chooser)
{
    /**
     * Calculate the derivative of the activation function type `chooser` on the input `x`
     */
    float value = 0.0;
    switch (chooser)
    {
    case 0:
        value = 1.0;
        break;
    case 1:
        value = act_fun_gpu(x, chooser) * (1.0 - act_fun_gpu(x, chooser));
        break;
    case 2:
        value = 1.0 - tanh(x) * tanh(x);
        // if (x>0){
        //     value = 1.0/((1.0+x)*(1.0+x));
        // }
        // else
        //{
        //     value = 1.0/((1.0-x)*(1.0-x));
        // }
        break;
    case 3:
        if (x > 0)
        {
            value = 1.0;
        }
        else
        {
            value = 0.1;
        }
        break;
    case 4:
        value = (1.0 + exp(-x) + x * exp(-x)) / pow(1.0 + exp(-x), 2.0);
        break;
    case 6:
        value = -1.0;
        break;
    case 7:
        value = -1.0 / pow(x, 2.0);
        break;
    case 8:
        value = -sinf(x);
        break;
    case 9:
        value = 1.0 / (1.0 + x * x);
        break;
    default:
        value = 0.0;
        break;
    }
    return value;
}

__device__ __forceinline__ float atomicMaxf(float *addr, float value)
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
{
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void maxnorm(const float *__restrict__ input, const unsigned long int size, float *maxOut)
// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-
{
    //
    // Calculating the max norm of a vector
    //
    __shared__ float sharedMax;

    if (0 == threadIdx.x)
    {
        sharedMax = 0.0;
    }

    __syncthreads();

    float localMax = 0.0;

    for (unsigned long int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];

        if (localMax < fabs(val))
        {
            localMax = fabs(val);
        }
    }

    atomicMaxf(&sharedMax, localMax);

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
    }
}

__global__ void maxnormDiff(const float *__restrict__ input1, const float *__restrict__ input2, const unsigned long int size, float *maxOut)
{
    //
    // Calculating max norm for the difference between two vectors
    //
    __shared__ float sharedMax;

    if (0 == threadIdx.x)
    {
        sharedMax = 0.0;
    }

    __syncthreads();

    float localMax = 0.0;

    for (unsigned long int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = fabs(input1[i] - input2[i]);

        if (localMax < fabs(val))
        {
            localMax = fabs(val);
        }
    }

    atomicMaxf(&sharedMax, localMax);

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
    }
}

__global__ void calc_gradient_mb_gpu(float *weight_grad_inp, float *weight_grad_inp_temp, float *weight_grad_inp_old,
                                     float *weight_grad_help,
                                     unsigned long int mini_batch_len,
                                     unsigned long int *neighbour_number, unsigned long int *bias_number,
                                     unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias,
                                     unsigned long int *graph_n, unsigned long int *graph_i,
                                     float *weight,
                                     unsigned long int neuron_num, unsigned long int all_input_num)
{
    //
    // Calculating the gradients in the network for the cyclic case --- one step of the gradient back propagation
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {

        unsigned long int startind_input = first_ind_bias[neuron_id];
        unsigned long int startind_neighbour = first_ind_neighbour[neuron_id];

        float temp_value = 0.0;
        for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
        {
            unsigned long int neighbour_ind_n = graph_n[startind_neighbour + neighbour_counter];
            unsigned long int neighbour_ind_i = graph_i[startind_neighbour + neighbour_counter];

            unsigned long int startind_neighbour_input = first_ind_bias[neighbour_ind_n];
            temp_value = temp_value +
                         weight[startind_neighbour + neighbour_counter] *
                             weight_grad_inp_old[data_index * all_input_num + startind_neighbour_input + neighbour_ind_i];
        }
        for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
        {
            weight_grad_inp_temp[data_index * all_input_num + startind_input + bias_id] = temp_value * weight_grad_help[data_index * all_input_num + startind_input + bias_id];

            weight_grad_inp[data_index * all_input_num + startind_input + bias_id] +=
                temp_value *
                weight_grad_help[data_index * all_input_num + startind_input + bias_id];
        }
    }
}

__global__ void calc_network_mb_gpu(float *datas, unsigned long int mini_batch_len,
                                    unsigned long int *bias_number, unsigned long int *parent_number,
                                    unsigned long int *first_ind_bias, unsigned long int *first_ind_parent,
                                    unsigned long int *graph_p,
                                    float *weight_trans, float *neuron_value, float *input_value,
                                    unsigned long int neuron_num, unsigned long int all_input_num)
{

    //
    // Calculating the input values in the network (cyclic case)
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int input_ind = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && input_ind < all_input_num)
    {

        unsigned long int start_parent_ind = first_ind_parent[input_ind];

        float temp_value = 0.0;
        for (unsigned long int parent_id = 0; parent_id < parent_number[input_ind]; parent_id++)
        {
            unsigned long int neuron_id_from = graph_p[start_parent_ind + parent_id];

            float neuron_value_temp =
                neuron_value[data_index * neuron_num + neuron_id_from];

            temp_value +=
                weight_trans[start_parent_ind + parent_id] * neuron_value_temp;
        }
        input_value[data_index * all_input_num + input_ind] += temp_value;
    }
}

__global__ void calc_network_mb_gpu_ff(float *datas, unsigned long int mini_batch_len,
                                       unsigned long int *bias_number, unsigned long int *parent_number,
                                       unsigned long int *first_ind_bias, unsigned long int *first_ind_parent,
                                       unsigned long int *graph_p,
                                       float *weight_trans, float *neuron_value, float *input_value,
                                       unsigned long int neuron_num, unsigned long int all_input_num,
                                       unsigned long int act_dist, unsigned long int *dist_input)
{

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int input_ind = blockIdx.y * blockDim.y + threadIdx.y;

    //
    // Calculating the input values in the network for the acyclic case (only the `act_dist` layer)
    //

    if (data_index < mini_batch_len && input_ind < all_input_num)
    {
        if (dist_input[input_ind] == act_dist)
        {
            unsigned long int start_parent_ind = first_ind_parent[input_ind];

            float temp_value = 0.0;
            for (unsigned long int parent_id = 0; parent_id < parent_number[input_ind]; parent_id++)
            {
                unsigned long int neuron_id_from = graph_p[start_parent_ind + parent_id];

                float neuron_value_temp =
                    neuron_value[data_index * neuron_num + neuron_id_from];

                temp_value +=
                    weight_trans[start_parent_ind + parent_id] * neuron_value_temp;
            }
            input_value[data_index * all_input_num + input_ind] += temp_value;
        }
    }
}

__global__ void calc_neuron_mb_gpu_ff(unsigned long int *bias_number,
                                      unsigned long int *first_ind_bias, unsigned long int *activation_type, float *bias,
                                      float *neuron_value, float *input_value,
                                      unsigned long int neuron_num, unsigned long int all_input_num, unsigned long int mini_batch_len,
                                      unsigned long int act_dist, unsigned long int *dist)
{
    //
    // Calculating the activation values for the acyclic case (only the `act_dist` layer)
    //

    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {
        if (dist[neuron_id] == act_dist)
        {
            float x = 1.0;
            unsigned long int startind_input = first_ind_bias[neuron_id];
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {

                float y = input_value[data_index * all_input_num + startind_input + j];
                unsigned long int act_type = activation_type[startind_input + j];
                float act_val = act_fun_gpu(y, act_type);
                x = x * act_val;
            }
            neuron_value[data_index * neuron_num + neuron_id] = x;
        }
    }
}

__global__ void calc_gradient_mb_gpu_ff(float *weight_grad_inp, float *weight_grad_inp_old,
                                        float *weight_grad_help,
                                        unsigned long int mini_batch_len,
                                        unsigned long int *neighbour_number, unsigned long int *bias_number,
                                        unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias,
                                        unsigned long int *graph_n, unsigned long int *graph_i,
                                        float *weight,
                                        unsigned long int neuron_num, unsigned long int all_input_num,
                                        unsigned long int act_layer, unsigned long int *dist)
{
    //
    // Gradient calculation for the acyclic case (for back propagation)
    //
    unsigned long int data_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

    if (data_index < mini_batch_len && neuron_id < neuron_num)
    {
        if (act_layer == dist[neuron_id])
        {
            unsigned long int startind_input = first_ind_bias[neuron_id];
            unsigned long int startind_neighbour = first_ind_neighbour[neuron_id];

            float temp_value = 0.0;
            for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
            {
                unsigned long int neighbour_ind_n = graph_n[startind_neighbour + neighbour_counter];
                unsigned long int neighbour_ind_i = graph_i[startind_neighbour + neighbour_counter];

                unsigned long int startind_neighbour_input = first_ind_bias[neighbour_ind_n];
                temp_value = temp_value +
                             weight[startind_neighbour + neighbour_counter] *
                                 weight_grad_inp_old[data_index * all_input_num + startind_neighbour_input + neighbour_ind_i];
            }
            for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
            {
                weight_grad_inp[data_index * all_input_num + startind_input + bias_id] +=
                    temp_value *
                    weight_grad_help[data_index * all_input_num + startind_input + bias_id];
            }
        }
    }
}

__global__ void divide_gpu(float *v, unsigned long int N, unsigned long int number)
{
    //
    // Dividing a vector by a number
    //
    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < N)
    {
        v[ind] /= number;
    }
}

__global__ void l1norm(const float *__restrict__ input, const unsigned long int size, float *sumOut)
{
    //
    // L1-norm with reduction
    //
    __shared__ float sharedsum;

    if (0 == threadIdx.x)
    {
        sharedsum = 0.0;
    }

    __syncthreads();

    float localsum = 0.0;

    for (unsigned long int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = fabs(input[i]);

        localsum += val;
    }

    for (unsigned long int i = 0; i < blockDim.x; i++)
    {
        sharedsum += localsum;
    }

    my_atomicAdd(&sharedsum, localsum);

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *sumOut = sharedsum / size;
    }
}

__global__ void l1normdiff(const float *__restrict__ input1, const float *__restrict__ input2, const unsigned long int size, float *sumOut)
// https://stackoverflow.com/questions/52772680/is-there-any-way-to-reduce-sum-100m-float-elements-of-an-array-in-cuda
{
    //
    // L1-norm for the difference between two vectors with reduction
    //
    __shared__ float sharedsum;

    if (0 == threadIdx.x)
    {
        sharedsum = 0.0;
    }

    __syncthreads();

    float localsum = 0.0;

    for (unsigned long int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = fabs(input1[i] - input2[i]);

        localsum += val;
    }

    for (unsigned long int i = 0; i < blockDim.x; i++)
    {
        sharedsum += localsum;
    }

    my_atomicAdd(&sharedsum, localsum);

    __syncthreads();

    if (0 == threadIdx.x)
    {
        *sumOut = sharedsum / size;
    }
}

__device__ float my_atomicAdd(float *addr, float val)
{
    return atomicAdd(addr, val);
}

__global__ void reg_weight_gpu(float *weight, float *bias,
                               float *weight_grad, float *bias_grad,
                               unsigned long int all_input_num, unsigned long int all_neighbour_num,
                               float alpha)
{
    //
    // L2-regularization for the weights
    //
    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < all_neighbour_num)
    {
        weight_grad[ind] += 2.0 * alpha * weight[ind];
    }
}

__global__ void reg_bias_gpu(float *weight, float *bias,
                             float *weight_grad, float *bias_grad,
                             unsigned long int all_input_num, unsigned long int all_neighbour_num,
                             float alpha)
{
    //
    // L2-regularization for bias (we won't use it)
    //
    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < all_input_num)
    {
        bias_grad[ind] += 2.0 * alpha * bias[ind];
    }
}

__global__ void clipping_weight_gpu(float *weight_grad, float *bias_grad,
                                    unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                    float threshold)
{
    //
    // Gradient clipping for weights
    //
    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < all_neighbour_num)
    {
        if (weight_grad[ind] > threshold)
        {
            weight_grad[ind] = threshold;
        }
        if (weight_grad[ind] < -threshold)
        {
            weight_grad[ind] = -threshold;
        }
    }
}

__global__ void clipping_bias_gpu(float *weight_grad, float *bias_grad,
                                    unsigned long int all_input_num, unsigned long int all_neighbour_num,
                                    float threshold)
{
    //
    // Gradient clipping for bias
    //
    unsigned long int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind < all_input_num)
    {
        if (bias_grad[ind] > threshold)
        {
            bias_grad[ind] = threshold;
        }
        if (bias_grad[ind] < -threshold)
        {
            bias_grad[ind] = -threshold;
        }
    }
}