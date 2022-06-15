/**
 *
 *  Generalized Neural Network
 *
 *      by: Dr. Béla J. Szekeres
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Block and thread per block
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define TPB 1024

// Cuda kernels
#include "kernels.cuh"

/* Global variables */
unsigned long int seed;
unsigned long int thread_num;
float tol_gradnorm;
float tol_error_diff;
float tol_fixit;
unsigned long int maxiter_grad;
unsigned long int maxiter_fix;
float initdx;
unsigned long int sfreq;
char input_name[100];
char output_name[100];
char predict_name[100];
char acc_name[100];
unsigned long int data_num;
unsigned long int learn_num;
unsigned long int mini_batch_size;
unsigned long int neuron_num;
unsigned long int input_num;
unsigned long int output_num;
char graph_datas[100];
char logic_datas[100];
char fixwb_datas[100];
char lossfunction_type[100];
float alpha;
unsigned long int optimizer;
float grad_alpha;
float adam_alpha;
float adam_beta1;
float adam_beta2;
float adam_eps;
unsigned long int ff_optimization;
unsigned long int chunker;
float chunk_treshold;
unsigned long int loaddatas;
char load_backup[100];
char save_backup[100];
unsigned long int numgrad;
unsigned long int zero_optim_param;
float numgrad_eps;
float inf;
unsigned long int all_neighbour_num;
unsigned long int all_input_num;
unsigned long int max_bias_num;

/* Prototypes */
void read_parameters(char file_name[100]);
void read_data(float *datas, unsigned long int line_number, FILE *f_data, unsigned long int test);
void read_graph(char graph_file_name[100], char logic_file_name[100], char fixwb_file_name[100],
                unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                unsigned long int **graph_logic, unsigned long int **bias_logic, unsigned long int **parent_number_m,
                float **fix_weight, float **fix_bias,
                unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias);
unsigned long int rand_range_int(unsigned long int min, unsigned long int max);
float rand_range(float min, float max);

float act_fun(float x, unsigned long int chooser);
float act_fun_diff(float x, unsigned long int chooser);

unsigned long int imax(unsigned long int a, unsigned long int b);
float dmax(float a, float b);
float **allocate_dmatrix(unsigned long int row_num, unsigned long int *col_num);
unsigned long int **allocate_imatrix(unsigned long int row_num, unsigned long int *col_num);
void deallocate_dmatrix(float **m, unsigned long int row_num);
void deallocate_imatrix(unsigned long int **m, unsigned long int row_num);
void print_progress_bar(unsigned long int max_length, float rate);
float calc_error(float *neuron_value, float *target_vector, unsigned long int mini_batch_len);

void print_graph(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int *activation_type, unsigned long int *graph_n, unsigned long int *graph_i,
                 unsigned long int *graph_logic, unsigned long int *bias_logic,
                 float *fix_weight, float *fix_bias,
                 unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias);

void program_failure(char str[]);
float random_normal(float mean, float std_dev);
void softmax(float *input, unsigned long int input_len);

void copy_dmatrix(float **input_matrix, unsigned long int row_num, unsigned long int *col_num, float *output_matrix);
void copy_imatrix(unsigned long int **input_matrix, unsigned long int row_num, unsigned long int *col_num, unsigned long int *output_matrix);

void initialize_weights(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                        unsigned long int **parent_number, float **weight, float **bias);

float calc_gradient_mini_batch(float *datas, unsigned long int mini_batch_len,
                               unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                               unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                               unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                               unsigned long int *graph_p_ind_n_g,
                               float *weight_g, float *bias_g,
                               float *weight_grad_g, float *bias_grad_g,
                               float *iter_forward, float *iter_backward);

float calc_gradient_mini_batch_ff(float *datas, unsigned long int mini_batch_len,
                                  unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                                  unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                                  unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                                  unsigned long int *graph_p_ind_n_g,
                                  float *weight_g, float *bias_g,
                                  float *weight_grad_g, float *bias_grad_g,
                                  float *iter_forward, float *iter_backward,
                                  unsigned long int dist_max,
                                  unsigned long int *dist_g, unsigned long int *dist,
                                  unsigned long int *dist_input_g, unsigned long int *dist_input);

float calc_network_mini_batch(float *datas, unsigned long int mini_batch_len,
                              unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                              unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                              unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                              unsigned long int *graph_p_ind_n_g,
                              float *weight_trans_g, float *bias_g,
                              float *iter_forward);

float calc_network_mini_batch_ff(float *datas, unsigned long int mini_batch_len,
                                 unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                                 unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                                 unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                                 unsigned long int *graph_p_ind_n_g,
                                 float *weight_trans_g, float *bias_g,
                                 float *iter_forward,
                                 unsigned long int dist_max,
                                 unsigned long int *dist_g, unsigned long int *dist,
                                 unsigned long int *dist_input_g, unsigned long int *dist_input);

void make_predictions(float *datas, unsigned long int mini_batch_len,
                      unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                      unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                      unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                      unsigned long int *graph_p_ind_n_g,
                      float *weight_trans_g, float *bias_g,
                      float **predictions_mini_batch);

void make_predictions_ff(float *datas, unsigned long int mini_batch_len,
                         unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                         unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                         unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                         unsigned long int *graph_p_ind_n_g,
                         float *weight_trans_g, float *bias_g,
                         unsigned long int dist_max,
                         unsigned long int *dist_g, unsigned long int *dist,
                         unsigned long int *dist_input_g, unsigned long int *dist_input,
                         float **predictions_mini_batch);

void save_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float *mt_weight, float *mth_weight, float *vt_weight, float *vth_weight, float *ut_weight,
                      float *mt_bias, float *mth_bias, float *vt_bias, float *vth_bias, float *ut_bias,
                      float adam_beta1t, float adam_beta2t,
                      unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias);

void load_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float *mt_weight, float *mth_weight, float *vt_weight, float *vth_weight, float *ut_weight,
                      float *mt_bias, float *mth_bias, float *vt_bias, float *vth_bias, float *ut_bias,
                      float adam_beta1t, float adam_beta2t,
                      unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias);

int main()
{
    // Allocatables

    // Set inf
    inf = 1.e20;

    // Read input parameters
    read_parameters("./inputs/simulparams.dat");

    // Set random seed
    if (seed == 0)
    {
        srand(time(0));
    }
    else
    {
        srand(seed);
    }

    cudaError_t error;

    cudaEvent_t start, stop;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float **predictions_mini_batch;

    predictions_mini_batch = (float **)malloc(mini_batch_size * sizeof(float *));

    for (unsigned long int i = 0; i < mini_batch_size; i++)
    {
        predictions_mini_batch[i] = (float *)malloc(output_num * sizeof(float));
    }

    // Minibatch allocations
    unsigned long int mini_batch_num = learn_num / mini_batch_size;
    if (mini_batch_size * mini_batch_num != learn_num)
    {
        mini_batch_num++;
    }
    unsigned long int mini_batch_num_valid = (data_num - learn_num) / mini_batch_size;
    if (mini_batch_size * mini_batch_num_valid != data_num - learn_num)
    {
        mini_batch_num_valid++;
    }
    float *datas_mini_batch = (float *)malloc(mini_batch_size * (input_num + output_num) * sizeof(float));

    // Allocations for the graph
    unsigned long int *neighbour_number = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));
    unsigned long int *bias_number = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));
    unsigned long int *first_ind_neighbour = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));
    unsigned long int *first_ind_bias = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        first_ind_neighbour[neuron_id] = 0;
        first_ind_bias[neuron_id] = 0;
    }

    unsigned long int **activation_type_m = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    unsigned long int **graph_n_m = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    unsigned long int **graph_i_m = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    unsigned long int **graph_logic_m = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    unsigned long int **bias_logic_m = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    float **fix_weight_m = (float **)malloc(neuron_num * sizeof(float *));
    float **fix_bias_m = (float **)malloc(neuron_num * sizeof(float *));
    unsigned long int **parent_number_m = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));

    read_graph(graph_datas, logic_datas, fixwb_datas, neighbour_number, bias_number,
               activation_type_m, graph_n_m, graph_i_m, graph_logic_m, bias_logic_m, parent_number_m,
               fix_weight_m, fix_bias_m,
               first_ind_neighbour, first_ind_bias);

    unsigned long int *first_ind_parent = (unsigned long int *)calloc(all_input_num, sizeof(unsigned long int));
    unsigned long int parent_number_old = 0;
    unsigned long int ind = 0;

    // first parent indices in all_input_num sized vectors
    for (unsigned long int i = 0; i < neuron_num; i++)
    {
        for (unsigned long int j = 0; j < bias_number[i]; j++)
        {
            if (ind > 0)
            {
                first_ind_parent[ind] = first_ind_parent[ind - 1] + parent_number_old;
            }
            parent_number_old = parent_number_m[i][j];
            ind++;
        }
    }

    max_bias_num = 0;
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        if (bias_number[neuron_id] > max_bias_num)
        {
            max_bias_num = bias_number[neuron_id];
        }
    }

    unsigned long int *activation_type = (unsigned long int *)malloc(all_input_num * sizeof(unsigned long int));
    unsigned long int *graph_n = (unsigned long int *)malloc(all_neighbour_num * sizeof(unsigned long int));
    unsigned long int *graph_i = (unsigned long int *)malloc(all_neighbour_num * sizeof(unsigned long int));
    unsigned long int *graph_logic = (unsigned long int *)malloc(all_neighbour_num * sizeof(unsigned long int));
    unsigned long int *bias_logic = (unsigned long int *)malloc(all_input_num * sizeof(unsigned long int));
    float *fix_weight = (float *)malloc(all_neighbour_num * sizeof(float));
    float *fix_bias = (float *)malloc(all_input_num * sizeof(float));
    unsigned long int *parent_number = (unsigned long int *)malloc(all_input_num * sizeof(unsigned long int));

    // Flattening
    copy_imatrix(activation_type_m, neuron_num, bias_number, activation_type);
    copy_imatrix(graph_n_m, neuron_num, neighbour_number, graph_n);
    copy_imatrix(graph_i_m, neuron_num, neighbour_number, graph_i);
    copy_imatrix(graph_logic_m, neuron_num, neighbour_number, graph_logic);
    copy_imatrix(bias_logic_m, neuron_num, bias_number, bias_logic);
    copy_imatrix(parent_number_m, neuron_num, bias_number, parent_number);
    copy_dmatrix(fix_weight_m, neuron_num, neighbour_number, fix_weight);
    copy_dmatrix(fix_bias_m, neuron_num, bias_number, fix_bias);

    // Copying parent_number_m to parent_number??

    // Creating the reversed graph, the second array is for the neighbor indices (the order, not the index)
    unsigned long int **graph_p_m = (unsigned long int **)malloc(all_input_num * sizeof(unsigned long int *));
    unsigned long int **graph_p_ind_n_m = (unsigned long int **)malloc(all_input_num * sizeof(unsigned long int *));
    // Allocating it
    for (unsigned long int i = 0; i < all_input_num; i++)
    {
        graph_p_m[i] = (unsigned long int *)malloc(parent_number[i] * sizeof(unsigned long int));
        graph_p_ind_n_m[i] = (unsigned long int *)malloc(parent_number[i] * sizeof(unsigned long int));
    }
    // Upload it
    unsigned long int *graph_p_m_counter = (unsigned long int *)calloc(all_input_num, sizeof(unsigned long int));
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
        {
            unsigned long int neighbour_ind = graph_n_m[neuron_id][neighbour_counter];
            unsigned long int bias_ind = graph_i_m[neuron_id][neighbour_counter];
            unsigned long int input_ind = first_ind_bias[neighbour_ind] + bias_ind;
            graph_p_m[input_ind][graph_p_m_counter[input_ind]] = neuron_id; // we need a counter here
            graph_p_ind_n_m[input_ind][graph_p_m_counter[input_ind]] = neighbour_counter;
            graph_p_m_counter[input_ind]++;
        }
    }

    unsigned long int *graph_p = (unsigned long int *)malloc(all_neighbour_num * sizeof(unsigned long int));
    unsigned long int *graph_p_ind_n = (unsigned long int *)malloc(all_neighbour_num * sizeof(unsigned long int));
    // Flattening it
    copy_imatrix(graph_p_m, all_input_num, parent_number, graph_p);
    copy_imatrix(graph_p_ind_n_m, all_input_num, parent_number, graph_p_ind_n);

    // Initializing the network
    float **weight_m = (float **)malloc(neuron_num * sizeof(float *));
    float **bias_m = (float **)malloc(neuron_num * sizeof(float *));
    initialize_weights(neighbour_number, bias_number, activation_type_m, graph_n_m, graph_i_m, parent_number_m, weight_m, bias_m);

    float *weight = (float *)malloc(all_neighbour_num * sizeof(float));
    float *bias = (float *)malloc(all_input_num * sizeof(float));
    copy_dmatrix(weight_m, neuron_num, neighbour_number, weight);
    copy_dmatrix(bias_m, neuron_num, bias_number, bias);

    float *vt_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *vt_bias = (float *)calloc(all_input_num, sizeof(float));
    float *ut_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *ut_bias = (float *)calloc(all_input_num, sizeof(float));
    float *mt_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *mt_bias = (float *)calloc(all_input_num, sizeof(float));
    float *vth_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *vth_bias = (float *)calloc(all_input_num, sizeof(float));
    float *mth_weight = (float *)calloc(all_neighbour_num, sizeof(float));
    float *mth_bias = (float *)calloc(all_input_num, sizeof(float));
    float *weight_grad = (float *)calloc(all_neighbour_num, sizeof(float));
    float *bias_grad = (float *)calloc(all_input_num, sizeof(float));

    // Setting the fix weights and biases
    for (unsigned long int ind = 0; ind < all_neighbour_num; ind++)
    {
        if (graph_logic[ind] == 0)
        {
            weight[ind] = fix_weight[ind];
        }
    }
    for (unsigned long int ind = 0; ind < all_input_num; ind++)
    {
        if (bias_logic[ind] == 0)
        {
            bias[ind] = fix_bias[ind];
        }
    }

    // Loading the weights and biases
    float adam_beta1t = 1.0, adam_beta2t = 1.0;

    if (loaddatas == 1)
    {
        load_weight_bias(load_backup, weight, bias, neuron_num, neighbour_number, bias_number,
                         mt_weight, mth_weight, vt_weight, vth_weight, ut_weight,
                         mt_bias, mth_bias, vt_bias, vth_bias, ut_bias,
                         adam_beta1t, adam_beta2t, first_ind_neighbour, first_ind_bias);

        // Setting the fix weights and biases
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {

                if (graph_logic[startind + neighbour_ind] == 0)
                {
                    weight[startind + neighbour_ind] = fix_weight[startind + neighbour_ind];
                }
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {

                if (bias_logic[startind + bias_ind] == 0)
                {
                    bias[startind + bias_ind] = fix_bias[startind + bias_ind];
                }
            }
        }
    }

    if (zero_optim_param == 1)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                mt_weight[startind + neighbour_ind] = 0.0;
                mth_weight[startind + neighbour_ind] = 0.0;
                vt_weight[startind + neighbour_ind] = 0.0;
                vth_weight[startind + neighbour_ind] = 0.0;
                ut_weight[startind + neighbour_ind] = 0.0;
            }

            startind = first_ind_bias[neuron_id];
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                mt_bias[startind + bias_ind] = 0.0;
                mth_bias[startind + bias_ind] = 0.0;
                vt_bias[startind + bias_ind] = 0.0;
                vth_bias[startind + bias_ind] = 0.0;
                ut_bias[startind + bias_ind] = 0.0;
            }
        }
        adam_beta1t = 1.0;
        adam_beta2t = 1.0;
    }

    if ((loaddatas == 0) && (zero_optim_param == 0))
    {
        program_failure("Logical error: zero_optim_param or loaddatas\n");
    }

    //+++++++++++++++++++++++++//
    //                         //
    //      Input checking     //
    //                         //
    //+++++++++++++++++++++++++//
    if ((strcmp(lossfunction_type, "bce_multilabeling") != 0) && (strcmp(lossfunction_type, "multilabeling_crossentropy") != 0) && (strcmp(lossfunction_type, "multiclassification_crossentropy") != 0) && (strcmp(lossfunction_type, "sumsquared") != 0))
    {
        printf("The loss_function_type should be:\n - 'multilabeling_crossentropy' or \n - 'multiclassification_crossentropy' or \n - 'sumsquared'\n - 'bce_multilabeling'\n\n");
        program_failure("Wrong loss function");
    }

    if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
    {
        // Checking that the output activations are identity, and all outputs have just one input.
        unsigned long int output_logic = 0;
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            if (bias_number[neuron_id] != 1)
            {
                output_logic += 1;
            }
            if (neighbour_number[neuron_id] != 0)
            {
                output_logic += 1;
            }
            if (activation_type_m[neuron_id][0] != 0)
            {
                output_logic += 1;
            }
        }

        if (output_logic > 0)
        {
            program_failure("Wrong activation function type on the output!");
        }
    }

    //+++++++++++++++++++++++++//
    //                         //
    //      PERT method        //
    //                         //
    //+++++++++++++++++++++++++//

    unsigned long int check_cycle = 0;
    unsigned long int *dist = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));          // neuron distances for PERT method
    unsigned long int *dist_input = (unsigned long int *)malloc(all_input_num * sizeof(unsigned long int)); // input distances for PERT method
    unsigned long int *dist_extra = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));    // neuron distances for PERT method
    unsigned long int dist_max;
    if (ff_optimization > 0)
    {

        // Calculate the distances for PERT method
        for (unsigned long int i = 0; i < neuron_num; i++)
        {
            dist[i] = 0;
        }

        for (unsigned long int i = 0; i < neuron_num; i++)
        {
            for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
            {
                for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
                {
                    dist[graph_n_m[neuron_id][neighbour_ind]] = imax(dist[graph_n_m[neuron_id][neighbour_ind]], dist[neuron_id] + 1);
                }
            }
        }
        dist_max = 0;
        for (unsigned long int i = 0; i < neuron_num; i++)
        {
            if (dist[i] > dist_max)
            {
                dist_max = dist[i];
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_extra[neuron_id] = dist[neuron_id];
        }
        // Make one extra step to check whether is a cycle in the graph
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                dist_extra[graph_n_m[neuron_id][neighbour_ind]] = imax(dist[graph_n_m[neuron_id][neighbour_ind]], dist[neuron_id] + 1);
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            if (dist[neuron_id] != dist_extra[neuron_id])
            {
                check_cycle = 1;
            }
        }
    }

    if ((ff_optimization > 0) && (check_cycle == 1))
    {
        program_failure("Logical error: cycle in the graph\n");
    }

    unsigned long int *dist_number;            // count the neurons by distance
    unsigned long int *dist_number_temp;       // temporal vector to count the neurons by distance
    unsigned long int **dist_indices_m;        // neuron indices by distances
    unsigned long int *dist_indices;           // the same in one vector
    unsigned long int *first_ind_dist_indices; // pointer to the first elements in dist_indices

    // Count the neurons by distance values
    if ((ff_optimization > 0) && (check_cycle == 0))
    {
        dist_number = (unsigned long int *)malloc((dist_max + 1) * sizeof(unsigned long int));

        for (unsigned long int i = 0; i <= dist_max; i++)
        {
            dist_number[i] = 0;
        }
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_number[dist[neuron_id]]++;
        }

        dist_indices_m = allocate_imatrix(dist_max + 1, dist_number);

        // Create the list of the neuron indices by distance
        dist_number_temp = (unsigned long int *)malloc((dist_max + 1) * sizeof(unsigned long int));

        // Check whether is there any distance with 0 neuron
        unsigned long int dist_min = dist_max;

        for (unsigned long int i = 0; i < neuron_num; i++)
        {
            if (dist_number[i] < dist_min)
            {
                dist_min = dist_number[i];
            }
        }

        for (unsigned long int i = 0; i <= dist_max; i++)
        {
            dist_number_temp[i] = 0;
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_indices_m[dist[neuron_id]][dist_number_temp[dist[neuron_id]]] = neuron_id;
            dist_number_temp[dist[neuron_id]]++;
        }

        dist_indices = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int)); // Ez nem jo itt!!! ???
        first_ind_dist_indices = (unsigned long int *)malloc((dist_max + 1) * sizeof(unsigned long int));
        for (unsigned long int i = 0; i <= dist_max; i++)
        {
            first_ind_dist_indices[i] = 0;
        }

        // copying dist_indices_m ---> dist_indices, first_ind_dist_indices
        unsigned long int copy_ind = 0;
        for (unsigned long int i = 0; i <= dist_max; i++)
        {
            for (unsigned long int j = 0; j < dist_number[i]; j++)
            {
                dist_indices[copy_ind] = dist_indices_m[i][j];
                copy_ind++;
            }
            if (i > 0)
            {
                first_ind_dist_indices[i] = first_ind_dist_indices[i - 1] + dist_number[i - 1];
            }
        }

        // Creating distance values for the inputs
        for (unsigned long int i = 0; i < all_input_num; i++)
        {
            dist_input[i] = 0;
        }

        unsigned long int counter = 0;
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
            {
                dist_input[counter] = dist[neuron_id];
                counter++;
            }
        }
    }

    //+++++++++++++++++++++++++//
    //                         //
    // Creating output headers //
    //                         //
    //+++++++++++++++++++++++++//

    unsigned long int iter_grad = 0;

    float elapsed_time = 0.0;
    FILE *f = fopen(output_name, "a");
    if (f)
    {
        time_t mytime = time(NULL);
        char *time_str = ctime(&mytime);
        time_str[strlen(time_str) - 1] = '\0';
        fprintf(f, "*************************************************************************\n");
        fprintf(f, "|%10s |%10s |%10s |%10s |%10s |%10s | %s \n", "ITER", "TE", "VE", "IF", "IB", "ET", time_str);
        fprintf(f, "-------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    fclose(f);

    if ((strcmp(lossfunction_type, "bce_multilabeling") == 0) || (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0) || (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0))
    {
        FILE *f = fopen(acc_name, "a");
        if (f)
        {
            time_t mytime = time(NULL);
            char *time_str = ctime(&mytime);
            time_str[strlen(time_str) - 1] = '\0';
            fprintf(f, "*************************************************************************\n");
            fprintf(f, "|%10s |%10s |%10s |%10s |%10s |%10s | %s \n", "ITER", "TA", "VA", "TE", "VE", "ET", time_str);
            fprintf(f, "-------------------------------------------------------------------------\n");
        }
        else
        {
            program_failure("File write error: logile\n");
        }
        fclose(f);
    }

    //++++++++++++++++++++++++++++++++++++++++++++++//
    //                                              //
    //        Copying the network to the gpu        //
    //                                              //
    //++++++++++++++++++++++++++++++++++++++++++++++//

    unsigned long int *neighbour_number_g, *bias_number_g, *parent_number_g,
        *first_ind_neighbour_g, *first_ind_bias_g, *first_ind_parent_g,
        *activation_type_g, *graph_p_ind_n_g,
        *graph_n_g, *graph_i_g, *graph_p_g,
        *graph_logic_g, *bias_logic_g,
        *dist_indices_g, *dist_number_g, *first_ind_dist_indices_g,
        *dist_g, *dist_input_g;

    float *weight_g, *bias_g, *weight_grad_g, *bias_grad_g;
    float *mt_weight_g, *mt_bias_g, *vt_weight_g, *vt_bias_g;
    float *mth_weight_g, *mth_bias_g, *vth_weight_g, *vth_bias_g;
    float *ut_weight_g, *ut_bias_g;
    float *weight_trans_g;

    cudaMalloc((void **)&weight_trans_g, sizeof(float) * all_neighbour_num);

    cudaMalloc((void **)&neighbour_number_g, sizeof(unsigned long int) * neuron_num);
    cudaMalloc((void **)&bias_number_g, sizeof(unsigned long int) * neuron_num);
    cudaMalloc((void **)&parent_number_g, sizeof(unsigned long int) * all_input_num);
    cudaMalloc((void **)&first_ind_neighbour_g, sizeof(unsigned long int) * neuron_num);
    cudaMalloc((void **)&first_ind_bias_g, sizeof(unsigned long int) * neuron_num);
    cudaMalloc((void **)&first_ind_parent_g, sizeof(unsigned long int) * all_input_num);
    cudaMalloc((void **)&activation_type_g, sizeof(unsigned long int) * all_input_num);
    cudaMalloc((void **)&graph_n_g, sizeof(unsigned long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_i_g, sizeof(unsigned long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_p_g, sizeof(unsigned long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_p_ind_n_g, sizeof(unsigned long int) * all_neighbour_num);
    cudaMalloc((void **)&graph_logic_g, sizeof(unsigned long int) * all_neighbour_num);
    cudaMalloc((void **)&bias_logic_g, sizeof(unsigned long int) * all_input_num);

    cudaMalloc((void **)&dist_g, sizeof(unsigned long int) * neuron_num);
    cudaMalloc((void **)&dist_input_g, sizeof(unsigned long int) * all_input_num);

    cudaMalloc((void **)&weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&weight_grad_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&bias_grad_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&mt_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&mt_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&ut_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&ut_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&vt_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&vt_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&mth_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&mth_bias_g, sizeof(float) * all_input_num);
    cudaMalloc((void **)&vth_weight_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&vth_bias_g, sizeof(float) * all_input_num);

    cudaMemcpy(first_ind_neighbour_g, first_ind_neighbour, sizeof(unsigned long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(first_ind_bias_g, first_ind_bias, sizeof(unsigned long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(first_ind_parent_g, first_ind_parent, sizeof(unsigned long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(neighbour_number_g, neighbour_number, sizeof(unsigned long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_number_g, bias_number, sizeof(unsigned long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(parent_number_g, parent_number, sizeof(unsigned long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_p_g, graph_p, sizeof(unsigned long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_p_ind_n_g, graph_p_ind_n, sizeof(unsigned long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(activation_type_g, activation_type, sizeof(unsigned long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_n_g, graph_n, sizeof(unsigned long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_i_g, graph_i, sizeof(unsigned long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(graph_logic_g, graph_logic, sizeof(unsigned long int) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_logic_g, bias_logic, sizeof(unsigned long int) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dist_g, dist, sizeof(unsigned long int) * neuron_num, cudaMemcpyHostToDevice);
    cudaMemcpy(dist_input_g, dist_input, sizeof(unsigned long int) * all_input_num, cudaMemcpyHostToDevice);

    cudaMemcpy(weight_g, weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_g, bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_g, weight_grad, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_grad_g, bias_grad, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mt_weight_g, mt_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mt_bias_g, mt_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vt_weight_g, vt_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vt_bias_g, vt_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mth_weight_g, mth_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mth_bias_g, mth_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vth_weight_g, vth_weight, sizeof(float) * all_neighbour_num, cudaMemcpyHostToDevice);
    cudaMemcpy(vth_bias_g, vth_bias, sizeof(float) * all_input_num, cudaMemcpyHostToDevice);

    //+++++++++++++++++++++++++//
    //                         //
    //        Main loop        //
    //                         //
    //+++++++++++++++++++++++++//

    float iter_forward_temp = 0.0, iter_backward_temp = 0.0;
    float iter_forward = 0.0, iter_backward = 0.0;

    iter_grad = 0;

    while (iter_grad < maxiter_grad)
    {
        iter_grad++;

        cudaEventRecord(start, NULL);

        float elapsed_time_temp = 0.0;

        iter_backward = 0.0;
        iter_forward = 0.0;
        float error_temp_mean = 0.0;

        FILE *f_data = fopen(input_name, "r");

        //+++++++++++++++++++++++++//
        //                         //
        //         Training        //
        //                         //
        //+++++++++++++++++++++++++//

        for (unsigned long int mini_batch_id = 0; mini_batch_id < mini_batch_num; mini_batch_id++)
        {
            // Read a mini-batch from file
            unsigned long int mini_batch_len;
            unsigned long int mini_batch_si = mini_batch_id * mini_batch_size;
            unsigned long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
            if (mini_batch_ei > learn_num - 1)
            {
                mini_batch_ei = learn_num - 1;
            }
            mini_batch_len = mini_batch_ei - mini_batch_si + 1;
            read_data(datas_mini_batch, mini_batch_len, f_data, 0);

            // Calculating the gradient on the mini-batch

            float error_temp = 0.0;
            if (ff_optimization == 0)
            {
                error_temp = calc_gradient_mini_batch(datas_mini_batch, mini_batch_len,
                                                      neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                                      first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                                      activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                                      graph_p_g, graph_p_ind_n_g, weight_g, bias_g,
                                                      weight_grad_g, bias_grad_g, &iter_forward_temp, &iter_backward_temp);
            }
            else
            {
                error_temp = calc_gradient_mini_batch_ff(datas_mini_batch, mini_batch_len,
                                                         neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                                         first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                                         activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                                         graph_p_g, graph_p_ind_n_g, weight_g, bias_g,
                                                         weight_grad_g, bias_grad_g, &iter_forward_temp, &iter_backward_temp,
                                                         dist_max, dist_g, dist, dist_input_g, dist_input);
            }

            error_temp_mean += error_temp;
            iter_forward += iter_forward_temp;
            iter_backward += iter_backward_temp;

            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_time_temp, start, stop);
            elapsed_time_temp /= 1000;

            // Update the weights
            switch (optimizer)
            {
            case 1:
                update_weight_gd_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, vt_weight_g, vt_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, grad_alpha, adam_beta1);
                update_bias_gd_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, vt_weight_g, vt_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, grad_alpha, adam_beta1);
                break;
            case 2:
                adam_beta1t *= adam_beta1;
                adam_beta2t *= adam_beta2;
                update_weight_adam_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, vt_weight_g, vt_bias_g, mth_weight_g, mth_bias_g, vth_weight_g, vth_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha, adam_beta1, adam_beta2, adam_beta1t, adam_beta2t, adam_eps);
                update_bias_adam_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, vt_weight_g, vt_bias_g, mth_weight_g, mth_bias_g, vth_weight_g, vth_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha, adam_beta1, adam_beta2, adam_beta1t, adam_beta2t, adam_eps);
                break;
            case 3:
                adam_beta1t *= adam_beta1;
                adam_beta2t *= adam_beta2;
                update_weight_adamax_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, ut_weight_g, ut_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha, adam_beta1, adam_beta2, adam_beta1t, adam_beta2t, adam_eps);
                update_bias_adamax_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, mt_weight_g, mt_bias_g, ut_weight_g, ut_bias_g, graph_logic_g, bias_logic_g, all_input_num, all_neighbour_num, adam_alpha, adam_beta1, adam_beta2, adam_beta1t, adam_beta2t, adam_eps);
                break;
            }

            // Display the results

            for (unsigned long int i = 0; i < 10; i++)
            {
                printf("\b \b");
            }
            printf("\r");
            // printf("\n");
            if (mini_batch_id < mini_batch_num - 1)
            {
                print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                printf(" [%lu/%lu] TE: %.5f ET: %.1fs ETA: %.1fs", mini_batch_id + 1,
                       mini_batch_num, error_temp,
                       elapsed_time_temp, elapsed_time_temp * mini_batch_num / (mini_batch_id + 1) - elapsed_time_temp + 0.01);
            }
            else
            {
                print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                printf(" [%lu/%lu] TE: %.5f ET: %.1fs", mini_batch_id + 1,
                       mini_batch_num, error_temp,
                       elapsed_time_temp);
            }
            fflush(stdout);
        }
        iter_forward /= mini_batch_num;
        iter_backward /= mini_batch_num;
        elapsed_time += elapsed_time_temp;

        //+++++++++++++++++++++++++//
        //                         //
        //        Validation       //
        //                         //
        //+++++++++++++++++++++++++//
        cudaEventRecord(start, NULL);

        float error_learn = error_temp_mean / mini_batch_num;
        float error_valid = 0.0;

        // Transposing the weight matrix
        weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                    neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);
        for (unsigned long int mini_batch_id = 0; mini_batch_id < mini_batch_num_valid; mini_batch_id++)
        {
            unsigned long int mini_batch_len;

            unsigned long int mini_batch_si = mini_batch_id * mini_batch_size + learn_num;
            unsigned long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1 + learn_num;
            if (mini_batch_ei > data_num - 1)
            {
                mini_batch_ei = data_num - 1;
            }
            mini_batch_len = mini_batch_ei - mini_batch_si + 1;

            read_data(datas_mini_batch, mini_batch_len, f_data, 0);
            float error_valid_temp = 0.0;

            if (ff_optimization == 0)
            {
                error_valid_temp = calc_network_mini_batch(datas_mini_batch, mini_batch_len,
                                                           neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                                           first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                                           activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                                           graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                                           &iter_forward_temp);
            }
            else
            {
                error_valid_temp = calc_network_mini_batch_ff(datas_mini_batch, mini_batch_len,
                                                              neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                                              first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                                              activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                                              graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                                              &iter_forward_temp,
                                                              dist_max, dist_g, dist, dist_input_g, dist_input);
            }
            error_valid += error_valid_temp * mini_batch_len;
        }
        error_valid /= data_num - learn_num;

        // Display the progress
        float elapsed_time_temp_valid = 0.0;

        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time_temp_valid, start, stop);
        elapsed_time_temp_valid /= 1000;
        elapsed_time_temp += elapsed_time_temp_valid;
        elapsed_time += elapsed_time_temp_valid;

        printf(" | ");
        print_progress_bar(10, iter_grad / (float)maxiter_grad);
        printf(" %3lu% [%lu/%lu] TE: %.5f: VE: %.5f ET: %.1fs ETA: %.1fs", iter_grad * 100 / maxiter_grad, iter_grad,
               maxiter_grad, error_learn, error_valid, elapsed_time, elapsed_time * maxiter_grad / iter_grad - elapsed_time + 0.01);
        printf("\n");

        fclose(f_data);

        //+++++++++++++++++++++++++//
        //                         //
        //  Logging - evaluation   //
        //                         //
        //+++++++++++++++++++++++++//
        if (iter_grad % sfreq == 0)
        {
            f = fopen(output_name, "a");
            if (f)
            {
                fprintf(f, "|%10u |%10.5f |%10.5f |%10.1f |%10.1f |%10.1f | \n",
                        iter_grad, error_learn, error_valid, iter_forward, iter_backward, elapsed_time);
            }
            else
            {
                program_failure("File write error: logile\n");
            }
            fclose(f);

            f = fopen(predict_name, "w");
            // Make predictions and evaluations
            if (f)
            {
                float acc_learn = 0.0, acc_valid = 0.0;

                // Training set
                FILE *f_data = fopen(input_name, "r");
                for (unsigned long int mini_batch_id = 0; mini_batch_id < mini_batch_num; mini_batch_id++)
                {
                    unsigned long int mini_batch_len;

                    unsigned long int mini_batch_si = mini_batch_id * mini_batch_size;
                    unsigned long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
                    if (mini_batch_ei > learn_num - 1)
                    {
                        mini_batch_ei = learn_num - 1;
                    }
                    mini_batch_len = mini_batch_ei - mini_batch_si + 1;
                    read_data(datas_mini_batch, mini_batch_len, f_data, 0);

                    if (ff_optimization == 0)
                    {
                        make_predictions(datas_mini_batch, mini_batch_len,
                                         neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                         first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                         activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                         graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                         predictions_mini_batch);
                    }
                    else
                    {
                        make_predictions_ff(datas_mini_batch, mini_batch_len,
                                            neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                            first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                            activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                            graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                            dist_max,
                                            dist_g, dist, dist_input_g, dist_input,
                                            predictions_mini_batch);
                    }

                    for (unsigned long int i = 0; i < mini_batch_len; i++)
                    {
                        for (unsigned long int j = 0; j < output_num; j++)
                        {
                            fprintf(f, "%f ", predictions_mini_batch[i][j]);
                        }
                        fprintf(f, "\n");
                    }

                    if ((strcmp(lossfunction_type, "bce_multilabeling") == 0) || (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0))
                    {

                        for (unsigned long int i = 0; i < mini_batch_len; i++)
                        {
                            for (unsigned long int j = 0; j < output_num; j++)
                            {
                                acc_learn += (unsigned long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long int)roundf(datas_mini_batch[i * (input_num + output_num) + input_num + j] + 0.01);
                            }
                        }
                    }

                    if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
                    {
                        // Search the max ind in the predictions
                        for (unsigned long int i = 0; i < mini_batch_len; i++)
                        {
                            unsigned long int pred_ind = 0;
                            float pred_max = predictions_mini_batch[i][0];
                            for (unsigned long int j = 1; j < output_num; j++)
                            {
                                if (predictions_mini_batch[i][j] > pred_max)
                                {
                                    pred_ind = j;
                                    pred_max = predictions_mini_batch[i][j];
                                }
                            }

                            // Search the max ind in the true classes
                            unsigned long int true_ind = 0;
                            float true_max = datas_mini_batch[i * (input_num + output_num) + input_num + 0];
                            for (unsigned long int j = 1; j < output_num; j++)
                            {
                                if (datas_mini_batch[i * (input_num + output_num) + input_num + j] > true_max)
                                {
                                    true_ind = j;
                                    true_max = datas_mini_batch[i * (input_num + output_num) + input_num + j];
                                }
                            }

                            acc_learn += (float)pred_ind == true_ind;
                        }
                    }
                }

                // Validation set
                for (unsigned long int mini_batch_id = 0; mini_batch_id < mini_batch_num_valid; mini_batch_id++)
                {
                    unsigned long int mini_batch_len;

                    unsigned long int mini_batch_si = mini_batch_id * mini_batch_size + learn_num;
                    unsigned long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1 + learn_num;
                    if (mini_batch_ei > data_num - 1)
                    {
                        mini_batch_ei = data_num - 1;
                    }
                    mini_batch_len = mini_batch_ei - mini_batch_si + 1;

                    read_data(datas_mini_batch, mini_batch_len, f_data, 0);

                    if (ff_optimization == 0)
                    {
                        make_predictions(datas_mini_batch, mini_batch_len,
                                         neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                         first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                         activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                         graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                         predictions_mini_batch);
                    }
                    else
                    {
                        make_predictions_ff(datas_mini_batch, mini_batch_len,
                                            neighbour_number_g, neighbour_number, bias_number_g, bias_number, parent_number_g,
                                            first_ind_neighbour_g, first_ind_neighbour, first_ind_bias_g, first_ind_bias, first_ind_parent_g,
                                            activation_type_g, activation_type, graph_n_g, graph_n, graph_i_g, graph_i,
                                            graph_p_g, graph_p_ind_n_g, weight_trans_g, bias_g,
                                            dist_max,
                                            dist_g, dist, dist_input_g, dist_input,
                                            predictions_mini_batch);
                    }

                    for (unsigned long int i = 0; i < mini_batch_len; i++)
                    {
                        for (unsigned long int j = 0; j < output_num; j++)
                        {
                            fprintf(f, "%f ", predictions_mini_batch[i][j]);
                        }
                        fprintf(f, "\n");
                    }

                    if ((strcmp(lossfunction_type, "bce_multilabeling") == 0) || (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0))
                    {

                        for (unsigned long int i = 0; i < mini_batch_len; i++)
                        {
                            for (unsigned long int j = 0; j < output_num; j++)
                            {
                                acc_valid += (unsigned long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long int)roundf(datas_mini_batch[i * (input_num + output_num) + input_num + j] + 0.01);
                            }
                        }
                    }

                    if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
                    {
                        // Search the max ind in the predictions
                        for (unsigned long int i = 0; i < mini_batch_len; i++)
                        {
                            unsigned long int pred_ind = 0;
                            float pred_max = predictions_mini_batch[i][0];
                            for (unsigned long int j = 1; j < output_num; j++)
                            {
                                if (predictions_mini_batch[i][j] > pred_max)
                                {
                                    pred_ind = j;
                                    pred_max = predictions_mini_batch[i][j];
                                }
                            }

                            // Search the max ind in the true classes
                            unsigned long int true_ind = 0;
                            float true_max = datas_mini_batch[i * (input_num + output_num) + input_num + 0];
                            for (unsigned long int j = 1; j < output_num; j++)
                            {
                                if (datas_mini_batch[i * (input_num + output_num) + input_num + j] > true_max)
                                {
                                    true_ind = j;
                                    true_max = datas_mini_batch[i * (input_num + output_num) + input_num + j];
                                }
                            }

                            acc_valid += (float)pred_ind == true_ind;
                        }
                    }
                }
                fclose(f_data);

                if ((strcmp(lossfunction_type, "bce_multilabeling") == 0) || (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0) ||
                    (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0))
                {
                    acc_learn /= learn_num;
                    acc_valid /= data_num - learn_num;

                    FILE *f_acc = fopen(acc_name, "a");
                    if (f_acc)
                    {
                        fprintf(f_acc, "|%10d |%10.5f |%10.5f |%10.5f |%10.5f |%10.1f | \n",
                                iter_grad, acc_learn, acc_valid, error_learn, error_valid, elapsed_time_temp);
                    }
                    else
                    {
                        program_failure("File write error: logile\n");
                    }
                    fclose(f_acc);
                }
            }
            else
            {
                program_failure("Prediction error \n");
            }

            fclose(f);
        }

        //+++++++++++++++++++++++++//
        //                         //
        //          Saving         //
        //                         //
        //+++++++++++++++++++++++++//
        if (iter_grad > 0)
        {
            if (iter_grad % sfreq == 0)
            {
                cudaMemcpy(weight, weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(bias, bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(mt_weight, mt_weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(mt_bias, mt_bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(vt_weight, vt_weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(vt_bias, vt_bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(mth_weight, mth_weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(mth_bias, mth_bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(vth_weight, vth_weight_g, sizeof(float) * all_neighbour_num, cudaMemcpyDeviceToHost);
                cudaMemcpy(vth_bias, vth_bias_g, sizeof(float) * all_input_num, cudaMemcpyDeviceToHost);

                cudaThreadSynchronize();

                save_weight_bias(save_backup, weight, bias, neuron_num, neighbour_number, bias_number,
                                 mt_weight, mth_weight, vt_weight, vth_weight, ut_weight,
                                 mt_bias, mth_bias, vt_bias, vth_bias, ut_bias,
                                 adam_beta1t, adam_beta2t, first_ind_neighbour, first_ind_bias);
            }
        }
    }

    f = fopen(output_name, "a");
    if (f)
    {
        fprintf(f, "-------------------------------------------------------------------------\n");
    }
    else
    {
        program_failure("File write error: logile\n");
    }
    fclose(f);
    if ((strcmp(lossfunction_type, "bce_multilabeling") == 0) || (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0) || (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0))
    {
        f = fopen(acc_name, "a");
        if (f)
        {
            fprintf(f, "-------------------------------------------------------------------------\n");
        }
        else
        {
            program_failure("File write error: logile\n");
        }
    }
    cudaFree(weight_trans_g);

    cudaFree(neighbour_number_g);
    cudaFree(bias_number_g);
    cudaFree(parent_number_g);
    cudaFree(first_ind_neighbour_g);
    cudaFree(first_ind_bias_g);
    cudaFree(first_ind_parent_g);
    cudaFree(activation_type_g);
    cudaFree(graph_n_g);
    cudaFree(graph_i_g);
    cudaFree(graph_p_g);
    cudaFree(graph_p_ind_n_g);
    cudaFree(graph_logic_g);
    cudaFree(bias_logic_g);
    cudaFree(dist_g);
    cudaFree(dist_input_g);
    cudaFree(weight_g);
    cudaFree(bias_g);
    cudaFree(weight_grad_g);
    cudaFree(bias_grad_g);
    cudaFree(mt_weight_g);
    cudaFree(mt_bias_g);
    cudaFree(ut_weight_g);
    cudaFree(ut_bias_g);
    cudaFree(vt_weight_g);
    cudaFree(vt_bias_g);
    cudaFree(mth_weight_g);
    cudaFree(mth_bias_g);
    cudaFree(vth_weight_g);
    cudaFree(vth_bias_g);

    // Deallocations --- graph
    deallocate_dmatrix(fix_weight_m, neuron_num);
    deallocate_dmatrix(fix_bias_m, neuron_num);
    deallocate_imatrix(activation_type_m, neuron_num);
    deallocate_imatrix(graph_p_m, all_input_num);
    deallocate_imatrix(graph_p_ind_n_m, all_input_num);

    deallocate_imatrix(graph_n_m, neuron_num);
    deallocate_imatrix(graph_i_m, neuron_num);
    deallocate_imatrix(graph_logic_m, neuron_num);
    deallocate_imatrix(bias_logic_m, neuron_num);
    deallocate_imatrix(parent_number_m, neuron_num);
    free(neighbour_number);
    free(bias_number);
    free(first_ind_neighbour);
    free(first_ind_bias);
    free(first_ind_parent);
    free(activation_type);
    free(graph_n);
    free(graph_i);
    free(graph_p);
    free(graph_p_ind_n);
    free(graph_logic);
    free(bias_logic);
    free(parent_number);
    free(fix_weight);
    free(fix_bias);
    free(graph_p_m_counter);

    // Deallocations --- weights, biases, gradients and momentums
    deallocate_dmatrix(weight_m, neuron_num);
    deallocate_dmatrix(bias_m, neuron_num);
    free(weight);
    free(bias);
    free(weight_grad);
    free(bias_grad);
    free(vt_weight);
    free(vt_bias);
    free(ut_weight);
    free(ut_bias);
    free(vth_weight);
    free(vth_bias);
    free(mt_weight);
    free(mt_bias);
    free(mth_weight);
    free(mth_bias);

    for (unsigned long int i = 0; i < mini_batch_size; i++)
    {
        free(predictions_mini_batch[i]);
    }
    free(predictions_mini_batch);

    // Deallocations --- PERT
    free(dist);
    free(dist_input);
    free(dist_extra);
    if ((ff_optimization > 0) && (check_cycle == 0))
    {
        free(dist_number);
        free(dist_number_temp);
        deallocate_imatrix(dist_indices_m, dist_max + 1);
        free(dist_indices);
        free(first_ind_dist_indices);
    }

    free(datas_mini_batch);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}

void read_parameters(char file_name[100])
{
    /**
     * Read the global variables from `file_name`
     */
    char temp_string[30];
    FILE *f = fopen(file_name, "r");
    if (f)
    {
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &seed);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &thread_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &tol_fixit);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &maxiter_grad);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &maxiter_fix);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &initdx);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &sfreq);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", input_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", output_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", predict_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", acc_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &data_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &learn_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &mini_batch_size);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &neuron_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &input_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &output_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", graph_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", logic_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", fixwb_datas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &alpha);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", lossfunction_type);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &optimizer);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &grad_alpha);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_alpha);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_beta1);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_beta2);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &adam_eps);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &ff_optimization);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &chunker);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &chunk_treshold);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &loaddatas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", load_backup);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", save_backup);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &zero_optim_param);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%lu", &numgrad);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &numgrad_eps);
        fclose(f);
    }
    else
    {
        program_failure("File read error: simulparams.dat\n");
    }
}

void read_graph(char graph_file_name[100], char logic_file_name[100], char fixwb_file_name[100],
                unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                unsigned long int **graph_logic, unsigned long int **bias_logic, unsigned long int **parent_number,
                float **fix_weight, float **fix_bias,
                unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias)
{
    /**
     * Read graph data, allocate memory
     */
    char temp_string[30];

    FILE *f_graph = fopen(graph_file_name, "r");
    FILE *f_logic = fopen(logic_file_name, "r");
    FILE *f_fixwb = fopen(fixwb_file_name, "r");
    if (f_graph && f_logic && f_fixwb)
    {
        // Read the graph
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            fscanf(f_graph, "%lu", &neighbour_number[neuron_id]);
            fscanf(f_graph, "%s", temp_string);

            if (neuron_id > 0)
            {
                first_ind_neighbour[neuron_id] = first_ind_neighbour[neuron_id - 1] + neighbour_number[neuron_id - 1];
            }
            all_neighbour_num += neighbour_number[neuron_id];

            graph_n[neuron_id] = (unsigned long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long int));
            graph_i[neuron_id] = (unsigned long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long int));

            graph_logic[neuron_id] = (unsigned long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long int));
            fix_weight[neuron_id] = (float *)malloc(neighbour_number[neuron_id] * sizeof(float));

            for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
            {
                fscanf(f_graph, "%lu", &graph_n[neuron_id][i]);
                graph_n[neuron_id][i]--;
                fscanf(f_graph, "%lu", &graph_i[neuron_id][i]);
                graph_i[neuron_id][i]--;
                fscanf(f_graph, "%s", temp_string);
            }

            for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
            {
                fscanf(f_logic, "%lu", &graph_logic[neuron_id][i]);
                if (graph_logic[neuron_id][i] == 0)
                {
                    fscanf(f_fixwb, "%f", &fix_weight[neuron_id][i]);
                }
            }

            fscanf(f_graph, "%s", temp_string);
            fscanf(f_logic, "%s", temp_string);
            fscanf(f_fixwb, "%s", temp_string);

            fscanf(f_graph, "%lu", &bias_number[neuron_id]);
            fscanf(f_graph, "%s", temp_string);

            if (neuron_id > 0)
            {
                first_ind_bias[neuron_id] = first_ind_bias[neuron_id - 1] + bias_number[neuron_id - 1];
            }
            all_input_num += bias_number[neuron_id];

            activation_type[neuron_id] = (unsigned long int *)malloc(bias_number[neuron_id] * sizeof(unsigned long int));
            bias_logic[neuron_id] = (unsigned long int *)malloc(bias_number[neuron_id] * sizeof(unsigned long int));
            fix_bias[neuron_id] = (float *)malloc(bias_number[neuron_id] * sizeof(float));
            for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
            {
                fscanf(f_graph, "%lu", &activation_type[neuron_id][i]);
            }

            for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
            {
                fscanf(f_logic, "%lu", &bias_logic[neuron_id][i]);
                if (bias_logic[neuron_id][i] == 0)
                {
                    fscanf(f_fixwb, "%f", &fix_bias[neuron_id][i]);
                }
            }
        }

        // Calculate the numbers of the parents
        for (unsigned long int i = 0; i < neuron_num; i++)
        {
            parent_number[i] = (unsigned long int *)malloc((bias_number[i]) * sizeof(unsigned long int));
            for (unsigned long int j = 0; j < bias_number[i]; j++)
            {
                parent_number[i][j] = 0;
            }
        };
        for (unsigned long int i = 0; i < neuron_num; i++)
        {
            for (unsigned long int j = 0; j < neighbour_number[i]; j++)
            {
                parent_number[graph_n[i][j]][graph_i[i][j]]++;
            }
        }
    }
    else
    {
        program_failure("File read error in graph files!");
    }
    fclose(f_graph);
    fclose(f_logic);
    fclose(f_fixwb);
}

void program_failure(char str[])
{
    /**
     * Program failure
     */
    perror(str);
    exit(EXIT_FAILURE);
}

void read_data(float *datas, unsigned long int line_number, FILE *f_data, unsigned long int test)
{
    /**
     * Read the data
     */
    if (f_data)
    {
        unsigned long int output_num_temp = 0;
        if (test == 0)
        {
            output_num_temp = output_num;
        }

        for (unsigned long int i = 0; i < line_number; i++)
        {
            for (unsigned long int j = 0; j < input_num + output_num_temp; j++)
            {
                fscanf(f_data, "%f", &datas[i * (input_num + output_num) + j]);
            }
        }
    }
    else
    {
        program_failure("File read error in data file!");
    }
}

unsigned long int rand_range_int(unsigned long int min, unsigned long int max)
{
    /**
     * Generates a random integer between min and max
     */
    return rand() % (max - min + 1) + min;
}

float rand_range(float min, float max)
{
    /**
     * Generates a random float number between min and max
     */

    return min + (float)rand() / RAND_MAX * (max - min);
}

float act_fun(float x, unsigned long int chooser)
{
    /**
     * Calculate the activation function type `chooser` on the input `x`
     */
    switch (chooser)
    {
    case 0:
        return x;
        break;
    case 1:
        return 1.0 / (1.0 + exp(-x));
        break;
    case 2:
        return tanh(x);
        // if (x>0){
        //     return x/(1.0+x);
        // }
        // else
        //{
        //     return x/(1.0-x);
        // }
        break;
    case 3:
        if (x > 0)
        {
            return x;
        }
        else
        {
            return 0.1 * x;
        }
        break;
    case 4:
        return x / (1.0 + exp(-x));
        break;
    case 6:
        return 1.0 - x;
        break;
    case 7:
        return 1.0 / x;
        break;
    case 8:
        return cos(x);
        break;
    case 9:
        return atanf(x);
        break;
    default:
        return 0.0;
        break;
    }
}

float act_fun_diff(float x, unsigned long int chooser)
{
    /**
     * Calculate the derivative of the activation function type `chooser` on the input `x`
     */
    switch (chooser)
    {
    case 0:
        return 1.0;
        break;
    case 1:
        return act_fun(x, chooser) * (1.0 - act_fun(x, chooser));
        break;
    case 2:
        return 1.0 - tanh(x) * tanh(x);
        // if (x>0){
        //     return 1.0/((1.0+x)*(1.0+x));
        // }
        // else
        //{
        //     return 1.0/((1.0-x)*(1.0-x));
        // }
        break;
    case 3:
        if (x > 0)
        {
            return 1.0;
        }
        else
        {
            return 0.1;
        }
        break;
    case 4:
        return (1.0 + exp(-x) + x * exp(-x)) / pow(1.0 + exp(-x), 2.0);
        break;
    case 6:
        return -1.0;
        break;
    case 7:
        return -1.0 / pow(x, 2.0);
        break;
    case 8:
        return -sin(x);
        break;
    case 9:
        return 1.0 / (1.0 + x * x);
        break;
    default:
        return 0.0;
        break;
    }
}

float calc_error(float *neuron_value, float *target_vector, unsigned long int mini_batch_len)
{
    /*
    Calculating the error functions
    */

    if (strcmp(lossfunction_type, "sumsquared") == 0)
    {

        float returner = 0.0;
        for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                returner += pow((neuron_value[data_index * neuron_num + neuron_id] - target_vector[data_index * output_num + i]), 2);
            }
        }
        return returner;
    }
    if (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0)
    {
        float returner = 0.0;

        for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                if ((neuron_value[data_index * neuron_num + neuron_id] > 0.0) && (neuron_value[data_index * neuron_num + neuron_id] < 1.0))
                {
                    returner -=
                        target_vector[data_index * output_num + i] * log(neuron_value[data_index * neuron_num + neuron_id]) +
                        (1.0 - target_vector[data_index * output_num + i]) * log(1.0 - neuron_value[data_index * neuron_num + neuron_id]);
                }
            }
        }
        return returner;
    }

    if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
    {

        float returner = 0.0;
        for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
        {
            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                returner +=
                    (1.0 - target_vector[data_index * output_num + i]) * neuron_value[data_index * neuron_num + neuron_id] +
                    log(1.0 + exp(-neuron_value[data_index * neuron_num + neuron_id]));
            }
        }
        return returner;
    }

    if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
    {
        float returner = 0.0;
        for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
        {

            float *softmax_vec = (float *)malloc(output_num * sizeof(float));

            float sum_softmax = 0.0;
            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                softmax_vec[i] = neuron_value[data_index * neuron_num + neuron_id];
                sum_softmax += exp(softmax_vec[i]);
            }

            // softmax(softmax_vec, output_num);

            for (unsigned long int i = 0; i < output_num; i++)
            {
                returner -= target_vector[data_index * output_num + i] * (softmax_vec[i] - log(sum_softmax));
            }

            free(softmax_vec);
        }
        return returner;
    }
}

unsigned long int imax(unsigned long int a, unsigned long int b)
{
    /**
     * Returns max(a,b) --- integer
     */
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

void copy_dmatrix(float **input_matrix, unsigned long int row_num, unsigned long int *col_num, float *output_matrix)
{
    unsigned long int ind = 0;
    for (unsigned long int i = 0; i < row_num; i++)
    {
        for (unsigned long int j = 0; j < col_num[i]; j++)
        {
            output_matrix[ind] = input_matrix[i][j];
            ind++;
        }
    }
}
void copy_imatrix(unsigned long int **input_matrix, unsigned long int row_num, unsigned long int *col_num, unsigned long int *output_matrix)
{
    unsigned long int ind = 0;
    for (unsigned long int i = 0; i < row_num; i++)
    {
        for (unsigned long int j = 0; j < col_num[i]; j++)
        {
            output_matrix[ind] = input_matrix[i][j];
            ind++;
        }
    }
}

float **allocate_dmatrix(unsigned long int row_num, unsigned long int *col_num)
{
    float **returner = (float **)malloc(row_num * sizeof(float *));
    for (unsigned long int i = 0; i < row_num; i++)
    {
        returner[i] = (float *)malloc(col_num[i] * sizeof(float));
    }
    return returner;
}

unsigned long int **allocate_imatrix(unsigned long int row_num, unsigned long int *col_num)
{
    unsigned long int **returner = (unsigned long int **)malloc(row_num * sizeof(unsigned long int *));
    for (unsigned long int i = 0; i < row_num; i++)
    {
        returner[i] = (unsigned long int *)malloc(col_num[i] * sizeof(unsigned long int));
    }
    return returner;
}

void deallocate_dmatrix(float **m, unsigned long int row_num)
{
    for (unsigned long int i = 0; i < row_num; i++)
    {
        free(m[i]);
    }
    free(m);
}

void deallocate_imatrix(unsigned long int **m, unsigned long int row_num)
{
    for (unsigned long int i = 0; i < row_num; i++)
    {
        free(m[i]);
    }
    free(m);
}

void print_progress_bar(unsigned long int max_length, float rate)
{

    printf("[");
    unsigned long int act_length = round(max_length * rate);
    for (unsigned long int i = 0; i < act_length; i++)
    {
        printf("=");
    }
    printf(">");
    for (unsigned long int i = 0; i < max_length - act_length; i++)
    {
        printf(".");
    }
    printf("] ");
}

//===========================================================================
//=  Function to generate normally distributed random variable using the    =
//=  Box-Muller method                                                      =
//=    - Input: mean and standard deviation                                 =
//=    - Output: Returns with normally distributed random variable          =
//===========================================================================

float random_normal(float mean, float std_dev)
{
    float u, r, theta; // Variables for Box-Muller method
    float x;           // Normal(0, 1) rv
    float norm_rv;     // The adjusted normal rv

    // Generate u
    u = 0.0;
    while (u == 0.0)
        u = rand_range(0.0, 1.0);

    // Compute r
    r = sqrt(-2.0 * log(u));

    // Generate theta
    theta = 0.0;
    while (theta == 0.0)
        theta = 2.0 * M_PI * rand_range(0.0, 1.0);

    // Generate x value
    x = r * cos(theta);

    // Adjust x value for specified mean and variance
    norm_rv = (x * std_dev) + mean;

    // Return the normally distributed RV value
    return (norm_rv);
}

void softmax(float *input, unsigned long int input_len)
{
    //    assert (input != NULL);
    //    assert (input_len != 0);
    unsigned long int i;
    float m;
    /* Find maximum value from input array */
    m = input[0];
    for (i = 1; i < input_len; i++)
    {
        if (input[i] > m)
        {
            m = input[i];
        }
    }

    float sum = 0;
    for (i = 0; i < input_len; i++)
    {
        sum += exp(input[i] - m);
    }

    for (i = 0; i < input_len; i++)
    {
        input[i] = exp(input[i] - m - log(sum));
    }
}

void initialize_weights(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                        unsigned long int **parent_number, float **weight, float **bias)
{
    /**
     *  Initialize the weights and the biases
     */

    // Initialize the weights
    for (unsigned long int i = 0; i < neuron_num; i++)
    {
        weight[i] = (float *)malloc((neighbour_number[i]) * sizeof(float));
        for (unsigned long int j = 0; j < neighbour_number[i]; j++)
        {
            // weight[i][j] = rand_range(-initdx, initdx) / (float)(parent_number[graph_n[i][j]][graph_i[i][j]] + 1.0);
            weight[i][j] = rand_range(-initdx, initdx) / (float)(parent_number[graph_n[i][j]][graph_i[i][j]] + neighbour_number[i]);
            // weight[i][j] = random_normal(0.0, 1.0) * sqrt(initdx * 2.0 / (parent_number[graph_n[i][j]][graph_i[i][j]])+1.0);
            // weight[i][j] = random_normal(0.0, 1.0) * sqrt(initdx * 2.0 / (parent_number[graph_n[i][j]][graph_i[i][j]]+neighbour_number[i]));
        }
    };

    // Initialize the bias
    for (unsigned long int i = 0; i < neuron_num; i++)
    {
        bias[i] = (float *)malloc((bias_number[i]) * sizeof(float));
        for (unsigned long int j = 0; j < bias_number[i]; j++)
        {
            bias[i][j] = 0.0;
        }
    }
}

float calc_gradient_mini_batch(float *datas, unsigned long int mini_batch_len,
                               unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                               unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                               unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                               unsigned long int *graph_p_ind_n_g,
                               float *weight_g, float *bias_g,
                               float *weight_grad_g, float *bias_grad_g,
                               float *iter_forward, float *iter_backward)
{
    // Calculating the gradient on a mini-batch

    // Definitions
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;
    float iter_backward_temp = 0.0;
    unsigned long int nthreads;

    // Reset gradients
    set_zero_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num);
    set_zero_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num);

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    // Loop over the elements on the elements of the mini-batch

    float error_temp;
    unsigned long int iter_f, iter_b;
    float error_iter;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *input_value_orig = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_help_temp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_help = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp_temp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *weight_trans = (float *)calloc(all_neighbour_num, sizeof(float));

    unsigned long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    float *weight_grad_inp_g, *weight_grad_inp_temp_g, *weight_grad_inp_old_g, *weight_grad_help_g, *datas_g,
        *weight_trans_g, *input_value_g,
        *input_value_old_g, *input_value_orig_g, *neuron_value_g,
        *weight_grad_help_temp_g;

    cudaMalloc((void **)&weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_inp_temp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_inp_old_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_temp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&weight_trans_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_old_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_orig_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);

    // Transposing the weight matrix for calculating the network
    weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(input_value_orig_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);

    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    // Iteration number
    float *error_iter_g;
    cudaMalloc((void **)&error_iter_g, sizeof(float));
    float *error_iter_c = (float *)calloc(1, sizeof(float));

    iter_f = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//
    while (error_iter > tol_fixit && iter_f < maxiter_fix)
    {
        iter_f++;

        // Calculating the neuron values
        calc_neuron_mb_gpu<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                  bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len);

        cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
        cudaMemcpy(input_value_g, input_value_orig_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_network_mb_gpu<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                   parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                   weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num);
        // Adding bias
        add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

        // Calculating the error and L1-error
        // maxnormDiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        l1normdiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);

        cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        error_iter = error_iter_c[0];
    }

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the gradient //
    //                          //
    //++++++++++++++++++++++++++//

    calc_grad_help_0_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                                input_value_g, weight_grad_help_temp_g, activation_type_g, neuron_num, all_input_num, mini_batch_len);

    calc_grad_help_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                              input_value_g, weight_grad_help_g, weight_grad_help_temp_g, activation_type_g,
                                              neuron_num, all_input_num, mini_batch_len);

    cudaMemcpy(weight_grad_help, weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_value, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(input_value_old, input_value_old_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Output neurons and targets
        for (unsigned long int i = 0; i < output_num; i++)
        {
            output_value[data_index * output_num + i] = neuron_value[data_index * neuron_num + neuron_num - output_num + i];
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
        if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            float *output_value_temp = (float *)malloc(output_num * sizeof(float));
            for (unsigned long int i = 0; i < output_num; i++)
            {
                output_value_temp[i] = output_value[data_index * output_num + i];
            }
            softmax(output_value_temp, output_num);

            for (unsigned long int i = 0; i < output_num; i++)
            {
                output_value[data_index * output_num + i] = output_value_temp[i];
            }

            free(output_value_temp);
        }

        if ((strcmp(lossfunction_type, "multiclassification_crossentropy") == 0) || (strcmp(lossfunction_type, "sumsquared") == 0))
        {

            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                unsigned long int startind = first_ind_bias[neuron_id];
                for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp_old[data_index * all_input_num + startind + j] =
                        (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                        weight_grad_help[data_index * all_input_num + startind + j];
                }
            }
        }

        if (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0)
        {

            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                unsigned long int startind = first_ind_bias[neuron_id];
                for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    if ((output_value[data_index * output_num + i] > 0.0) && (output_value[data_index * output_num + i] < 1.0))
                    {
                        weight_grad_inp_old[data_index * all_input_num + startind + j] = (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                                                                                         weight_grad_help[data_index * all_input_num + startind + j] /
                                                                                         (output_value[data_index * output_num + i] * (1.0 - output_value[data_index * output_num + i]));
                    }
                }
            }
        }

        if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
        {

            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                unsigned long int startind = first_ind_bias[neuron_id];
                for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp_old[data_index * all_input_num + startind + j] =
                        act_fun(output_value[data_index * output_num + i], 1) - target_vector[data_index * output_num + i];
                }
            }
        }
    }

    cudaMemcpy(weight_grad_inp_temp_g, weight_grad_inp_old, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_inp_g, weight_grad_inp_old, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_help_g, weight_grad_help, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_old, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);

    iter_b = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    //     Back propagation     //
    //                          //
    //++++++++++++++++++++++++++//

    while (iter_b < maxiter_fix && error_iter > tol_fixit)
    {
        iter_b++;

        cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_temp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_gradient_mb_gpu<<<dimGrid, dimBlock>>>(weight_grad_inp_g, weight_grad_inp_temp_g, weight_grad_inp_old_g,
                                                    weight_grad_help_g, mini_batch_len, neighbour_number_g, bias_number_g,
                                                    first_ind_neighbour_g, first_ind_bias_g,
                                                    graph_n_g, graph_i_g, weight_g, neuron_num, all_input_num);

        // Max abs error and L1-error
        // maxnorm<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(weight_grad_inp_temp_g, all_input_num * mini_batch_len, error_iter_g);
        l1norm<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(weight_grad_inp_temp_g, all_input_num * mini_batch_len, error_iter_g);

        cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        error_iter = error_iter_c[0];
    }

    // Calculating the gradients
    calc_gradient_mb_sum_gpu_w<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, weight_grad_g, neuron_value_g, first_ind_neighbour_g, first_ind_bias_g, neighbour_number_g, graph_n_g, graph_i_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    calc_gradient_mb_sum_gpu_b<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, bias_grad_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num, mini_batch_len);

    // Regularization
    reg_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, alpha);

    // Clipping
    if (chunker == 1)
    {
        clipping_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, chunk_treshold);
        clipping_bias_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, chunk_treshold);
    }

    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(weight_grad_inp_g);
    cudaFree(weight_grad_inp_temp_g);
    cudaFree(weight_grad_inp_old_g);
    cudaFree(weight_grad_help_g);
    cudaFree(weight_grad_help_temp_g);
    cudaFree(datas_g);
    cudaFree(weight_trans_g);
    cudaFree(input_value_g);
    cudaFree(input_value_old_g);
    cudaFree(input_value_orig_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(input_value_old);
    free(input_value_orig);
    free(neuron_value);
    free(target_vector);
    free(weight_grad_help_temp);
    free(weight_grad_help);
    free(weight_grad_inp);
    free(weight_grad_inp_temp);
    free(weight_grad_inp_old);
    free(output_value);
    free(weight_trans);
    free(error_iter_c);

    cudaFree(error_iter_g);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;
    *iter_backward = iter_b;

    return error_mini_batch;
}

float calc_gradient_mini_batch_ff(float *datas, unsigned long int mini_batch_len,
                                  unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                                  unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                                  unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                                  unsigned long int *graph_p_ind_n_g,
                                  float *weight_g, float *bias_g,
                                  float *weight_grad_g, float *bias_grad_g,
                                  float *iter_forward, float *iter_backward,
                                  unsigned long int dist_max,
                                  unsigned long int *dist_g, unsigned long int *dist,
                                  unsigned long int *dist_input_g, unsigned long int *dist_input)
{
    // Calculating the gradient on a mini-batch

    // Definitions
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;
    float iter_backward_temp = 0.0;
    unsigned long int nthreads;

    // Reset gradients
    set_zero_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num);
    set_zero_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num);

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    unsigned long int iter_f, iter_b;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *weight_grad_help = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_help_temp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *weight_grad_inp = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *weight_trans = (float *)calloc(all_neighbour_num, sizeof(float));

    unsigned long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    float *weight_grad_inp_g, *weight_grad_inp_temp_g, *weight_grad_inp_old_g, *weight_grad_help_g, *datas_g,
        *weight_trans_g, *input_value_g,
        *neuron_value_g,
        *weight_grad_help_temp_g;

    cudaMalloc((void **)&weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_inp_old_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&weight_grad_help_temp_g, sizeof(float) * mini_batch_len * all_input_num);
    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&weight_trans_g, sizeof(float) * all_neighbour_num);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);

    // Transposing the weight matrix for calculating the network
    weight_transpose_gpu<<<(neuron_num + TPB - 1) / TPB, TPB>>>(first_ind_neighbour_g, first_ind_bias_g, first_ind_parent_g,
                                                                neighbour_number_g, bias_number_g, parent_number_g, graph_p_g, graph_p_ind_n_g, weight_g, weight_trans_g, neuron_num);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    // Adding bias
    add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_help_temp_g, weight_grad_help_temp, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);

    iter_f = 0;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//

    // The main loop
    unsigned long int iter_fix = 0;

    for (unsigned long int layer_id = 0; layer_id < dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_f++;
        calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                     bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                     layer_id, dist_g);

        calc_network_mb_gpu_ff<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                      parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                      weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num, layer_id + 1, dist_input_g);
    }
    calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                 bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                 dist_max, dist_g);

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the gradient //
    //                          //
    //++++++++++++++++++++++++++//

    calc_grad_help_0_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                                input_value_g, weight_grad_help_temp_g, activation_type_g, neuron_num, all_input_num, mini_batch_len);

    calc_grad_help_gpu<<<dimGrid, dimBlock>>>(first_ind_neighbour_g, first_ind_bias_g, bias_number_g,
                                              input_value_g, weight_grad_help_g, weight_grad_help_temp_g, activation_type_g,
                                              neuron_num, all_input_num, mini_batch_len);

    cudaMemcpy(weight_grad_help, weight_grad_help_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToHost);
    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Output neurons and targets
        for (unsigned long int i = 0; i < output_num; i++)
        {
            output_value[data_index * output_num + i] = neuron_value[data_index * neuron_num + neuron_num - output_num + i];
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
        if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
        {
            float *output_value_temp = (float *)malloc(output_num * sizeof(float));
            for (unsigned long int i = 0; i < output_num; i++)
            {
                output_value_temp[i] = output_value[data_index * output_num + i];
            }
            softmax(output_value_temp, output_num);

            for (unsigned long int i = 0; i < output_num; i++)
            {
                output_value[data_index * output_num + i] = output_value_temp[i];
            }

            free(output_value_temp);
        }

        if ((strcmp(lossfunction_type, "multiclassification_crossentropy") == 0) || (strcmp(lossfunction_type, "sumsquared") == 0))
        {

            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                unsigned long int startind = first_ind_bias[neuron_id];
                for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp[data_index * all_input_num + startind + j] =
                        (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                        weight_grad_help[data_index * all_input_num + startind + j];
                }
            }
        }

        if (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0)
        {

            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                unsigned long int startind = first_ind_bias[neuron_id];
                for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    if ((output_value[data_index * output_num + i] > 0.0) && (output_value[data_index * output_num + i] < 1.0))
                    {
                        weight_grad_inp[data_index * all_input_num + startind + j] = (output_value[data_index * output_num + i] - target_vector[data_index * output_num + i]) *
                                                                                     weight_grad_help[data_index * all_input_num + startind + j] /
                                                                                     (output_value[data_index * output_num + i] * (1.0 - output_value[data_index * output_num + i]));
                    }
                }
            }
        }

        if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
        {

            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                unsigned long int startind = first_ind_bias[neuron_id];
                for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
                {
                    weight_grad_inp[data_index * all_input_num + startind + j] =
                        act_fun(output_value[data_index * output_num + i], 1) - target_vector[data_index * output_num + i];
                }
            }
        }
    }

    cudaMemcpy(weight_grad_inp_g, weight_grad_inp, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_grad_help_g, weight_grad_help, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyHostToDevice);

    iter_b = 0;

    //++++++++++++++++++++++++++//
    //                          //
    //     Back propagation     //
    //                          //
    //++++++++++++++++++++++++++//

    for (unsigned long int layer_id = dist_max - 1; layer_id > 0; layer_id--)
    {
        iter_b++;

        cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        calc_gradient_mb_gpu_ff<<<dimGrid, dimBlock>>>(weight_grad_inp_g, weight_grad_inp_old_g,
                                                       weight_grad_help_g, mini_batch_len, neighbour_number_g, bias_number_g,
                                                       first_ind_neighbour_g, first_ind_bias_g,
                                                       graph_n_g, graph_i_g, weight_g, neuron_num, all_input_num,
                                                       layer_id, dist_g);
    }
    cudaMemcpy(weight_grad_inp_old_g, weight_grad_inp_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
    calc_gradient_mb_gpu_ff<<<dimGrid, dimBlock>>>(weight_grad_inp_g, weight_grad_inp_old_g,
                                                   weight_grad_help_g, mini_batch_len, neighbour_number_g, bias_number_g,
                                                   first_ind_neighbour_g, first_ind_bias_g,
                                                   graph_n_g, graph_i_g, weight_g, neuron_num, all_input_num,
                                                   0, dist_g);

    // Calculating the gradients
    calc_gradient_mb_sum_gpu_w<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, weight_grad_g, neuron_value_g, first_ind_neighbour_g, first_ind_bias_g, neighbour_number_g, graph_n_g, graph_i_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    calc_gradient_mb_sum_gpu_b<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_inp_g, bias_grad_g, neuron_num, all_input_num, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, all_neighbour_num, mini_batch_len);
    divide_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(bias_grad_g, all_input_num, mini_batch_len);

    // Regularization
    reg_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_g, bias_g, weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, alpha);

    // Clipping
    if (chunker == 1)
    {
        clipping_weight_gpu<<<(all_neighbour_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, chunk_treshold);
        clipping_bias_gpu<<<(all_input_num + TPB - 1) / TPB, TPB>>>(weight_grad_g, bias_grad_g, all_input_num, all_neighbour_num, chunk_treshold);
    }

    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(weight_grad_inp_g);
    cudaFree(weight_grad_inp_old_g);
    cudaFree(weight_grad_help_g);
    cudaFree(weight_grad_help_temp_g);
    cudaFree(datas_g);
    cudaFree(weight_trans_g);
    cudaFree(input_value_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(neuron_value);
    free(target_vector);
    free(weight_grad_help);
    free(weight_grad_help_temp);
    free(weight_grad_inp);
    free(output_value);
    free(weight_trans);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;
    *iter_backward = iter_b;

    return error_mini_batch;
}

float calc_network_mini_batch(float *datas, unsigned long int mini_batch_len,
                              unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                              unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                              unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                              unsigned long int *graph_p_ind_n_g,
                              float *weight_trans_g, float *bias_g,
                              float *iter_forward)
{
    // Definitions
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;

    unsigned long int nthreads;

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    // Loop over the elements on the elements of the mini-batch

    float error_temp;
    unsigned long int iter_f;
    float error_iter;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *input_value_orig = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(mini_batch_len * output_num, sizeof(float));

    float *datas_g,
        *input_value_g,
        *input_value_old_g, *input_value_orig_g, *neuron_value_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_old_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_orig_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);
    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);

    unsigned long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(input_value_orig_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    float *error_iter_g;
    unsigned long int *maxid_g;

    cudaMalloc((void **)&error_iter_g, sizeof(float));

    float *error_iter_c = (float *)calloc(1, sizeof(float));

    iter_f = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//
    while (error_iter > tol_fixit && iter_f < maxiter_fix)
    {
        iter_f++;

        // Calculating the neuron values
        calc_neuron_mb_gpu<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                  bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len);

        cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
        cudaMemcpy(input_value_g, input_value_orig_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_network_mb_gpu<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                   parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                   weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num);
        // Adding bias
        add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

        // Calculating the error
        // maxnormDiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        l1normdiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);

        cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        error_iter = error_iter_c[0];
    }

    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Targets
        for (unsigned long int i = 0; i < output_num; i++)
        {
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
    }

    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(datas_g);
    cudaFree(input_value_g);
    cudaFree(input_value_old_g);
    cudaFree(input_value_orig_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(input_value_old);
    free(input_value_orig);
    free(neuron_value);
    free(target_vector);
    free(output_value);
    free(error_iter_c);

    cudaFree(error_iter_g);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;

    return error_mini_batch;
}

float calc_network_mini_batch_ff(float *datas, unsigned long int mini_batch_len,
                                 unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                                 unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                                 unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                                 unsigned long int *graph_p_ind_n_g,
                                 float *weight_trans_g, float *bias_g,
                                 float *iter_forward,
                                 unsigned long int dist_max,
                                 unsigned long int *dist_g, unsigned long int *dist,
                                 unsigned long int *dist_input_g, unsigned long int *dist_input)
{

    unsigned long int iter_f = 0;
    float error_iter = inf, error_mini_batch = 0.0;

    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *datas_g, *neuron_value_g, *input_value_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);
    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    unsigned long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);
    // Adding bias
    add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

    // The main loop
    unsigned long int iter_fix = 0;

    for (unsigned long int layer_id = 0; layer_id < dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_fix++;
        calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                     bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                     layer_id, dist_g);

        calc_network_mb_gpu_ff<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                      parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                      weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num, layer_id + 1, dist_input_g);
    }
    calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                 bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                 dist_max, dist_g);

    // Targets
    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        for (unsigned long int i = 0; i < output_num; i++)
        {
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
    }

    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Calculate the error
    error_mini_batch = calc_error(neuron_value, target_vector, mini_batch_len);

    cudaFree(datas_g);
    cudaFree(neuron_value_g);
    cudaFree(input_value_g);

    free(neuron_value);
    free(target_vector);
    free(input_value);

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_f;

    return error_mini_batch;
}

void make_predictions_ff(float *datas, unsigned long int mini_batch_len,
                         unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                         unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                         unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                         unsigned long int *graph_p_ind_n_g,
                         float *weight_trans_g, float *bias_g,
                         unsigned long int dist_max,
                         unsigned long int *dist_g, unsigned long int *dist,
                         unsigned long int *dist_input_g, unsigned long int *dist_input,
                         float **predictions_mini_batch)
{
    unsigned long int iter_f = 0;

    for (unsigned long int i = 0; i < mini_batch_len; i++)
    {
        for (unsigned long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[i][j] = 0.0;
        }
    }

    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(output_num, sizeof(float));

    float *datas_g, *neuron_value_g, *input_value_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);
    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    unsigned long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);
    // Adding bias
    add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

    // The main loop
    unsigned long int iter_fix = 0;

    for (unsigned long int layer_id = 0; layer_id < dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_fix++;
        calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                     bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                     layer_id, dist_g);

        calc_network_mb_gpu_ff<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                      parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                      weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num, layer_id + 1, dist_input_g);
    }
    calc_neuron_mb_gpu_ff<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                 bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len,
                                                 dist_max, dist_g);

    // Targets
    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        for (unsigned long int i = 0; i < output_num; i++)
        {
            target_vector[data_index * output_num + i] = datas[data_index * (input_num + output_num) + input_num + i];
        }
    }

    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Copying the predictions to predictions_mini_batch
    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Calculate the output
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            output_value[i] = neuron_value[data_index * neuron_num + neuron_id];
        }

        if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
        {

            softmax(output_value, output_num);
        }

        if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
        {
            for (unsigned long int i = 0; i < output_num; i++)
            {

                output_value[i] = act_fun(output_value[i], 1);
            }
        }

        for (unsigned long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[data_index][j] = output_value[j];
        }
    }

    cudaFree(datas_g);
    cudaFree(neuron_value_g);
    cudaFree(input_value_g);

    free(neuron_value);
    free(target_vector);
    free(input_value);
    free(output_value);
}

void make_predictions(float *datas, unsigned long int mini_batch_len,
                      unsigned long int *neighbour_number_g, unsigned long int *neighbour_number, unsigned long int *bias_number_g, unsigned long int *bias_number, unsigned long int *parent_number_g,
                      unsigned long int *first_ind_neighbour_g, unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias_g, unsigned long int *first_ind_bias, unsigned long int *first_ind_parent_g,
                      unsigned long int *activation_type_g, unsigned long int *activation_type, unsigned long int *graph_n_g, unsigned long int *graph_n, unsigned long int *graph_i_g, unsigned long int *graph_i, unsigned long int *graph_p_g,
                      unsigned long int *graph_p_ind_n_g,
                      float *weight_trans_g, float *bias_g,
                      float **predictions_mini_batch)
{
    //
    // Creating predictions
    //

    for (unsigned long int i = 0; i < mini_batch_len; i++)
    {
        for (unsigned long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[i][j] = 0.0;
        }
    }

    float iter_forward_temp = 0.0;
    unsigned long int nthreads;

    //++++++++++++++++++++++++++//
    //                          //
    // Loop over the mini-batch //
    //                          //
    //++++++++++++++++++++++++++//

    float error_temp;
    unsigned long int iter_f;
    float error_iter;

    float *input_value = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *neuron_value = (float *)calloc(mini_batch_len * neuron_num, sizeof(float));
    float *target_vector = (float *)calloc(mini_batch_len * output_num, sizeof(float));
    float *input_value_old = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *input_value_orig = (float *)calloc(mini_batch_len * all_input_num, sizeof(float));
    float *output_value = (float *)calloc(output_num, sizeof(float));
    float *datas_g, *input_value_g,
        *input_value_old_g, *input_value_orig_g, *neuron_value_g, *neuron_value_temp_g;

    cudaMalloc((void **)&datas_g, sizeof(float) * mini_batch_len * (input_num + output_num));
    cudaMalloc((void **)&input_value_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_old_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&input_value_orig_g, sizeof(float) * all_input_num * mini_batch_len);
    cudaMalloc((void **)&neuron_value_g, sizeof(float) * neuron_num * mini_batch_len);

    cudaMemcpy(datas_g, datas, sizeof(float) * mini_batch_len * (input_num + output_num), cudaMemcpyHostToDevice);

    unsigned long int grid_rows = (mini_batch_len + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X; // mini_batch_len
    unsigned long int grid_cols = (all_input_num + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;  // neuron_num
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    cudaMemcpy(input_value_g, input_value, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyHostToDevice);
    // Copying the input data to input_value_g
    copy_input_gpu<<<dimGrid, dimBlock>>>(datas_g, input_value_g, first_ind_bias_g, mini_batch_len, all_input_num, neuron_num, input_num, output_num);

    cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(input_value_orig_g, input_value_g, sizeof(float) * all_input_num * mini_batch_len, cudaMemcpyDeviceToDevice);
    cudaMemcpy(neuron_value_g, neuron_value, sizeof(float) * neuron_num * mini_batch_len, cudaMemcpyHostToDevice);

    float *error_iter_g;

    cudaMalloc((void **)&error_iter_g, sizeof(float));

    float *error_iter_c = (float *)calloc(1, sizeof(float));

    iter_f = 0;
    error_iter = inf;

    //++++++++++++++++++++++++++//
    //                          //
    // Calculating the network  //
    //                          //
    //++++++++++++++++++++++++++//
    while (error_iter > tol_fixit && iter_f < maxiter_fix)
    {
        iter_f++;

        // Calculating the neuron values
        calc_neuron_mb_gpu<<<dimGrid, dimBlock>>>(bias_number_g, first_ind_bias_g, activation_type_g,
                                                  bias_g, neuron_value_g, input_value_g, neuron_num, all_input_num, mini_batch_len);

        cudaMemcpy(input_value_old_g, input_value_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);
        cudaMemcpy(input_value_g, input_value_orig_g, sizeof(float) * mini_batch_len * all_input_num, cudaMemcpyDeviceToDevice);

        // Main part
        calc_network_mb_gpu<<<dimGrid, dimBlock>>>(datas_g, mini_batch_len, bias_number_g,
                                                   parent_number_g, first_ind_bias_g, first_ind_parent_g, graph_p_g,
                                                   weight_trans_g, neuron_value_g, input_value_g, neuron_num, all_input_num);
        // Adding bias
        add_bias_bcast<<<dimGrid, dimBlock>>>(mini_batch_len, all_input_num, bias_g, input_value_g);

        // Calculating the error
        // maxnormDiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);
        l1normdiff<<<(all_input_num * mini_batch_len + TPB - 1) / TPB, TPB>>>(input_value_old_g, input_value_g, all_input_num * mini_batch_len, error_iter_g);

        cudaMemcpy(error_iter_c, error_iter_g, sizeof(float), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();

        error_iter = error_iter_c[0];
    }
    cudaMemcpy(neuron_value, neuron_value_g, sizeof(float) * mini_batch_len * neuron_num, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    // Copying the predictions to predictions_mini_batch
    for (unsigned long int data_index = 0; data_index < mini_batch_len; data_index++)
    {
        // Calculate the output

        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            output_value[i] = neuron_value[data_index * neuron_num + neuron_id];
        }

        if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
        {

            softmax(output_value, output_num);
        }

        if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
        {
            for (unsigned long int i = 0; i < output_num; i++)
            {

                output_value[i] = act_fun(output_value[i], 1);
            }
        }

        for (unsigned long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[data_index][j] = output_value[j];
        }
    }

    cudaFree(datas_g);
    cudaFree(input_value_g);
    cudaFree(input_value_old_g);
    cudaFree(input_value_orig_g);
    cudaFree(neuron_value_g);

    free(input_value);
    free(input_value_old);
    free(input_value_orig);
    free(neuron_value);
    free(target_vector);
    free(output_value);
    free(error_iter_c);

    cudaFree(error_iter_g);
}

void save_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float *mt_weight, float *mth_weight, float *vt_weight, float *vth_weight, float *ut_weight,
                      float *mt_bias, float *mth_bias, float *vt_bias, float *vth_bias, float *ut_bias,
                      float adam_beta1t, float adam_beta2t,
                      unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias)
{
    FILE *f = fopen(filename, "w");
    if (f)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {

                fprintf(f, "%f ", weight[startind + neighbour_ind]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                fprintf(f, "%f ", bias[startind + bias_ind]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");

        // Optimizer parameters (on the weights and on the biases)
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                float mt_temp = mt_weight[startind + neighbour_ind];
                float mth_temp = mth_weight[startind + neighbour_ind];
                float vt_temp = vt_weight[startind + neighbour_ind];
                float vth_temp = vth_weight[startind + neighbour_ind];
                float ut_temp = ut_weight[startind + neighbour_ind];
                fprintf(f, "%f %f %f %f %f ", mt_temp, mth_temp, vt_temp, vth_temp, ut_temp);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                float mt_temp = mt_bias[startind + bias_ind];
                float mth_temp = mth_bias[startind + bias_ind];
                float vt_temp = vt_bias[startind + bias_ind];
                float vth_temp = vth_bias[startind + bias_ind];
                float ut_temp = ut_bias[startind + bias_ind];
                fprintf(f, "%f %f %f %f %f ", mt_temp, mth_temp, vt_temp, vth_temp, ut_temp);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "%f %f ", adam_beta1t, adam_beta2t);
        fprintf(f, "\n");
    }
    else
    {
        program_failure("File write error in backup file!");
    }
    fclose(f);
}

void load_weight_bias(char filename[100], float *weight, float *bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float *mt_weight, float *mth_weight, float *vt_weight, float *vth_weight, float *ut_weight,
                      float *mt_bias, float *mth_bias, float *vt_bias, float *vth_bias, float *ut_bias,
                      float adam_beta1t, float adam_beta2t,
                      unsigned long int *first_ind_neighbour, unsigned long int *first_ind_bias)
{
    FILE *f = fopen(filename, "r");
    if (f)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                fscanf(f, "%f", &weight[startind + neighbour_ind]);
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                fscanf(f, "%f ", &bias[startind + bias_ind]);
            }
        }

        // Optimizer parameters (on the weights and on the biases)
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_neighbour[neuron_id];
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                float mt_temp;
                float mth_temp;
                float vt_temp;
                float vth_temp;
                float ut_temp;
                fscanf(f, "%f %f %f %f %f ", &mt_temp, &mth_temp, &vt_temp, &vth_temp, &ut_temp);
                mt_weight[startind + neighbour_ind] = mt_temp;
                mth_weight[startind + neighbour_ind] = mth_temp;
                vt_weight[startind + neighbour_ind] = vt_temp;
                vth_weight[startind + neighbour_ind] = vth_temp;
                ut_weight[startind + neighbour_ind] = ut_temp;
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            unsigned long int startind = first_ind_bias[neuron_id];
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                float mt_temp;
                float mth_temp;
                float vt_temp;
                float vth_temp;
                float ut_temp;
                fscanf(f, "%f %f %f %f %f ", &mt_temp, &mth_temp, &vt_temp, &vth_temp, &ut_temp);
                mt_bias[startind + bias_ind] = mt_temp;
                mth_bias[startind + bias_ind] = mth_temp;
                vt_bias[startind + bias_ind] = vt_temp;
                vth_bias[startind + bias_ind] = vth_temp;
                ut_bias[startind + bias_ind] = ut_temp;
            }
        }
        fscanf(f, "%f %f ", &adam_beta1t, &adam_beta2t);
    }
    else
    {
        program_failure("File read error in backup file!");
    }
    fclose(f);
}

float dmax(float a, float b)
{
    /**
     * Returns max(a,b) --- float
     */
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

