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
#include <omp.h>

#define M_PI acos(-1.0)

/* Global variables */
unsigned long int seed;
unsigned long int thread_num;
float tol_fixit;
unsigned  int maxiter_grad;
unsigned  int maxiter_fix;
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

/* Prototypes */
void read_parameters(char file_name[100]);
void read_data(float **datas, unsigned long int line_number, FILE *f_data, unsigned long int test);
void read_graph(char graph_file_name[100], char logic_file_name[100], char fixwb_file_name[100],
                unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                unsigned long int **graph_logic, unsigned long int **bias_logic,
                float **fix_weight, float **fix_bias);
int rand_range_int(int min, int max);
float rand_range(float min, float max);
void initialize_weights(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                        unsigned long int **parent_number, float **weight, float **bias);
float act_fun(float x, unsigned long int chooser);
float act_fun_diff(float x, unsigned long int chooser);
unsigned long int calc_network_one_sample(unsigned long int *neighbour_number, unsigned long int *bias_number,
                            unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                            unsigned long int **parent_number, float **weight, float **bias,
                            float **input_value, float *neuron_value);
unsigned long int calc_network_one_sample_ff(unsigned long int *neighbour_number, unsigned long int *bias_number,
                               unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                               unsigned long int **parent_number, float **weight, float **bias,
                               float **input_value, float *neuron_value,
                               unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices);
unsigned long int calc_gradient_one_sample(unsigned long int *neighbour_number, unsigned long int *bias_number,
                             unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                             unsigned long int **parent_number, float **weight, float **bias,
                             float **input_value, float *neuron_value,
                             float **weight_grad, float **bias_grad, float *target_vector);
unsigned long int calc_gradient_one_sample_ff(unsigned long int *neighbour_number, unsigned long int *bias_number,
                                unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                                unsigned long int **parent_number, float **weight, float **bias,
                                float **input_value, float *neuron_value,
                                float **weight_grad, float **bias_grad, float *target_vector,
                                unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices);

float calc_diff_matrices(float **m1, float **m2, unsigned long int row_nums, unsigned long int *col_nums);

float calc_gradient_mini_batch(float **datas, unsigned long int mini_batch_len,
                               unsigned long int *neighbour_number, unsigned long int *bias_number,
                               unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                               unsigned long int **parent_number, float **weight, float **bias,
                               float **weight_grad, float **bias_grad,
                               float *iter_forward, float *iter_backward,
                               unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices);
float calc_network_mini_batch(float **datas, unsigned long int mini_batch_len,
                              unsigned long int *neighbour_number, unsigned long int *bias_number,
                              unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                              unsigned long int **parent_number, float **weight, float **bias,
                              unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices);

void update_weight_bias_gd(unsigned long int *neighbour_number, unsigned long int *bias_number,
                           float **weight, float **bias,
                           float **weight_grad, float **bias_grad,
                           float **vt_weight, float **vt_bias,
                           unsigned long int **graph_logic, unsigned long int **bias_logic,
                           float **fix_weight, float **fix_bias);

void update_weight_bias_adamax(unsigned long int *neighbour_number, unsigned long int *bias_number,
                               float **weight, float **bias,
                               float **weight_grad, float **bias_grad,
                               float adam_beta1t, float **mt_weight, float **ut_weight,
                               float **mt_bias, float **ut_bias,
                               unsigned long int **graph_logic, unsigned long int **bias_logic,
                               float **fix_weight, float **fix_bias);

void update_weight_bias_adam(unsigned long int *neighbour_number, unsigned long int *bias_number,
                             float **weight, float **bias,
                             float **weight_grad, float **bias_grad,
                             float adam_beta1t, float adam_beta2t,
                             float **mt_weight, float **vt_weight,
                             float **mth_weight, float **vth_weight,
                             float **mt_bias, float **vt_bias,
                             float **mth_bias, float **vth_bias,
                             unsigned long int **graph_logic, unsigned long int **bias_logic,
                             float **fix_weight, float **fix_bias);

void update_weight_bias_radam(unsigned long int *neighbour_number, unsigned long int *bias_number,
                              float **weight, float **bias,
                              float **weight_grad, float **bias_grad,
                              float adam_beta1t, float adam_beta2t, float time_rho,
                              float **mt_weight, float **vt_weight,
                              float **mth_weight, float **vth_weight,
                              float **mt_bias, float **vt_bias,
                              float **mth_bias, float **vth_bias,
                              unsigned long int **graph_logic, unsigned long int **bias_logic,
                              float **fix_weight, float **fix_bias);
float dmax(float a, float b);
unsigned long int imax(unsigned long int a, unsigned long int b);
void calc_num_gradient_mini_batch(float **datas, unsigned long int mini_batch_len,
                                  unsigned long int *neighbour_number, unsigned long int *bias_number,
                                  unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                                  unsigned long int **parent_number, float **weight, float **bias,
                                  float **weight_num_grad, float **bias_num_grad,
                                  unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices);
float **allocate_dmatrix(unsigned long int row_num, unsigned long int *col_num);
unsigned long int **allocate_imatrix(unsigned long int row_num, unsigned long int *col_num);
void deallocate_dmatrix(float **m, unsigned long int row_num);
void deallocate_imatrix(unsigned long int **m, unsigned long int row_num);
void print_progress_bar(unsigned long int max_length, float rate);
void save_weight_bias(char filename[100], float **weight, float **bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float **mt_weight, float **mth_weight, float **vt_weight, float **vth_weight, float **ut_weight,
                      float **mt_bias, float **mth_bias, float **vt_bias, float **vth_bias, float **ut_bias,
                      float adam_beta1t, float adam_beta2t, float time_rho);
void load_weight_bias(char filename[100], float **weight, float **bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float **mt_weight, float **mth_weight, float **vt_weight, float **vth_weight, float **ut_weight,
                      float **mt_bias, float **mth_bias, float **vt_bias, float **vth_bias, float **ut_bias,
                      float adam_beta1t, float adam_beta2t, float time_rho);
float matrix_norm(float **m, unsigned long int row_num, unsigned long int *col_num);
float calc_error(float *neuron_value, float *target_vector);

void print_graph(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n,
                 unsigned long int **graph_i, unsigned long int **parent_number,
                 unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices,
                 float **weight, float **bias,
                 unsigned long int **graph_logic, unsigned long int **bias_logic,
                 float **fix_weight, float **fix_bias);
void program_failure(char str[]);
float random_normal(float mean, float std_dev);
void softmax(float *input, unsigned long int input_len);
void make_predictions(float **datas, unsigned long int mini_batch_len,
                      unsigned long int *neighbour_number, unsigned long int *bias_number,
                      unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                      unsigned long int **parent_number, float **weight, float **bias,
                      unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices,
                      float **predictions_mini_batch);

float calc_matrix_norm(float **m1, unsigned long int row_nums, unsigned long int *col_nums);
float calc_matrix_norm_l2(float **m1, unsigned long int row_nums, unsigned long int *col_nums);

int main()
{
    // Allocatables

    float **datas_mini_batch;
    unsigned long int *neighbour_number;
    unsigned long int *bias_number;
    unsigned long int **activation_type;
    unsigned long int **graph_n;
    unsigned long int **graph_i;
    unsigned long int **parent_number;
    float **weight;
    float **bias;
    float **weight_grad;
    float **bias_grad;
    float **weight_num_grad;
    float **bias_num_grad;
    float **mt_weight;
    float **mth_weight;
    float **vt_weight;
    float **vth_weight;
    float **ut_weight;
    float **mt_bias;
    float **mth_bias;
    float **vt_bias;
    float **vth_bias;
    float **ut_bias;
    unsigned long int *dist;             // <--- distances for PERT method
    unsigned long int *dist_extra;       // <--- distances for PERT method with an extra step to check whether is a cycle in the graph
    unsigned long int *dist_number;      // <--- count the neurons by distance
    unsigned long int *dist_number_temp; // <--- temporal vector to count the neurons by distance
    unsigned long int **dist_indices;    // <--- list of the neuron indices by distance
    unsigned long int **graph_logic;
    unsigned long int **bias_logic;

    float **fix_weight;
    float **fix_bias;
    float **predictions;
    float **predictions_mini_batch;

    // Set inf
    inf = 1.e20;

    // Read input parameters
    read_parameters("./inputs/simulparams.dat");
    // printf("sfreq: %u\n",sfreq);

    omp_set_num_threads(thread_num);

    // Set random seed
    if (seed == 0)
    {
        srand(time(0));
    }
    else
    {
        srand(seed);
    }

    float iter_backward_old;

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

    unsigned long int valid_num = (data_num - learn_num);
    unsigned long int *learn_indexes = (unsigned long int *)malloc(learn_num * sizeof(unsigned long int));
    unsigned long int *valid_indexes = (unsigned long int *)malloc(valid_num * sizeof(unsigned long int));
    for (unsigned long int i = 0; i < learn_num; i++)
    {
        learn_indexes[i] = i;
    }
    for (unsigned long int i = 0; i < valid_num; i++)
    {
        valid_indexes[i] = learn_num + i;
    }

    // Allocation for data
    // datas = (float **)malloc(data_num * sizeof(float *));
    datas_mini_batch = (float **)malloc(mini_batch_size * sizeof(float *));

    for (unsigned long int i = 0; i < mini_batch_size; i++)
    {
        datas_mini_batch[i] = (float *)malloc((input_num + output_num) * sizeof(float));
    }

    predictions = (float **)malloc(data_num * sizeof(float *));
    predictions_mini_batch = (float **)malloc(mini_batch_size * sizeof(float *));

    for (unsigned long int i = 0; i < data_num; i++)
    {
        predictions[i] = (float *)malloc(output_num * sizeof(float));
    }
    for (unsigned long int i = 0; i < mini_batch_size; i++)
    {
        predictions_mini_batch[i] = (float *)malloc(output_num * sizeof(float));
    }

    // Read graph data and allocate them
    neighbour_number = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));
    bias_number = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));
    activation_type = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    graph_n = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    graph_i = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    graph_logic = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    bias_logic = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));

    fix_weight = (float **)malloc(neuron_num * sizeof(float *));
    fix_bias = (float **)malloc(neuron_num * sizeof(float *));

    read_graph(graph_datas, logic_datas, fixwb_datas, neighbour_number, bias_number,
               activation_type, graph_n, graph_i, graph_logic, bias_logic, fix_weight, fix_bias);

    parent_number = (unsigned long int **)malloc(neuron_num * sizeof(unsigned long int *));
    weight = (float **)malloc(neuron_num * sizeof(float *));
    bias = (float **)malloc(neuron_num * sizeof(float *));
    bias_grad = (float **)malloc(neuron_num * sizeof(float *));
    initialize_weights(neighbour_number, bias_number, activation_type, graph_n, graph_i, parent_number, weight, bias);

    dist = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));       // neuron distances for PERT method
    dist_extra = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int)); // neuron distances for PERT method

    // Required allocations
    weight_grad = allocate_dmatrix(neuron_num, neighbour_number);
    weight_num_grad = allocate_dmatrix(neuron_num, neighbour_number);
    bias_grad = allocate_dmatrix(neuron_num, bias_number);
    bias_num_grad = allocate_dmatrix(neuron_num, bias_number);

    mt_weight = allocate_dmatrix(neuron_num, neighbour_number);
    mth_weight = allocate_dmatrix(neuron_num, neighbour_number);
    vt_weight = allocate_dmatrix(neuron_num, neighbour_number);
    vth_weight = allocate_dmatrix(neuron_num, neighbour_number);
    ut_weight = allocate_dmatrix(neuron_num, neighbour_number);

    mt_bias = allocate_dmatrix(neuron_num, bias_number);
    mth_bias = allocate_dmatrix(neuron_num, bias_number);
    vt_bias = allocate_dmatrix(neuron_num, bias_number);
    vth_bias = allocate_dmatrix(neuron_num, bias_number);
    ut_bias = allocate_dmatrix(neuron_num, bias_number);

    // Setting the fix weights and biases
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
        {
            if (graph_logic[neuron_id][neighbour_ind] == 0)
            {
                weight[neuron_id][neighbour_ind] = fix_weight[neuron_id][neighbour_ind];
            }
        }
    }
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
        {
            if (bias_logic[neuron_id][bias_ind] == 0)
            {
                bias[neuron_id][bias_ind] = fix_bias[neuron_id][bias_ind];
            }
        }
    }

    float adam_beta1t, adam_beta2t, time_rho;

    if (loaddatas == 1)
    {
        load_weight_bias(load_backup, weight, bias, neuron_num, neighbour_number, bias_number,
                         mt_weight, mth_weight, vt_weight, vth_weight, ut_weight,
                         mt_bias, mth_bias, vt_bias, vth_bias, ut_bias,
                         adam_beta1t, adam_beta2t, time_rho);

        // Setting the fix weights and biases
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                if (graph_logic[neuron_id][neighbour_ind] == 0)
                {
                    weight[neuron_id][neighbour_ind] = fix_weight[neuron_id][neighbour_ind];
                }
            }
        }
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                if (bias_logic[neuron_id][bias_ind] == 0)
                {
                    bias[neuron_id][bias_ind] = fix_bias[neuron_id][bias_ind];
                }
            }
        }
    }

    if (zero_optim_param == 1)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                mt_weight[neuron_id][neighbour_ind] = 0.0;
                mth_weight[neuron_id][neighbour_ind] = 0.0;
                vt_weight[neuron_id][neighbour_ind] = 0.0;
                vth_weight[neuron_id][neighbour_ind] = 0.0;
                ut_weight[neuron_id][neighbour_ind] = 0.0;
            }
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                mt_bias[neuron_id][bias_ind] = 0.0;
                mth_bias[neuron_id][bias_ind] = 0.0;
                vt_bias[neuron_id][bias_ind] = 0.0;
                vth_bias[neuron_id][bias_ind] = 0.0;
                ut_bias[neuron_id][bias_ind] = 0.0;
            }
        }
        adam_beta1t = 1.0;
        adam_beta2t = 1.0;
        time_rho = 0.0;
    }

    if ((loaddatas == 0) && (zero_optim_param == 0))
    {
        program_failure("Logical error: zero_optim_param or loaddatas\n");
    }
    unsigned long int check_cycle = 0;
    dist = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int));       // neuron distances for PERT method
    dist_extra = (unsigned long int *)malloc(neuron_num * sizeof(unsigned long int)); // neuron distances for PERT method
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
                    dist[graph_n[neuron_id][neighbour_ind]] = imax(dist[graph_n[neuron_id][neighbour_ind]], dist[neuron_id] + 1);
                }
            }
        }
        dist_max = dist[neuron_num - 1];

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            dist_extra[neuron_id] = dist[neuron_id];
        }

        // Make one extra step to check whether is a cycle in the graph
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                dist_extra[graph_n[neuron_id][neighbour_ind]] = imax(dist[graph_n[neuron_id][neighbour_ind]], dist[neuron_id] + 1);
            }
        }

        check_cycle = 0;
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
            if (activation_type[neuron_id][0] != 0)
            {
                output_logic += 1;
            }
        }

        if (output_logic > 0)
        {
            //    program_failure("Wrong activation function type on the output!");
        }
    }

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

        // dist_indices = (int **)malloc(dist_max * sizeof(int *)); // list of the neurons
        dist_indices = allocate_imatrix(dist_max + 1, dist_number);

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
            dist_indices[dist[neuron_id]][dist_number_temp[dist[neuron_id]]] = neuron_id;
            dist_number_temp[dist[neuron_id]]++;
        }
    }

    // print_graph(neighbour_number, bias_number, activation_type, graph_n, graph_i, parent_number,
    //              dist, dist_max, dist_number, dist_indices, weight, bias,
    //              graph_logic, bias_logic, fix_weight, fix_bias);

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

    float iter_forward_temp = 0.0, iter_backward_temp = 0.0;
    float iter_forward = 0.0, iter_backward = 0.0;

    while (iter_grad < maxiter_grad)
    {

        iter_grad++;

        float start_clock_one_step, end_clock_one_step;
        start_clock_one_step = omp_get_wtime();
        float elapsed_time_temp = 0.0;

        iter_backward_old = iter_backward;

        float numgrad_error = 0.0;
        float error_temp_mean = 0.0;

        iter_backward = 0.0;
        iter_forward = 0.0;

        FILE *f_data = fopen(input_name, "r");
        for (int mini_batch_id = 0; mini_batch_id < mini_batch_num; mini_batch_id++)
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

            float error_temp = calc_gradient_mini_batch(datas_mini_batch, mini_batch_len,
                                                        neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                                        parent_number, weight, bias, weight_grad, bias_grad,
                                                        &iter_forward_temp, &iter_backward_temp,
                                                        dist, dist_max, dist_number, dist_indices);
            error_temp_mean += error_temp;

            iter_forward += iter_forward_temp;
            iter_backward += iter_backward_temp;

            // Numerical gradient
            float numgrad_error_temp = 0.0;
            if (numgrad > 0)
            {
                if (iter_grad % numgrad == 0)
                {
                    calc_num_gradient_mini_batch(datas_mini_batch, mini_batch_len,
                                                 neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                                 parent_number, weight, bias,
                                                 weight_num_grad, bias_num_grad,
                                                 dist, dist_max, dist_number, dist_indices);
                    float numgrad_error_temp_weight = calc_diff_matrices(weight_grad, weight_num_grad, neuron_num, neighbour_number);
                    float numgrad_error_temp_bias = calc_diff_matrices(bias_grad, bias_num_grad, neuron_num, bias_number);

                    numgrad_error_temp = dmax(numgrad_error_temp_weight, numgrad_error_temp_bias);
                    numgrad_error += numgrad_error_temp;
                }
            }

            end_clock_one_step = omp_get_wtime();
            elapsed_time_temp = end_clock_one_step - start_clock_one_step;

            switch (optimizer)
            {
            case 1:
                update_weight_bias_gd(neighbour_number, bias_number, weight, bias, weight_grad, bias_grad,
                                      vt_weight, vt_bias,
                                      graph_logic, bias_logic, fix_weight, fix_bias);
                break;
            case 2:
                adam_beta1t *= adam_beta1;
                adam_beta2t *= adam_beta2;
                update_weight_bias_adam(neighbour_number, bias_number, weight, bias, weight_grad, bias_grad,
                                        adam_beta1t, adam_beta2t,
                                        mt_weight, vt_weight, mth_weight, vth_weight,
                                        mt_bias, vt_bias, mth_bias, vth_bias,
                                        graph_logic, bias_logic, fix_weight, fix_bias);
                break;
            case 3:
                adam_beta1t *= adam_beta1;
                update_weight_bias_adamax(neighbour_number, bias_number, weight, bias, weight_grad, bias_grad,
                                          adam_beta1t, mt_weight, ut_weight, mt_bias, ut_bias,
                                          graph_logic, bias_logic, fix_weight, fix_bias);

                break;
            case 4:
                adam_beta1t *= adam_beta1;
                adam_beta2t *= adam_beta2;
                time_rho += 1.0;
                update_weight_bias_radam(neighbour_number, bias_number, weight, bias, weight_grad, bias_grad,
                                         adam_beta1t, adam_beta2t, time_rho,
                                         mt_weight, vt_weight, mth_weight, vth_weight,
                                         mt_bias, vt_bias, mth_bias, vth_bias,
                                         graph_logic, bias_logic, fix_weight, fix_bias);
                break;
            }

            for (unsigned long int i = 0; i < 10; i++)
            {
                printf("\b \b");
            }

            printf("\r");
            // printf("\n");
            if (mini_batch_id < mini_batch_num - 1)
            {
                print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                // printf(" [%u/%u] TE: %.5f ETA: %.1fs", mini_batch_id + 1,
                //        mini_batch_num, error_temp,
                //        elapsed_time_temp * mini_batch_num / (mini_batch_id + 1) - elapsed_time_temp + 0.01);
                printf(" [%u/%u] TE: %.5f ET: %.1fs ETA: %.1fs", mini_batch_id + 1,
                       mini_batch_num, error_temp,
                       elapsed_time_temp, elapsed_time_temp * mini_batch_num / (mini_batch_id + 1) - elapsed_time_temp + 0.01);
            }
            else
            {
                print_progress_bar(10, (mini_batch_id + 1) / (float)mini_batch_num);
                printf(" [%u/%u] TE: %.5f ET: %.1fs", mini_batch_id + 1,
                       mini_batch_num, error_temp,
                       elapsed_time_temp);
            }

            fflush(stdout);
        }
        iter_forward /= mini_batch_num;
        iter_backward /= mini_batch_num;
        numgrad_error /= mini_batch_num;
        elapsed_time += elapsed_time_temp;

        start_clock_one_step = omp_get_wtime();

        float error_learn = error_temp_mean / mini_batch_num;
        float error_valid = 0.0;

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
            float error_valid_temp = calc_network_mini_batch(datas_mini_batch, mini_batch_len,
                                                             neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                                             parent_number, weight, bias,
                                                             dist, dist_max, dist_number, dist_indices);
            error_valid += error_valid_temp * mini_batch_len;
        }
        error_valid /= data_num - learn_num;

        // Display the progress
        end_clock_one_step = omp_get_wtime();
        float elapsed_time_temp_valid = end_clock_one_step - start_clock_one_step;
        elapsed_time_temp += elapsed_time_temp_valid;
        elapsed_time += elapsed_time_temp_valid;

        printf(" | ");
        print_progress_bar(10, iter_grad / (float)maxiter_grad);
        printf(" %3lu%% [%u/%u] TE: %.5f: VE: %.5f ET: %.1fs ETA: %.1fs", iter_grad * 100 / maxiter_grad, iter_grad,
               maxiter_grad, error_learn, error_valid, elapsed_time, elapsed_time * maxiter_grad / iter_grad - elapsed_time + 0.01);

        if (numgrad > 0)
        {
            if (iter_grad % numgrad == 0)
            {
                printf(" NG. err: %.4E", numgrad_error);
            }
        }
        fclose(f_data);
        printf("\n");

        // Logging --- Evaluation
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

                    make_predictions(datas_mini_batch, mini_batch_len, neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                     parent_number, weight, bias, dist, dist_max, dist_number, dist_indices, predictions_mini_batch);
                    for (unsigned long int i = 0; i < mini_batch_len; i++)
                    {
                        for (unsigned long int j = 0; j < output_num; j++)
                        {
                            predictions[mini_batch_si + i][j] = predictions_mini_batch[i][j];
                        }
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
                                acc_learn += (unsigned long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long int)roundf(datas_mini_batch[i][input_num + j] + 0.01);
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
                            float true_max = datas_mini_batch[i][input_num + 0];
                            for (unsigned long int j = 1; j < output_num; j++)
                            {
                                if (datas_mini_batch[i][input_num + j] > true_max)
                                {
                                    true_ind = j;
                                    true_max = datas_mini_batch[i][input_num + j];
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

                    make_predictions(datas_mini_batch, mini_batch_len, neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                     parent_number, weight, bias, dist, dist_max, dist_number, dist_indices, predictions_mini_batch);
                    for (unsigned long int i = 0; i < mini_batch_len; i++)
                    {
                        for (unsigned long int j = 0; j < output_num; j++)
                        {
                            predictions[mini_batch_si + i][j] = predictions_mini_batch[i][j];
                        }
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
                                acc_valid += (unsigned long int)roundf(predictions_mini_batch[i][j] + 0.01) == (unsigned long int)roundf(datas_mini_batch[i][input_num + j] + 0.01);
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
                            float true_max = datas_mini_batch[i][input_num + 0];
                            for (unsigned long int j = 1; j < output_num; j++)
                            {
                                if (datas_mini_batch[i][input_num + j] > true_max)
                                {
                                    true_ind = j;
                                    true_max = datas_mini_batch[i][input_num + j];
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

        if (iter_grad > 0)
        {
            if (iter_grad % sfreq == 0)
            {
                save_weight_bias(save_backup, weight, bias, neuron_num, neighbour_number, bias_number,
                                 mt_weight, mth_weight, vt_weight, vth_weight, ut_weight,
                                 mt_bias, mth_bias, vt_bias, vth_bias, ut_bias,
                                 adam_beta1t, adam_beta2t, time_rho);
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

    if (maxiter_grad == 0)
    {
        // Final prediction
        f = fopen(predict_name, "w");

        // Make predictions and evaluations
        if (f)
        {
            unsigned long int mini_batch_num = data_num / mini_batch_size;
            if (mini_batch_size * mini_batch_num != data_num)
            {
                mini_batch_num++;
            }

            // All data
            FILE *f_data = fopen(input_name, "r");

            for (unsigned long int mini_batch_id = 0; mini_batch_id < mini_batch_num; mini_batch_id++)
            {
                unsigned long int mini_batch_len;

                unsigned long int mini_batch_si = mini_batch_id * mini_batch_size;
                unsigned long int mini_batch_ei = (mini_batch_id + 1) * mini_batch_size - 1;
                if (mini_batch_ei > data_num - 1)
                {
                    mini_batch_ei = data_num - 1;
                }
                mini_batch_len = mini_batch_ei - mini_batch_si + 1;
                read_data(datas_mini_batch, mini_batch_len, f_data, 1);

                make_predictions(datas_mini_batch, mini_batch_len, neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                 parent_number, weight, bias, dist, dist_max, dist_number, dist_indices, predictions_mini_batch);

                for (unsigned long int i = 0; i < mini_batch_len; i++)
                {
                    for (unsigned long int j = 0; j < output_num; j++)
                    {
                        fprintf(f, "%f ", predictions_mini_batch[i][j]);
                    }
                    fprintf(f, "\n");
                }
            }

            fclose(f_data);
        }
        else
        {
            program_failure("File write error: final predictions\n");
        }
        fclose(f);
    }

    // Deallocations
    deallocate_dmatrix(bias, neuron_num);
    deallocate_dmatrix(bias_grad, neuron_num);
    deallocate_dmatrix(weight_num_grad, neuron_num);
    deallocate_dmatrix(bias_num_grad, neuron_num);
    deallocate_dmatrix(fix_weight, neuron_num);
    deallocate_dmatrix(fix_bias, neuron_num);
    deallocate_imatrix(activation_type, neuron_num);
    deallocate_imatrix(graph_n, neuron_num);
    deallocate_imatrix(graph_i, neuron_num);
    deallocate_imatrix(graph_logic, neuron_num);
    deallocate_imatrix(bias_logic, neuron_num);
    deallocate_imatrix(parent_number, neuron_num);
    deallocate_dmatrix(mt_weight, neuron_num);
    deallocate_dmatrix(mth_weight, neuron_num);
    deallocate_dmatrix(vt_weight, neuron_num);
    deallocate_dmatrix(vth_weight, neuron_num);
    deallocate_dmatrix(ut_weight, neuron_num);
    deallocate_dmatrix(mt_bias, neuron_num);
    deallocate_dmatrix(mth_bias, neuron_num);
    deallocate_dmatrix(vt_bias, neuron_num);
    deallocate_dmatrix(vth_bias, neuron_num);
    deallocate_dmatrix(ut_bias, neuron_num);

    for (unsigned long int i = 0; i < data_num; i++)
    {
        free(predictions[i]);
    }
    for (unsigned long int i = 0; i < mini_batch_size; i++)
    {
        free(predictions_mini_batch[i]);
    }
    free(predictions_mini_batch);

    for (unsigned long int i = 0; i < mini_batch_size; i++)
    {
        free(datas_mini_batch[i]);
    }
    free(datas_mini_batch);

    free(predictions);

    free(neighbour_number);
    free(bias_number);

    deallocate_dmatrix(weight, neuron_num);
    deallocate_dmatrix(weight_grad, neuron_num);
    free(learn_indexes);
    free(valid_indexes);
    free(dist);
    if ((ff_optimization > 0) && (check_cycle == 0))
    {
        free(dist_number);
        free(dist_number_temp);
        deallocate_imatrix(dist_indices, dist_max + 1);
    }
    free(dist_extra);

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
        fscanf(f, "%u", &seed);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &thread_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &tol_fixit);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &maxiter_grad);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &maxiter_fix);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &initdx);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &sfreq);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", input_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", output_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", predict_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", acc_name);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &data_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &learn_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &mini_batch_size);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &neuron_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &input_num);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &output_num);
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
        fscanf(f, "%u", &optimizer);
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
        fscanf(f, "%u", &ff_optimization);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &chunker);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &chunk_treshold);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &loaddatas);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", load_backup);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%s", save_backup);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &zero_optim_param);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%u", &numgrad);
        fscanf(f, "%s", temp_string);
        fscanf(f, "%f", &numgrad_eps);
        fclose(f);
    }
    else
    {
        program_failure("File read error: simulparams.dat\n");
    }
}

void program_failure(char str[])
{
    /**
     * Program failure
     */
    perror(str);
    exit(EXIT_FAILURE);
}

void read_data(float **datas, unsigned long int line_number, FILE *f_data, unsigned long int test)
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
                fscanf(f_data, "%f", &datas[i][j]);
            }
        }
    }
    else
    {
        program_failure("File read error in data file!");
    }
}

void read_graph(char graph_file_name[100], char logic_file_name[100], char fixwb_file_name[100],
                unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                unsigned long int **graph_logic, unsigned long int **bias_logic,
                float **fix_weight, float **fix_bias)
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
            fscanf(f_graph, "%u", &neighbour_number[neuron_id]);
            fscanf(f_graph, "%s", temp_string);

            graph_n[neuron_id] = (unsigned long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long int));
            graph_i[neuron_id] = (unsigned long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long int));

            graph_logic[neuron_id] = (unsigned long int *)malloc(neighbour_number[neuron_id] * sizeof(unsigned long int));
            fix_weight[neuron_id] = (float *)malloc(neighbour_number[neuron_id] * sizeof(float));

            for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
            {
                fscanf(f_graph, "%u", &graph_n[neuron_id][i]);
                graph_n[neuron_id][i]--;
                fscanf(f_graph, "%u", &graph_i[neuron_id][i]);
                graph_i[neuron_id][i]--;
                fscanf(f_graph, "%s", temp_string);
            }

            for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
            {
                fscanf(f_logic, "%u", &graph_logic[neuron_id][i]);
                if (graph_logic[neuron_id][i] == 0)
                {
                    fscanf(f_fixwb, "%f", &fix_weight[neuron_id][i]);
                }
            }

            fscanf(f_graph, "%s", temp_string);
            fscanf(f_logic, "%s", temp_string);
            fscanf(f_fixwb, "%s", temp_string);

            fscanf(f_graph, "%u", &bias_number[neuron_id]);
            fscanf(f_graph, "%s", temp_string);

            activation_type[neuron_id] = (unsigned long int *)malloc(bias_number[neuron_id] * sizeof(unsigned long int));
            bias_logic[neuron_id] = (unsigned long int *)malloc(bias_number[neuron_id] * sizeof(unsigned long int));
            fix_bias[neuron_id] = (float *)malloc(bias_number[neuron_id] * sizeof(float));
            for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
            {
                fscanf(f_graph, "%u", &activation_type[neuron_id][i]);
            }

            for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
            {
                fscanf(f_logic, "%u", &bias_logic[neuron_id][i]);
                if (bias_logic[neuron_id][i] == 0)
                {
                    fscanf(f_fixwb, "%f", &fix_bias[neuron_id][i]);
                }
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

void print_graph(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                 unsigned long int **parent_number,
                 unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices,
                 float **weight, float **bias,
                 unsigned long int **graph_logic, unsigned long int **bias_logic,
                 float **fix_weight, float **fix_bias)
{
    printf("The graph:\n");
    printf("----------------\n");
    for (int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%u |", neighbour_number[neuron_id]);
        for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
        {
            printf(" %u %u ; ", graph_n[neuron_id][i], graph_i[neuron_id][i]);
        }
        printf("\n");
    }
    printf("\nThe activations:\n");
    printf("----------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%u |", bias_number[neuron_id]);
        for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
        {
            printf(" %u ", activation_type[neuron_id][i]);
        }
        printf("\n");
    }
    printf("\nThe parent numbers:\n");
    printf("-------------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%u | ", neuron_id);
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            printf("%u ", parent_number[neuron_id][j]);
        }
        printf("\n");
    }

    printf("\nThe weights:\n");
    printf("------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%u | ", neuron_id);
        for (unsigned long int j = 0; j < neighbour_number[neuron_id]; j++)
        {
            printf(" %f ", weight[neuron_id][j]);
        }
        printf("\n");
    };
    printf("\nThe biases:\n");
    printf("-----------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%u | ", neuron_id);
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            printf(" %f ", bias[neuron_id][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("The distances:\n");
    printf("----------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%u | %u \n", neuron_id + 1, dist[neuron_id]);
    }
    printf("Max distance: %u \n", dist_max);
    printf("\n");
    printf("How many neurons by distance:\n");
    printf("----------------\n");
    for (unsigned long int i = 0; i <= dist_max; i++)
    {
        printf("%u | %3lu | ", i, dist_number[i]);
        for (unsigned long int j = 0; j < dist_number[i]; j++)
        {
            printf("%3lu ", dist_indices[i][j] + 1);
        }
        printf("\n");
    }
    printf("\n");
    printf("The mutability of the graph:\n");
    printf("----------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%3lu | %3lu |", neuron_id + 1, neighbour_number[neuron_id]);
        for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
        {
            printf(" %u ", graph_logic[neuron_id][i]);
        }
        printf("\n");
    }
    printf("\n");
    printf("The inmutable weights:\n");
    printf("----------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%3lu | %3lu |", neuron_id + 1, neighbour_number[neuron_id]);
        for (unsigned long int i = 0; i < neighbour_number[neuron_id]; i++)
        {
            if (graph_logic[neuron_id][i] == 0)
            {
                printf(" (%u): %5f ", graph_n[neuron_id][i] + 1, fix_weight[neuron_id][i]);
            }
        }
        printf("\n");
    }
    printf("\n");
    printf("The mutability of the biases:\n");
    printf("----------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%3lu | %3lu |", neuron_id + 1, bias_number[neuron_id]);
        for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
        {
            printf(" %u ", bias_logic[neuron_id][i]);
        }
        printf("\n");
    }
    printf("\n");
    printf("The inmutable biases:\n");
    printf("----------------\n");
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        printf("%3lu | %3lu |", neuron_id + 1, bias_number[neuron_id]);
        for (unsigned long int i = 0; i < bias_number[neuron_id]; i++)
        {
            if (bias_logic[neuron_id][i] == 0)
            {
                printf(" (%u): %5f ", i + 1, fix_bias[neuron_id][i]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

unsigned long int calc_network_one_sample(unsigned long int *neighbour_number, unsigned long int *bias_number,
                            unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                            unsigned long int **parent_number, float **weight, float **bias,
                            float **input_value, float *neuron_value)
{
    /**
     * Calculate the network on one sample
     * returns the iteration number
     */

    float error_iter = inf;
    unsigned long int iter_fix = 0;
    float **input_value_old = allocate_dmatrix(neuron_num, bias_number);
    float **input_value_orig = allocate_dmatrix(neuron_num, bias_number);
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        float x = 1.0;
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            input_value_old[neuron_id][j] = input_value[neuron_id][j];
            input_value_orig[neuron_id][j] = input_value[neuron_id][j];
            x = x * act_fun(input_value[neuron_id][j], activation_type[neuron_id][j]);
        }
        neuron_value[neuron_id] = x;
    }

    while (error_iter > tol_fixit && iter_fix < maxiter_fix)
    {
        iter_fix++;
        // input_value_old
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                input_value_old[neuron_id][j] = input_value[neuron_id][j];
                input_value[neuron_id][j] = input_value_orig[neuron_id][j];
            }
        }

        // Calculate the input input values
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int j = 0; j < neighbour_number[neuron_id]; j++)
            {
                unsigned long int input_ind_1 = graph_n[neuron_id][j];
                unsigned long int input_ind_2 = graph_i[neuron_id][j];
                input_value[input_ind_1][input_ind_2] += weight[neuron_id][j] * neuron_value[neuron_id];
            }
        }

        // Add the bias and calculate the values of the neuron
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            float x = 1.0;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                input_value[neuron_id][j] += bias[neuron_id][j];
                x = x * act_fun(input_value[neuron_id][j], activation_type[neuron_id][j]);
            }
            neuron_value[neuron_id] = x;
        }

        // Calculate the error
        error_iter = calc_diff_matrices(input_value, input_value_old, neuron_num, bias_number);
    }
    deallocate_dmatrix(input_value_old, neuron_num);
    deallocate_dmatrix(input_value_orig, neuron_num);

    return iter_fix;
}

unsigned long int calc_network_one_sample_ff(unsigned long int *neighbour_number, unsigned long int *bias_number,
                               unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                               unsigned long int **parent_number, float **weight, float **bias,
                               float **input_value, float *neuron_value,
                               unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices)
{
    /**
     * Calculate the network on one sample in a feed forwarded network
     * returns the iteration number
     */

    unsigned long int iter_fix = 0;

    for (unsigned long int layer_id = 0; layer_id <= dist_max; layer_id++) // Here we need `<=` because the indexing of the layers
    {
        iter_fix++;
        for (int dist_index = 0; dist_index < dist_number[layer_id]; dist_index++)
        {
            // Calculating the values of the neurons in the `layer_id`-th layer
            unsigned long int neuron_id = dist_indices[layer_id][dist_index];
            float x = 1.0;
            for (unsigned long int input_id = 0; input_id < bias_number[neuron_id]; input_id++)
            {
                input_value[neuron_id][input_id] += bias[neuron_id][input_id]; // Adding the bias term
                x = x * act_fun(input_value[neuron_id][input_id], activation_type[neuron_id][input_id]);
            }
            neuron_value[neuron_id] = x;
            // printf("output: %f \n",x);

            // Forward propagation from this neuron
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                unsigned long int input_ind_1 = graph_n[neuron_id][neighbour_ind];
                unsigned long int input_ind_2 = graph_i[neuron_id][neighbour_ind];
                input_value[input_ind_1][input_ind_2] += weight[neuron_id][neighbour_ind] * neuron_value[neuron_id];
            }
        }
    }
    // printf("output: %f \n",neuron_value[neuron_num-1]);

    return iter_fix;
}

int rand_range_int(int min, int max)
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

void initialize_weights(unsigned long int *neighbour_number, unsigned long int *bias_number, unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                        unsigned long int **parent_number, float **weight, float **bias)
{
    /**
     *  Initialize the weights and the biases
     */

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
        return 1.0 / (1.0 + expf(-x));
        break;
    case 2:
        return tanhf(x);
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
        return x / (1.0 + expf(-x));
        break;
    case 6:
        return 1.0 - x;
        break;
    case 7:
        return 1.0 / x;
        break;
    case 8:
        return cosf(x);
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
        return 1.0 - tanhf(x) * tanhf(x);
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
        return (1.0 + expf(-x) + x * expf(-x)) / powf(1.0 + expf(-x), 2.0);
        break;
    case 6:
        return -1.0;
        break;
    case 7:
        return -1.0 / powf(x, 2.0);
        break;
    case 8:
        return -sinf(x);
        break;
    case 9:
        return 1.0 / (1.0 + x * x);
        break;
    default:
        return 0.0;
        break;
    }
}

unsigned long int calc_gradient_one_sample(unsigned long int *neighbour_number, unsigned long int *bias_number,
                             unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                             unsigned long int **parent_number, float **weight, float **bias,
                             float **input_value, float *neuron_value,
                             float **weight_grad, float **bias_grad, float *target_vector)
{
    /**
     * Calculate the gradient for feed forwarded network
     */

    // Calculate help vector
    float **weight_grad_help_temp = allocate_dmatrix(neuron_num, bias_number);
    float **weight_grad_help = allocate_dmatrix(neuron_num, bias_number);
    float **weight_grad_inp = allocate_dmatrix(neuron_num, bias_number);
    float **weight_grad_inp_temp = allocate_dmatrix(neuron_num, bias_number);
    float **weight_grad_inp_old = allocate_dmatrix(neuron_num, bias_number);
    float *output_value = (float *)malloc(output_num * sizeof(float));

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            weight_grad_help_temp[neuron_id][j] = act_fun_diff(input_value[neuron_id][j], activation_type[neuron_id][j]);
            weight_grad_inp[neuron_id][j] = 0.0;
            weight_grad_inp_temp[neuron_id][j] = 0.0;
            weight_grad_inp_old[neuron_id][j] = 0.0;
        }
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            weight_grad_help[neuron_id][j] = weight_grad_help_temp[neuron_id][j];
            for (unsigned long int k = 0; k < bias_number[neuron_id]; k++)
            {
                if (j != k)
                {
                    weight_grad_help[neuron_id][j] *= act_fun(input_value[neuron_id][k], activation_type[neuron_id][k]);
                }
            }
        }
    }

    // Output neurons
    for (unsigned long int i = 0; i < output_num; i++)
    {
        output_value[i] = neuron_value[neuron_num - output_num + i];
    }
    if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
    {
        softmax(output_value, output_num);
    }

    if ((strcmp(lossfunction_type, "multiclassification_crossentropy") == 0) || (strcmp(lossfunction_type, "sumsquared") == 0))
    {

        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                weight_grad_inp_old[neuron_id][j] = (output_value[i] - target_vector[i]) *
                                                    weight_grad_help[neuron_id][j];
            }
        }
    }

    if (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0)
    {

        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                if ((output_value[i] > 0.0) && (output_value[i] < 1.0))
                {
                    weight_grad_inp_old[neuron_id][j] = (output_value[i] - target_vector[i]) *
                                                        weight_grad_help[neuron_id][j] / (output_value[i] * (1.0 - output_value[i]));
                }
            }
        }
    }

    if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
    {

        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                // weight_grad_inp_old[neuron_id][j] = (1.0 - target_vector[i])*output_value[i] + log(1.0+exp(-output_value[i]));
                weight_grad_inp_old[neuron_id][j] = act_fun(output_value[i], 1) - target_vector[i];
                // weight_grad_inp_old[neuron_id][j] = (output_value[i] - target_vector[i]) *
                //                                     weight_grad_help[neuron_id][j];
            }
        }
    }

    // weight_grad_inp_old = weight_grad_inp
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            weight_grad_inp_temp[neuron_id][j] = weight_grad_inp_old[neuron_id][j];
            weight_grad_inp[neuron_id][j] = weight_grad_inp_temp[neuron_id][j];
        }
    }

    // Calculate the derivatives by the inputs: weight_grad_inp
    float error_iter = inf;
    unsigned long int iter_fix = 0;

    while (error_iter > tol_fixit && iter_fix < maxiter_fix)
    {
        iter_fix++;

        // weight_grad_inp_old = weight_grad_inp
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                weight_grad_inp_old[neuron_id][j] = weight_grad_inp_temp[neuron_id][j];
                weight_grad_inp_temp[neuron_id][j] = 0.0;
            }
        }

        // Main part
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
            {
                for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
                {
                    unsigned long int neighbour_ind_n = graph_n[neuron_id][neighbour_counter];
                    unsigned long int neighbour_ind_i = graph_i[neuron_id][neighbour_counter];
                    weight_grad_inp_temp[neuron_id][bias_id] += weight[neuron_id][neighbour_counter] *
                                                                weight_grad_inp_old[neighbour_ind_n][neighbour_ind_i] *
                                                                weight_grad_help[neuron_id][bias_id];
                }
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
            {
                weight_grad_inp[neuron_id][bias_id] += weight_grad_inp_temp[neuron_id][bias_id];
            }
        }

        // Calculate error
        error_iter = calc_matrix_norm(weight_grad_inp_temp, neuron_num, bias_number);
    }
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
        {
            unsigned long int neighbour_ind_n = graph_n[neuron_id][neighbour_counter];
            unsigned long int neighbour_ind_i = graph_i[neuron_id][neighbour_counter];
            weight_grad[neuron_id][neighbour_counter] = weight_grad_inp[neighbour_ind_n][neighbour_ind_i] * neuron_value[neuron_id];
        }
    }

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
        {
            bias_grad[neuron_id][bias_id] = weight_grad_inp[neuron_id][bias_id];
        }
    }

    deallocate_dmatrix(weight_grad_help, neuron_num);
    deallocate_dmatrix(weight_grad_help_temp, neuron_num);
    deallocate_dmatrix(weight_grad_inp, neuron_num);
    deallocate_dmatrix(weight_grad_inp_temp, neuron_num);
    deallocate_dmatrix(weight_grad_inp_old, neuron_num);
    free(output_value);

    return iter_fix;
}

unsigned long int calc_gradient_one_sample_ff(unsigned long int *neighbour_number, unsigned long int *bias_number,
                                unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                                unsigned long int **parent_number, float **weight, float **bias,
                                float **input_value, float *neuron_value,
                                float **weight_grad, float **bias_grad, float *target_vector,
                                unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices)
{

    // Calculate help vector
    float **weight_grad_help_temp = allocate_dmatrix(neuron_num, bias_number);
    float **weight_grad_help = allocate_dmatrix(neuron_num, bias_number);
    float **weight_grad_inp = allocate_dmatrix(neuron_num, bias_number);
    float *output_value = (float *)malloc(output_num * sizeof(float));

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            weight_grad_help_temp[neuron_id][j] = act_fun_diff(input_value[neuron_id][j], activation_type[neuron_id][j]);
            weight_grad_inp[neuron_id][j] = 0.0;
        }
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            weight_grad_help[neuron_id][j] = weight_grad_help_temp[neuron_id][j];
            for (unsigned long int k = 0; k < bias_number[neuron_id]; k++)
            {
                if (j != k)
                {
                    weight_grad_help[neuron_id][j] *= act_fun(input_value[neuron_id][k], activation_type[neuron_id][k]);
                }
            }
        }
    }

    // Calculate the derivatives by the inputs: weight_grad_inp
    unsigned long int iter_fix = 0;

    // Output layer
    // Output neurons
    for (unsigned long int i = 0; i < output_num; i++)
    {
        output_value[i] = neuron_value[neuron_num - output_num + i];
    }
    if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
    {
        softmax(output_value, output_num);
    }

    if ((strcmp(lossfunction_type, "multiclassification_crossentropy") == 0) || (strcmp(lossfunction_type, "sumsquared") == 0))
    {

        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                weight_grad_inp[neuron_id][j] = (output_value[i] - target_vector[i]) *
                                                weight_grad_help[neuron_id][j];
            }
        }
    }

    if (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0)
    {

        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                if ((output_value[i] > 0.0) && (output_value[i] < 1.0))
                {
                    weight_grad_inp[neuron_id][j] = (output_value[i] - target_vector[i]) *
                                                    weight_grad_help[neuron_id][j] / (output_value[i] * (1.0 - output_value[i]));
                }
            }
        }
    }

    if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
    {
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
            {
                // weight_grad_inp_old[neuron_id][j] = (1.0 - target_vector[i])*output_value[i] + log(1.0+exp(-output_value[i]));
                weight_grad_inp[neuron_id][j] = (act_fun(output_value[i], 1) - target_vector[i]);
                // printf("output: %f | %f \n",output_value[i], act_fun(output_value[i],1));
                // weight_grad_inp_old[neuron_id][j] = (output_value[i] - target_vector[i]) *
                //                                     weight_grad_help[neuron_id][j];
            }
        }
    }

    // Hidden layers (and input layer, too) --- HOPPPP --- layer_id must be int
    for (int layer_id = dist_max - 1; layer_id >= 0; layer_id--)
    {
        iter_fix++;
        for (unsigned long int dist_index = 0; dist_index < dist_number[layer_id]; dist_index++)
        {
            unsigned long int neuron_id = dist_indices[layer_id][dist_index];
            for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
            {
                for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
                {
                    unsigned long int neighbour_ind_n = graph_n[neuron_id][neighbour_counter];
                    unsigned long int neighbour_ind_i = graph_i[neuron_id][neighbour_counter];
                    weight_grad_inp[neuron_id][bias_id] += weight[neuron_id][neighbour_counter] *
                                                           weight_grad_inp[neighbour_ind_n][neighbour_ind_i] *
                                                           weight_grad_help[neuron_id][bias_id];
                }
            }
        }
    }

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_counter = 0; neighbour_counter < neighbour_number[neuron_id]; neighbour_counter++)
        {
            unsigned long int neighbour_ind_n = graph_n[neuron_id][neighbour_counter];
            unsigned long int neighbour_ind_i = graph_i[neuron_id][neighbour_counter];
            weight_grad[neuron_id][neighbour_counter] = weight_grad_inp[neighbour_ind_n][neighbour_ind_i] * neuron_value[neuron_id];
        }
    }

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
        {
            bias_grad[neuron_id][bias_id] = weight_grad_inp[neuron_id][bias_id];
        }
    }

    deallocate_dmatrix(weight_grad_help, neuron_num);
    deallocate_dmatrix(weight_grad_help_temp, neuron_num);
    deallocate_dmatrix(weight_grad_inp, neuron_num);
    free(output_value);

    return iter_fix;
}

float calc_error(float *neuron_value, float *target_vector)
{
    /*
    Calculating the error functions
    */

    if (strcmp(lossfunction_type, "sumsquared") == 0)
    {

        float returner = 0.0;
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            returner += pow((neuron_value[neuron_id] - target_vector[i]), 2);
        }
        return returner;
    }
    if (strcmp(lossfunction_type, "multilabeling_crossentropy") == 0)
    {
        float returner = 0.0;
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            if ((neuron_value[neuron_id] > 0.0) && (neuron_value[neuron_id] < 1.0))
            {
                returner -= target_vector[i] * log(neuron_value[neuron_id]) + (1.0 - target_vector[i]) * log(1.0 - neuron_value[neuron_id]);
            }
        }
        return returner;
    }

    if (strcmp(lossfunction_type, "bce_multilabeling") == 0)
    {

        float returner = 0.0;
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            returner += (1.0 - target_vector[i]) * neuron_value[neuron_id] + log(1.0 + exp(-neuron_value[neuron_id]));
        }
        return returner;
    }

    if (strcmp(lossfunction_type, "multiclassification_crossentropy") == 0)
    {
        float returner = 0.0;
        float *softmax_vec = (float *)malloc(output_num * sizeof(float));

        float sum_softmax = 0.0;
        for (unsigned long int i = 0; i < output_num; i++)
        {
            unsigned long int neuron_id = neuron_num - output_num + i;
            softmax_vec[i] = neuron_value[neuron_id];
            sum_softmax += exp(softmax_vec[i]);
        }

        // softmax(softmax_vec, output_num);

        for (unsigned long int i = 0; i < output_num; i++)
        {
            returner -= target_vector[i] * (softmax_vec[i] - log(sum_softmax));
        }

        free(softmax_vec);
        return returner;
    }
}

float calc_gradient_mini_batch(float **datas, unsigned long int mini_batch_len,
                               unsigned long int *neighbour_number, unsigned long int *bias_number,
                               unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                               unsigned long int **parent_number, float **weight, float **bias,
                               float **weight_grad, float **bias_grad,
                               float *iter_forward, float *iter_backward,
                               unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices)
{
    /**
     * Calculate the gradient on a mini-batch
     *
     */

    // Allocations
    float error_mini_batch = 0.0;
    float iter_forward_temp = 0.0;
    float iter_backward_temp = 0.0;

    unsigned long int nthreads;

    // Reset weight_grad
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
        {
            weight_grad[neuron_id][neighbour_ind] = 0.0;
        }
    }
    // Reset bias_grad
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
        {
            bias_grad[neuron_id][bias_ind] = 0.0;
        }
    }

// Loop over the elements on the elements of the mini-batch
#pragma omp parallel
    {

        unsigned long int id = omp_get_thread_num();
        float error_temp;
        unsigned long int iter_f, iter_b, mini_batch_ind;
        nthreads = omp_get_num_threads();

        if (id == 0)
        {
            nthreads = omp_get_num_threads();
        }

        float **input_value = allocate_dmatrix(neuron_num, bias_number);
        float **weight_grad_temp = allocate_dmatrix(neuron_num, neighbour_number);
        float **bias_grad_temp = allocate_dmatrix(neuron_num, bias_number);
        float *neuron_value = (float *)malloc(neuron_num * sizeof(float));
        float *target_vector = (float *)malloc(output_num * sizeof(float));

#pragma omp barrier
        for (mini_batch_ind = id; mini_batch_ind < mini_batch_len; mini_batch_ind = mini_batch_ind + nthreads)
        {
            unsigned long int data_index = mini_batch_ind;

            // Copy the data for input_value with the input data
            for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
            {
                for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
                {
                    input_value[neuron_id][bias_id] = 0.0;
                }
                if (neuron_id < input_num)
                {
                    input_value[neuron_id][0] = datas[data_index][neuron_id];
                }
            }

            // Copy the data for target_vector with the target data
            for (unsigned long int i = 0; i < output_num; i++)
            {
                target_vector[i] = datas[data_index][input_num + i];
            }

            // Calculate the network
            if (ff_optimization == 0)
            {
                iter_f = calc_network_one_sample(neighbour_number, bias_number,
                                                 activation_type, graph_n, graph_i, parent_number,
                                                 weight, bias, input_value, neuron_value);
            }
            else
            {
                iter_f = calc_network_one_sample_ff(neighbour_number, bias_number,
                                                    activation_type, graph_n, graph_i, parent_number,
                                                    weight, bias, input_value, neuron_value,
                                                    dist, dist_max, dist_number, dist_indices);
            }

#pragma omp critical
            {
                iter_forward_temp += iter_f;
            }
            if ((ff_optimization == 0) || (ff_optimization == 2))
            {
                iter_b = calc_gradient_one_sample(neighbour_number, bias_number, activation_type, graph_n, graph_i, parent_number,
                                                  weight, bias, input_value, neuron_value, weight_grad_temp, bias_grad_temp, target_vector);
            }
            else
            {
                iter_b = calc_gradient_one_sample_ff(neighbour_number, bias_number, activation_type, graph_n, graph_i, parent_number,
                                                     weight, bias, input_value, neuron_value, weight_grad_temp, bias_grad_temp, target_vector,
                                                     dist, dist_max, dist_number, dist_indices);
            }

#pragma omp critical
            {
                iter_backward_temp += iter_b;
            }

            // Summing the gradients
#pragma omp critical
            {
                for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
                {
                    for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
                    {
                        weight_grad[neuron_id][neighbour_ind] += weight_grad_temp[neuron_id][neighbour_ind];
                    }
                }
            }
#pragma omp critical
            {

                for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
                {
                    for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
                    {
                        bias_grad[neuron_id][bias_id] += bias_grad_temp[neuron_id][bias_id];
                    }
                }
            }

            // Calculate the error
            error_temp = calc_error(neuron_value, target_vector);
            // printf("\n hiba: %f \n",error_temp);
#pragma omp critical
            {
                error_mini_batch += error_temp;
                // printf("\n hiba: %f %f \n",error_mini_batch, error_temp);
            }
        }
        free(neuron_value);
        free(target_vector);

        deallocate_dmatrix(input_value, neuron_num);
        deallocate_dmatrix(weight_grad_temp, neuron_num);
        deallocate_dmatrix(bias_grad_temp, neuron_num);
    }

    // Dividing
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
        {
            weight_grad[neuron_id][neighbour_ind] /= mini_batch_len;
        }
    }

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
        {
            bias_grad[neuron_id][bias_ind] /= mini_batch_len;
        }
    }

    // Regularization
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
        {
            weight_grad[neuron_id][neighbour_ind] += 2.0 * alpha * weight[neuron_id][neighbour_ind];
        }
    }

    // Chunking type 1
    if (chunker == 1)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                if (weight_grad[neuron_id][neighbour_ind] > chunk_treshold)
                {
                    weight_grad[neuron_id][neighbour_ind] = chunk_treshold;
                }
                if (weight_grad[neuron_id][neighbour_ind] < -chunk_treshold)
                {
                    weight_grad[neuron_id][neighbour_ind] = -chunk_treshold;
                }
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                if (bias_grad[neuron_id][bias_ind] > chunk_treshold)
                {
                    bias_grad[neuron_id][bias_ind] = chunk_treshold;
                }
                if (bias_grad[neuron_id][bias_ind] < -chunk_treshold)
                {
                    bias_grad[neuron_id][bias_ind] = -chunk_treshold;
                }
            }
        }
    }

    // Chunking type 2
    if (chunker == 2)
    {
        float weight_norm = matrix_norm(weight_grad, neuron_num, neighbour_number);
        float bias_norm = matrix_norm(weight_grad, neuron_num, neighbour_number);
        float scaler = dmax(weight_norm, bias_norm);

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                weight_grad[neuron_id][neighbour_ind] /= scaler;
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                bias_grad[neuron_id][bias_ind] /= scaler;
            }
        }
    }

    error_mini_batch /= mini_batch_len;
    *iter_forward = iter_forward_temp / mini_batch_len;
    *iter_backward = iter_backward_temp / mini_batch_len;

    return error_mini_batch;
}

float calc_network_mini_batch(float **datas, unsigned long int mini_batch_len,
                              unsigned long int *neighbour_number, unsigned long int *bias_number,
                              unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                              unsigned long int **parent_number, float **weight, float **bias,
                              unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices)
{
    /**
     * Calculate the network on a mini-batch
     *
     */

    // Allocations
    float error_mini_batch = 0.0;
    float iter_forward = 0.0;

    unsigned long int nthreads;

#pragma omp parallel
    {
        float **input_value = allocate_dmatrix(neuron_num, bias_number);
        float *neuron_value = (float *)malloc(neuron_num * sizeof(float));
        float *target_vector = (float *)malloc(output_num * sizeof(float));
        unsigned long int id;
        unsigned long int mini_batch_ind;
        id = omp_get_thread_num();

        nthreads = omp_get_num_threads();

        if (id == 0)
        {
            nthreads = omp_get_num_threads();
        }

        unsigned long int iter_f;
        float error_temp;

#pragma omp barrier

        // Loop over the elements on the elements of the mini-batch
        for (mini_batch_ind = id; mini_batch_ind < mini_batch_len; mini_batch_ind = mini_batch_ind + nthreads)
        {
            unsigned long int data_index = mini_batch_ind;

            // Copy the data for input_value with the input data
            for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
            {
                for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
                {
                    input_value[neuron_id][bias_id] = 0.0;
                }
                if (neuron_id < input_num)
                {
                    input_value[neuron_id][0] = datas[data_index][neuron_id];
                }
            }

            // Copy the data for target_vector with the target data
            for (unsigned long int i = 0; i < output_num; i++)
            {
                target_vector[i] = datas[data_index][input_num + i];
            }

            // Calculate the network

            if (ff_optimization == 0)
            {
                iter_f = calc_network_one_sample(neighbour_number, bias_number,
                                                 activation_type, graph_n, graph_i, parent_number,
                                                 weight, bias, input_value, neuron_value);
            }
            else
            {
                iter_f = calc_network_one_sample_ff(neighbour_number, bias_number,
                                                    activation_type, graph_n, graph_i, parent_number,
                                                    weight, bias, input_value, neuron_value,
                                                    dist, dist_max, dist_number, dist_indices);
            }

#pragma omp critical
            {
                iter_forward += iter_f;
            }

            // Calculate the error
            error_temp = calc_error(neuron_value, target_vector);
#pragma omp critical
            {
                error_mini_batch += error_temp;
            }
        }
        deallocate_dmatrix(input_value, neuron_num);
        free(neuron_value);
        free(target_vector);
    }

    error_mini_batch /= mini_batch_len;
    iter_forward /= mini_batch_len;

    return error_mini_batch;
}

void update_weight_bias_gd(unsigned long int *neighbour_number, unsigned long int *bias_number,
                           float **weight, float **bias,
                           float **weight_grad, float **bias_grad,
                           float **vt_weight, float **vt_bias,
                           unsigned long int **graph_logic, unsigned long int **bias_logic,
                           float **fix_weight, float **fix_bias)
{
    /**
     *  Update weights and biases by gradient descent
     */

    // Update weights
    unsigned long int nthreads;
#pragma omp parallel
    {

        unsigned long int id = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        if (id == 0)
        {
            nthreads = omp_get_num_threads();
        }

#pragma omp barrier
        for (unsigned long int neuron_id = id; neuron_id < neuron_num; neuron_id += nthreads)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                vt_weight[neuron_id][neighbour_ind] = adam_beta1 * vt_weight[neuron_id][neighbour_ind] + (1.0 - adam_beta1) * weight_grad[neuron_id][neighbour_ind];
                if (graph_logic[neuron_id][neighbour_ind] == 1)
                {
                    weight[neuron_id][neighbour_ind] -= grad_alpha * vt_weight[neuron_id][neighbour_ind];
                }
            }

            //}

            // Update biases

            // for (int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
            //{
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                vt_bias[neuron_id][bias_ind] = adam_beta1 * vt_bias[neuron_id][bias_ind] + (1.0 - adam_beta1) * bias_grad[neuron_id][bias_ind];
                if (bias_logic[neuron_id][bias_ind] == 1)
                {
                    bias[neuron_id][bias_ind] -= grad_alpha * vt_bias[neuron_id][bias_ind];
                }
            }
        }
    }
}

void update_weight_bias_adamax(unsigned long int *neighbour_number, unsigned long int *bias_number,
                               float **weight, float **bias,
                               float **weight_grad, float **bias_grad,
                               float adam_beta1t, float **mt_weight, float **ut_weight,
                               float **mt_bias, float **ut_bias,
                               unsigned long int **graph_logic, unsigned long int **bias_logic,
                               float **fix_weight, float **fix_bias)
{
    /**
     *  Update weights with Adamax
     */
    float wg_max = calc_matrix_norm(weight_grad, neuron_num, neighbour_number);
    float bg_max = calc_matrix_norm(bias_grad, neuron_num, bias_number);

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
        {
            mt_weight[neuron_id][neighbour_ind] = adam_beta1 * mt_weight[neuron_id][neighbour_ind] +
                                                  (1.0 - adam_beta1) * weight_grad[neuron_id][neighbour_ind];

            ut_weight[neuron_id][neighbour_ind] =
                dmax(adam_beta2 * ut_weight[neuron_id][neighbour_ind],
                     fabs(weight_grad[neuron_id][neighbour_ind]));

            if (graph_logic[neuron_id][neighbour_ind] == 1)
            {
                weight[neuron_id][neighbour_ind] -= adam_alpha / (1.0 - adam_beta1t) *
                                                    mt_weight[neuron_id][neighbour_ind] / (ut_weight[neuron_id][neighbour_ind] + adam_eps);
            }
        }

        for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
        {
            mt_bias[neuron_id][bias_ind] = adam_beta1 * mt_bias[neuron_id][bias_ind] +
                                           (1.0 - adam_beta1) * bias_grad[neuron_id][bias_ind];

            ut_bias[neuron_id][bias_ind] =
                dmax(adam_beta2 * ut_bias[neuron_id][bias_ind],
                     fabs(bias_grad[neuron_id][bias_ind]));

            if (bias_logic[neuron_id][bias_ind] == 1)
            {
                bias[neuron_id][bias_ind] -= adam_alpha / (1.0 - adam_beta1t) *
                                             mt_bias[neuron_id][bias_ind] / (ut_bias[neuron_id][bias_ind] + adam_eps);
            }
        }
    }
}

void update_weight_bias_adam(unsigned long int *neighbour_number, unsigned long int *bias_number,
                             float **weight, float **bias,
                             float **weight_grad, float **bias_grad,
                             float adam_beta1t, float adam_beta2t,
                             float **mt_weight, float **vt_weight,
                             float **mth_weight, float **vth_weight,
                             float **mt_bias, float **vt_bias,
                             float **mth_bias, float **vth_bias,
                             unsigned long int **graph_logic, unsigned long int **bias_logic,
                             float **fix_weight, float **fix_bias)
{
    /**
     *  Update weights with Adam
     */

    unsigned long int nthreads;
#pragma omp parallel
    {

        unsigned long int id = omp_get_thread_num();
        nthreads = omp_get_num_threads();

        if (id == 0)
        {
            nthreads = omp_get_num_threads();
        }

#pragma omp barrier
        for (unsigned long int neuron_id = id; neuron_id < neuron_num; neuron_id += nthreads)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                mt_weight[neuron_id][neighbour_ind] = adam_beta1 * mt_weight[neuron_id][neighbour_ind] +
                                                      (1.0 - adam_beta1) * weight_grad[neuron_id][neighbour_ind];
                vt_weight[neuron_id][neighbour_ind] = adam_beta2 * vt_weight[neuron_id][neighbour_ind] +
                                                      (1.0 - adam_beta2) * powf(weight_grad[neuron_id][neighbour_ind], 2);
                mth_weight[neuron_id][neighbour_ind] = mt_weight[neuron_id][neighbour_ind] / (1.0 - adam_beta1t);
                vth_weight[neuron_id][neighbour_ind] = vt_weight[neuron_id][neighbour_ind] / (1.0 - adam_beta2t);

                if (graph_logic[neuron_id][neighbour_ind] == 1)
                {
                    weight[neuron_id][neighbour_ind] -= adam_alpha * mth_weight[neuron_id][neighbour_ind] /
                                                        (sqrtf(fabs(vth_weight[neuron_id][neighbour_ind])) + adam_eps);
                }
            }

            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                mt_bias[neuron_id][bias_ind] = adam_beta1 * mt_bias[neuron_id][bias_ind] +
                                               (1.0 - adam_beta1) * bias_grad[neuron_id][bias_ind];
                vt_bias[neuron_id][bias_ind] = adam_beta2 * vt_bias[neuron_id][bias_ind] +
                                               (1.0 - adam_beta2) * powf(bias_grad[neuron_id][bias_ind], 2);
                mth_bias[neuron_id][bias_ind] = mt_bias[neuron_id][bias_ind] / (1.0 - adam_beta1t);
                vth_bias[neuron_id][bias_ind] = vt_bias[neuron_id][bias_ind] / (1.0 - adam_beta2t);

                if (bias_logic[neuron_id][bias_ind] == 1)
                {
                    bias[neuron_id][bias_ind] -= adam_alpha * mth_bias[neuron_id][bias_ind] /
                                                 (sqrtf(fabs(vth_bias[neuron_id][bias_ind])) + adam_eps);
                }
            }
        }
    }
}

void update_weight_bias_radam(unsigned long int *neighbour_number, unsigned long int *bias_number,
                              float **weight, float **bias,
                              float **weight_grad, float **bias_grad,
                              float adam_beta1t, float adam_beta2t, float time_rho,
                              float **mt_weight, float **vt_weight,
                              float **mth_weight, float **vth_weight,
                              float **mt_bias, float **vt_bias,
                              float **mth_bias, float **vth_bias,
                              unsigned long int **graph_logic, unsigned long int **bias_logic,
                              float **fix_weight, float **fix_bias)
{
    /**
     *  Update weights with RAdam
     */
    float rho_infty = 2.0 / (1.0 - adam_beta2) - 1.0;
    float rhot = rho_infty - 2.0 * time_rho * adam_beta2t / (1.0 - adam_beta2t);
    float r_t = sqrt((rhot - 4.0) * (rhot - 2.0) * rho_infty / ((rho_infty - 4.0) * (rho_infty - 2.0) * rhot));

    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
        {
            mt_weight[neuron_id][neighbour_ind] = adam_beta1 * mt_weight[neuron_id][neighbour_ind] +
                                                  (1.0 - adam_beta1) * weight_grad[neuron_id][neighbour_ind];
            vt_weight[neuron_id][neighbour_ind] = adam_beta2 * vt_weight[neuron_id][neighbour_ind] +
                                                  (1.0 - adam_beta2) * pow(weight_grad[neuron_id][neighbour_ind], 2);
            mth_weight[neuron_id][neighbour_ind] = mt_weight[neuron_id][neighbour_ind] / (1.0 - adam_beta1t);

            if (graph_logic[neuron_id][neighbour_ind] == 1)
            {
                if (rhot > 4.0)
                {
                    vth_weight[neuron_id][neighbour_ind] = sqrt(vt_weight[neuron_id][neighbour_ind] / (1.0 - adam_beta2t));
                    weight[neuron_id][neighbour_ind] -= adam_alpha * mth_weight[neuron_id][neighbour_ind] /
                                                        vth_weight[neuron_id][neighbour_ind] *
                                                        r_t;
                }
                else
                {
                    weight[neuron_id][neighbour_ind] -= adam_alpha * mth_weight[neuron_id][neighbour_ind];
                }
            }
        }

        for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
        {
            mt_bias[neuron_id][bias_ind] = adam_beta1 * mt_bias[neuron_id][bias_ind] +
                                           (1.0 - adam_beta1) * bias_grad[neuron_id][bias_ind];
            vt_bias[neuron_id][bias_ind] = adam_beta2 * vt_bias[neuron_id][bias_ind] +
                                           (1.0 - adam_beta2) * pow(bias_grad[neuron_id][bias_ind], 2);
            mth_bias[neuron_id][bias_ind] = mt_bias[neuron_id][bias_ind] / (1.0 - adam_beta1t);

            if (bias_logic[neuron_id][bias_ind] == 1)
            {
                if (rhot > 4.0)
                {
                    vth_bias[neuron_id][bias_ind] = sqrt(vt_bias[neuron_id][bias_ind] / (1.0 - adam_beta2t));
                    bias[neuron_id][bias_ind] -= adam_alpha * mth_bias[neuron_id][bias_ind] /
                                                 vth_bias[neuron_id][bias_ind] *
                                                 r_t;
                }
                else
                {
                    bias[neuron_id][bias_ind] -= adam_alpha * mth_bias[neuron_id][bias_ind];
                }
            }
        }
    }
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

void calc_num_gradient_mini_batch(float **datas, unsigned long int mini_batch_len,
                                  unsigned long int *neighbour_number, unsigned long int *bias_number,
                                  unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                                  unsigned long int **parent_number, float **weight, float **bias,
                                  float **weight_num_grad, float **bias_num_grad,
                                  unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices)
{
    /**
     *  Calculate numerical gradient on a mini-batch
     */
    float **weight_temp = allocate_dmatrix(neuron_num, neighbour_number);
    float **bias_temp = allocate_dmatrix(neuron_num, bias_number);

    // Numerical gradient on the weights
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int j = 0; j < neighbour_number[neuron_id]; j++)
        {
            weight_num_grad[neuron_id][j] = 0.0;

            // pp
            for (unsigned long int neuron_id_c = 0; neuron_id_c < neuron_num; neuron_id_c++)
            {
                for (unsigned long int j_c = 0; j_c < neighbour_number[neuron_id_c]; j_c++)
                {
                    weight_temp[neuron_id_c][j_c] = weight[neuron_id_c][j_c];
                }
            }

            weight_temp[neuron_id][j] += numgrad_eps;
            float pp = calc_network_mini_batch(datas, mini_batch_len,
                                               neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                               parent_number, weight_temp, bias,
                                               dist, dist_max, dist_number, dist_indices);

            // mm
            weight_temp[neuron_id][j] -= 2.0 * numgrad_eps;

            float mm = calc_network_mini_batch(datas, mini_batch_len,
                                               neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                               parent_number, weight_temp, bias,
                                               dist, dist_max, dist_number, dist_indices);

            weight_num_grad[neuron_id][j] = (pp - mm) / (2.0 * numgrad_eps);
        }
    }

    // Numerical gradient on the bias
    for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
    {
        for (unsigned long int j = 0; j < bias_number[neuron_id]; j++)
        {
            bias_num_grad[neuron_id][j] = 0.0;
            // pp
            for (unsigned long int neuron_id_c = 0; neuron_id_c < neuron_num; neuron_id_c++)
            {
                for (unsigned long int j_c = 0; j_c < bias_number[neuron_id_c]; j_c++)
                {
                    bias_temp[neuron_id_c][j_c] = bias[neuron_id_c][j_c];
                }
            }
            bias_temp[neuron_id][j] += numgrad_eps;

            float pp = calc_network_mini_batch(datas, mini_batch_len,
                                               neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                               parent_number, weight, bias_temp,
                                               dist, dist_max, dist_number, dist_indices);

            bias_temp[neuron_id][j] -= 2.0 * numgrad_eps;

            float mm = calc_network_mini_batch(datas, mini_batch_len,
                                               neighbour_number, bias_number, activation_type, graph_n, graph_i,
                                               parent_number, weight, bias_temp,
                                               dist, dist_max, dist_number, dist_indices);

            bias_num_grad[neuron_id][j] = (pp - mm) / (2.0 * numgrad_eps);
        }
    }

    deallocate_dmatrix(weight_temp, neuron_num);
    deallocate_dmatrix(bias_temp, neuron_num);
}

float calc_diff_matrices(float **m1, float **m2, unsigned long int row_nums, unsigned long int *col_nums)
{
    float diff_norm = 0.0;

    for (unsigned long int i = 0; i < row_nums; i++)
    {
        // float line_diff_norm = 0.0;
        for (unsigned long int j = 0; j < col_nums[i]; j++)
        {
            float line_diff_norm = fabs(m1[i][j] - m2[i][j]);

            if (line_diff_norm > diff_norm)
            {
                diff_norm = line_diff_norm;
            }
        }
    }

    return diff_norm;
}

float calc_matrix_norm(float **m1, unsigned long int row_nums, unsigned long int *col_nums)
{
    float norm = 0.0;

    for (unsigned long int i = 0; i < row_nums; i++)
    {

        for (unsigned long int j = 0; j < col_nums[i]; j++)
        {
            float line_norm = fabs(m1[i][j]);
            if (line_norm > norm)
            {
                norm = line_norm;
            }
        }
    }

    return norm;
}

float calc_matrix_norm_l2(float **m1, unsigned long int row_nums, unsigned long int *col_nums)
{
    float norm = 0.0;
    float param_num = 0.0;

    for (unsigned long int i = 0; i < row_nums; i++)
    {

        for (unsigned long int j = 0; j < col_nums[i]; j++)
        {
            norm += fabs(m1[i][j]);
            param_num += 1.0;
        }
    }
    norm = sqrt(norm / param_num);

    return norm;
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
    int **returner = (int **)malloc(row_num * sizeof(int *));
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

void save_weight_bias(char filename[100], float **weight, float **bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float **mt_weight, float **mth_weight, float **vt_weight, float **vth_weight, float **ut_weight,
                      float **mt_bias, float **mth_bias, float **vt_bias, float **vth_bias, float **ut_bias,
                      float adam_beta1t, float adam_beta2t, float time_rho)
{
    FILE *f = fopen(filename, "w");
    if (f)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                fprintf(f, "%f ", weight[neuron_id][neighbour_ind]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                fprintf(f, "%f ", bias[neuron_id][bias_ind]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");

        // Optimizer parameters (on the weights and on the biases)
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                float mt_temp = mt_weight[neuron_id][neighbour_ind];
                float mth_temp = mth_weight[neuron_id][neighbour_ind];
                float vt_temp = vt_weight[neuron_id][neighbour_ind];
                float vth_temp = vth_weight[neuron_id][neighbour_ind];
                float ut_temp = ut_weight[neuron_id][neighbour_ind];
                fprintf(f, "%f %f %f %f %f ", mt_temp, mth_temp, vt_temp, vth_temp, ut_temp);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                float mt_temp = mt_bias[neuron_id][bias_ind];
                float mth_temp = mth_bias[neuron_id][bias_ind];
                float vt_temp = vt_bias[neuron_id][bias_ind];
                float vth_temp = vth_bias[neuron_id][bias_ind];
                float ut_temp = ut_bias[neuron_id][bias_ind];
                fprintf(f, "%f %f %f %f %f ", mt_temp, mth_temp, vt_temp, vth_temp, ut_temp);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "%f %f %f ", adam_beta1t, adam_beta2t, time_rho);
        fprintf(f, "\n");
    }
    else
    {
        program_failure("File write error in backup file!");
    }
    fclose(f);
}

void load_weight_bias(char filename[100], float **weight, float **bias,
                      unsigned long int neuron_num, unsigned long int *neighbour_number, unsigned long int *bias_number,
                      float **mt_weight, float **mth_weight, float **vt_weight, float **vth_weight, float **ut_weight,
                      float **mt_bias, float **mth_bias, float **vt_bias, float **vth_bias, float **ut_bias,
                      float adam_beta1t, float adam_beta2t, float time_rho)
{
    FILE *f = fopen(filename, "r");
    if (f)
    {
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                fscanf(f, "%f", &weight[neuron_id][neighbour_ind]);
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                fscanf(f, "%f ", &bias[neuron_id][bias_ind]);
            }
        }

        // Optimizer parameters (on the weights and on the biases)
        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int neighbour_ind = 0; neighbour_ind < neighbour_number[neuron_id]; neighbour_ind++)
            {
                float mt_temp;
                float mth_temp;
                float vt_temp;
                float vth_temp;
                float ut_temp;
                fscanf(f, "%f %f %f %f %f ", &mt_temp, &mth_temp, &vt_temp, &vth_temp, &ut_temp);
                mt_weight[neuron_id][neighbour_ind] = mt_temp;
                mth_weight[neuron_id][neighbour_ind] = mth_temp;
                vt_weight[neuron_id][neighbour_ind] = vt_temp;
                vth_weight[neuron_id][neighbour_ind] = vth_temp;
                ut_weight[neuron_id][neighbour_ind] = ut_temp;
            }
        }

        for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
        {
            for (unsigned long int bias_ind = 0; bias_ind < bias_number[neuron_id]; bias_ind++)
            {
                float mt_temp;
                float mth_temp;
                float vt_temp;
                float vth_temp;
                float ut_temp;
                fscanf(f, "%f %f %f %f %f ", &mt_temp, &mth_temp, &vt_temp, &vth_temp, &ut_temp);
                mt_bias[neuron_id][bias_ind] = mt_temp;
                mth_bias[neuron_id][bias_ind] = mth_temp;
                vt_bias[neuron_id][bias_ind] = vt_temp;
                vth_bias[neuron_id][bias_ind] = vth_temp;
                ut_bias[neuron_id][bias_ind] = ut_temp;
            }
        }
        fscanf(f, "%f %f %f ", &adam_beta1t, &adam_beta2t, &time_rho);
    }
    else
    {
        program_failure("File read error in backup file!");
    }
    fclose(f);
}

float matrix_norm(float **m, unsigned long int row_num, unsigned long int *col_num)
{
    float norm = 0.0;
    for (unsigned long int i = 0; i < row_num; i++)
    {
        for (unsigned long int j = 0; j < col_num[i]; j++)
        {
            norm += pow(m[i][j], 2);
        }
    }
    return sqrt(norm);
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

void make_predictions(float **datas, unsigned long int mini_batch_len,
                      unsigned long int *neighbour_number, unsigned long int *bias_number,
                      unsigned long int **activation_type, unsigned long int **graph_n, unsigned long int **graph_i,
                      unsigned long int **parent_number, float **weight, float **bias,
                      unsigned long int *dist, unsigned long int dist_max, unsigned long int *dist_number, unsigned long int **dist_indices,
                      float **predictions_mini_batch)
{
    /*
    Creating predictions on the whole dataset
    */

    for (unsigned long int i = 0; i < mini_batch_len; i++)
    {
        for (unsigned long int j = 0; j < output_num; j++)
        {
            predictions_mini_batch[i][j] = 0.0;
        }
    }

    unsigned long int nthreads;

#pragma omp parallel
    {
        float **input_value = allocate_dmatrix(neuron_num, bias_number);
        float *neuron_value = (float *)malloc(neuron_num * sizeof(float));
        float *output_value = (float *)malloc(output_num * sizeof(float));
        unsigned long int id;
        id = omp_get_thread_num();

        if (id == 0)
        {
            nthreads = omp_get_num_threads();
        }

#pragma omp barrier

        int iter_f;

        // Loop over the elements on the elements of the mini-batch
        for (unsigned long int data_index = id; data_index < mini_batch_len; data_index = data_index + nthreads)
        {

            // Copy the data for input_value with the input data
            for (unsigned long int neuron_id = 0; neuron_id < neuron_num; neuron_id++)
            {
                for (unsigned long int bias_id = 0; bias_id < bias_number[neuron_id]; bias_id++)
                {
                    input_value[neuron_id][bias_id] = 0.0;
                }
                if (neuron_id < input_num)
                {
                    input_value[neuron_id][0] = datas[data_index][neuron_id];
                }
            }

            // Calculate the network
            if (ff_optimization == 0)
            {
                iter_f = calc_network_one_sample(neighbour_number, bias_number,
                                                 activation_type, graph_n, graph_i, parent_number,
                                                 weight, bias, input_value, neuron_value);
            }
            else
            {
                iter_f = calc_network_one_sample_ff(neighbour_number, bias_number,
                                                    activation_type, graph_n, graph_i, parent_number,
                                                    weight, bias, input_value, neuron_value,
                                                    dist, dist_max, dist_number, dist_indices);
            }

            // Calculate the output
            for (unsigned long int i = 0; i < output_num; i++)
            {
                unsigned long int neuron_id = neuron_num - output_num + i;
                output_value[i] = neuron_value[neuron_id];
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

#pragma omp critical
            {
                for (unsigned long int j = 0; j < output_num; j++)
                {
                    predictions_mini_batch[data_index][j] = output_value[j];
                }
            }
        }
        deallocate_dmatrix(input_value, neuron_num);
        free(neuron_value);
        free(output_value);
    }
}
