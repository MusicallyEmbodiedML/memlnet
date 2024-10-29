#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <tuple>
#include <limits>
#include "npy.hpp"
#include <cassert>

using namespace std;

using NumpyFloat = float;


// Utility function for initializing random numbers
NumpyFloat rand_double() {
    return (static_cast<NumpyFloat>(rand()) / RAND_MAX) * 2.0 - 1.0;
}

NumpyFloat rand_double_up() {
    return (static_cast<NumpyFloat>(rand()) / RAND_MAX) ;
}

// Utility function to generate spiral data
void generate_spiral_data(int points_per_class, int num_classes, vector<vector<NumpyFloat>>& X, vector<vector<NumpyFloat>>& y) {
    for (int class_number = 0; class_number < num_classes; ++class_number) {
        NumpyFloat angle_start = class_number * 4.0;
        for (int i = 0; i < points_per_class; ++i) {
            NumpyFloat t = (NumpyFloat)i / points_per_class;
            NumpyFloat r = t * 1.0;
            NumpyFloat theta = angle_start + t * 4.0 * M_PI;
            NumpyFloat x1 = r * sin(theta);
            NumpyFloat x2 = r * cos(theta);
            X.push_back({x1, x2});
            vector<NumpyFloat> class_label(num_classes, 0.0);
            class_label[class_number] = 1.0;
            y.push_back(class_label);
        }
    }
}


std::tuple<NumpyFloat, NumpyFloat, NumpyFloat> analyzeMatrix(const std::vector<std::vector<NumpyFloat>>& matrix) {
    NumpyFloat sum = 0;
    NumpyFloat minValue = std::numeric_limits<NumpyFloat>::max();
    NumpyFloat maxValue = std::numeric_limits<NumpyFloat>::lowest();
    std::size_t count = 0;

    for (const auto& row : matrix) {
        for (const auto& value : row) {
            sum += value;
            if (value < minValue) {
                minValue = value;
            }
            if (value > maxValue) {
                maxValue = value;
            }
            ++count;
        }
    }

    NumpyFloat average = count > 0 ? sum / count : 0.0;
    return std::make_tuple(average, minValue, maxValue);
}


class Layer_Input {
public:
    vector<vector<NumpyFloat>> output;

    // Forward pass for the input layer
    void forward(const vector<vector<NumpyFloat>>& inputs, bool training) {
        // Just set the inputs as the output for the input layer
        output = inputs;
    }
};

// Dense layer with L1 and L2 regularization
class Layer_Dense {
public:
    vector<vector<NumpyFloat>> weights;
    vector<NumpyFloat> biases;
    vector<vector<NumpyFloat>> inputs;
    vector<vector<NumpyFloat>> output;
    vector<vector<NumpyFloat>> dweights;
    vector<NumpyFloat> dbiases;
    vector<vector<NumpyFloat>> dinputs;

    // For Adam optimizer
    vector<vector<NumpyFloat>> weight_momentums;
    vector<vector<NumpyFloat>> weight_caches;
    vector<NumpyFloat> bias_momentums;
    vector<NumpyFloat> bias_caches;

    // Regularization parameters
    NumpyFloat weight_regularizer_l1;
    NumpyFloat weight_regularizer_l2;
    NumpyFloat bias_regularizer_l1;
    NumpyFloat bias_regularizer_l2;

    Layer_Dense(int n_inputs, int n_neurons,
                NumpyFloat weight_regularizer_l1 = 0.0, NumpyFloat weight_regularizer_l2 = 0.0,
                NumpyFloat bias_regularizer_l1 = 0.0, NumpyFloat bias_regularizer_l2 = 0.0)
        : weight_regularizer_l1(weight_regularizer_l1),
          weight_regularizer_l2(weight_regularizer_l2),
          bias_regularizer_l1(bias_regularizer_l1),
          bias_regularizer_l2(bias_regularizer_l2) {

        // Initialize weights with small random values and biases with zero
        weights.resize(n_inputs, vector<NumpyFloat>(n_neurons));
        NumpyFloat start_val = -0.01;
        for (int i = 0; i < n_inputs; ++i) {
            for (int j = 0; j < n_neurons; ++j) {
                weights[i][j] = start_val; //(rand_double_up() * rand_double_up() * rand_double_up()) * 0.02 - 0.01;
                start_val += 0.00001;
            }
        }
        biases.resize(n_neurons, 0.0);

        // Initialize Adam optimizer terms
        weight_momentums.resize(n_inputs, vector<NumpyFloat>(n_neurons, 0.0));
        weight_caches.resize(n_inputs, vector<NumpyFloat>(n_neurons, 0.0));
        bias_momentums.resize(n_neurons, 0.0);
        bias_caches.resize(n_neurons, 0.0);
    }

    // Forward pass
    void forward(const vector<vector<NumpyFloat>>& inputs) {
        this->inputs = inputs;
        output.resize(inputs.size(), vector<NumpyFloat>(weights[0].size()));

        for (size_t i = 0; i < inputs.size(); ++i) {  // batch size
            for (size_t j = 0; j < weights[0].size(); ++j) {  // output size
                output[i][j] = biases[j];
                for (size_t k = 0; k < weights.size(); ++k) {  // input size
                    output[i][j] += inputs[i][k] * weights[k][j];
                }
            }
        }
    }

    // Backward pass
    void backward(const vector<vector<NumpyFloat>>& dvalues) {
        dweights.resize(weights.size(), vector<NumpyFloat>(weights[0].size()));
        dbiases.resize(weights[0].size());
        dinputs.resize(inputs.size(), vector<NumpyFloat>(inputs[0].size()));

        //std::cout << "DVALUES: {" << std::endl;
        // Calculate dweights and dbiases
        for (size_t i = 0; i < inputs.size(); ++i) {
            //std::cout << "    { ";
            for (size_t j = 0; j < weights[0].size(); ++j) {
                dbiases[j] += dvalues[i][j];
                for (size_t k = 0; k < weights.size(); ++k) {
                    dweights[k][j] += inputs[i][k] * dvalues[i][j];
                }
                //std::cout << dvalues[i][j] << ", ";
            }
            //std::cout << "}," << std::endl;
        }
        //std::cout << "}" << std::endl;
        std::cout << "DBIASES: { ";
        for (size_t i = 0; i < biases.size(); ++i) {
            std::cout << dbiases[i] << ", ";
        }
        std::cout << " }" << std::endl;

        // Apply L1 regularization on weights
        if (weight_regularizer_l1 > 0.0) {
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[0].size(); ++j) {
                    dweights[i][j] += weight_regularizer_l1 * (weights[i][j] > 0 ? 1 : -1);
                }
            }
        }

        // Apply L2 regularization on weights
        if (weight_regularizer_l2 > 0.0) {
            for (size_t i = 0; i < weights.size(); ++i) {
                for (size_t j = 0; j < weights[0].size(); ++j) {
                    dweights[i][j] += 2 * weight_regularizer_l2 * weights[i][j];
                }
            }
        }

        // Apply L1 regularization on biases
        if (bias_regularizer_l1 > 0.0) {
            for (size_t i = 0; i < biases.size(); ++i) {
                dbiases[i] += bias_regularizer_l1 * (biases[i] > 0 ? 1 : -1);
            }
        }

        // Apply L2 regularization on biases
        if (bias_regularizer_l2 > 0.0) {
            for (size_t i = 0; i < biases.size(); ++i) {
                dbiases[i] += 2 * bias_regularizer_l2 * biases[i];
            }
        }

        // Calculate dinputs
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < weights.size(); ++j) {
                dinputs[i][j] = 0.0;
                for (size_t k = 0; k < weights[0].size(); ++k) {
                    dinputs[i][j] += dvalues[i][k] * weights[j][k];
                }
            }
        }
    }
};

// ReLU activation
class Activation_ReLU {
public:
    vector<vector<NumpyFloat>> output;
    vector<vector<NumpyFloat>> dinputs;

    // Forward pass
    void forward(const vector<vector<NumpyFloat>>& inputs) {
        output.resize(inputs.size(), vector<NumpyFloat>(inputs[0].size()));

        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[0].size(); ++j) {
                output[i][j] = max(0.0f, inputs[i][j]);
            }
        }
    }

    // Backward pass
    void backward(const vector<vector<NumpyFloat>>& dvalues, const vector<vector<NumpyFloat>>& inputs) {
        dinputs.resize(dvalues.size(), vector<NumpyFloat>(dvalues[0].size()));

        for (size_t i = 0; i < dvalues.size(); ++i) {
            for (size_t j = 0; j < dvalues[0].size(); ++j) {
                dinputs[i][j] = inputs[i][j] > 0.0 ? dvalues[i][j] : 0.0;
            }
        }
    }
};


class Loss {
public:
    // Regularization loss calculation
    NumpyFloat regularization_loss(const Layer_Dense& layer) {
        NumpyFloat regularization_loss = 0.0;

        // L1 regularization - weights
        if (layer.weight_regularizer_l1 > 0) {
            NumpyFloat l1_weight_loss = 0.0;
            for (const auto& row : layer.weights) {
                for (NumpyFloat weight : row) {
                    l1_weight_loss += std::abs(weight);
                }
            }
            regularization_loss += layer.weight_regularizer_l1 * l1_weight_loss;
        }

        // L2 regularization - weights
        if (layer.weight_regularizer_l2 > 0) {
            NumpyFloat l2_weight_loss = 0.0;
            for (const auto& row : layer.weights) {
                for (NumpyFloat weight : row) {
                    l2_weight_loss += weight * weight;
                }
            }
            regularization_loss += layer.weight_regularizer_l2 * l2_weight_loss;
        }

        // L1 regularization - biases
        if (layer.bias_regularizer_l1 > 0) {
            NumpyFloat l1_bias_loss = 0.0;
            for (NumpyFloat bias : layer.biases) {
                l1_bias_loss += std::abs(bias);
            }
            regularization_loss += layer.bias_regularizer_l1 * l1_bias_loss;
        }

        // L2 regularization - biases
        if (layer.bias_regularizer_l2 > 0) {
            NumpyFloat l2_bias_loss = 0.0;
            for (NumpyFloat bias : layer.biases) {
                l2_bias_loss += bias * bias;
            }
            regularization_loss += layer.bias_regularizer_l2 * l2_bias_loss;
        }

        return regularization_loss;
    }

    // Calculates the data loss given model output and ground truth values
    NumpyFloat calculate(const std::vector<std::vector<NumpyFloat>>& output, const std::vector<std::vector<NumpyFloat>>& y) {
        auto sample_losses = forward(output, y);
        NumpyFloat data_loss = std::accumulate(sample_losses.begin(), sample_losses.end(), 0.0) / sample_losses.size();
        return data_loss;
    }

protected:
    // Sample forward method, to be overridden
    virtual std::vector<NumpyFloat> forward(const std::vector<std::vector<NumpyFloat>>& output, const std::vector<std::vector<NumpyFloat>>& y) = 0;
};


class Loss_CategoricalCrossentropy : public Loss {
public:
    vector<vector<NumpyFloat>> dinputs;

    // Forward pass
    vector<NumpyFloat> forward(const vector<vector<NumpyFloat>>& y_pred, const vector<vector<NumpyFloat>>& y_true) {
        size_t samples = y_pred.size();
        vector<NumpyFloat> correct_confidences(samples);

        // Clip predictions to prevent division by 0 and dragging log towards infinity
        vector<vector<NumpyFloat>> y_pred_clipped = y_pred;
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < y_pred[i].size(); ++j) {
                y_pred_clipped[i][j] = max(1e-7f, min(1.f - 1e-7f, y_pred[i][j]));
            }
        }

        // Calculate confidence for the correct class
        for (size_t i = 0; i < samples; ++i) {
            if (y_true[i].size() == 1) {  // Sparse categorical labels
                size_t correct_class = distance(y_true[i].begin(), find(y_true[i].begin(), y_true[i].end(), 1.0));
                correct_confidences[i] = y_pred_clipped[i][correct_class];
            } else {  // One-hot encoded labels
                NumpyFloat sum = 0.0;
                for (size_t j = 0; j < y_true[i].size(); ++j) {
                    sum += y_pred_clipped[i][j] * y_true[i][j];
                }
                correct_confidences[i] = sum;
            }
        }

        // Calculate losses
        vector<NumpyFloat> negative_log_likelihoods(samples);
        for (size_t i = 0; i < samples; ++i) {
            negative_log_likelihoods[i] = -log(correct_confidences[i]);
        }
        return negative_log_likelihoods;
    }

    // Backward pass
    vector<vector<NumpyFloat>> backward(const vector<vector<NumpyFloat>>& dvalues, const vector<vector<NumpyFloat>>& y_true) {
        size_t samples = dvalues.size();
        size_t labels = dvalues[0].size();

        // If labels are sparse, convert them to one-hot encoding
        vector<vector<NumpyFloat>> y_true_one_hot = y_true;
        if (y_true[0].size() == 1) {
            y_true_one_hot.resize(samples, vector<NumpyFloat>(labels, 0.0));
            for (size_t i = 0; i < samples; ++i) {
                size_t correct_class = distance(y_true[i].begin(), find(y_true[i].begin(), y_true[i].end(), 1.0));
                y_true_one_hot[i][correct_class] = 1.0;
            }
        }

        // Initialize gradients (dinputs) with zeros
        dinputs.resize(samples, vector<NumpyFloat>(labels, 0.0));

        // Compute gradients
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < labels; ++j) {
                dinputs[i][j] = -y_true_one_hot[i][j] / dvalues[i][j];
            }
        }

        // Normalize gradient by number of samples
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < labels; ++j) {
                dinputs[i][j] /= samples;
            }
        }

        return dinputs;
    }
};


class Activation_Softmax {
public:
    // Forward pass
    void forward(const std::vector<std::vector<NumpyFloat>>& inputs) {
        // Store inputs
        this->inputs = inputs;
        size_t num_samples = inputs.size();
        size_t num_classes = inputs[0].size();
        
        // Resize output vector
        output.resize(num_samples, std::vector<NumpyFloat>(num_classes));

        // Get unnormalized probabilities
        std::vector<std::vector<NumpyFloat>> exp_values(num_samples, std::vector<NumpyFloat>(num_classes));
        for (size_t i = 0; i < num_samples; ++i) {
            NumpyFloat max_input = *std::max_element(inputs[i].begin(), inputs[i].end());
            for (size_t j = 0; j < num_classes; ++j) {
                exp_values[i][j] = std::exp(inputs[i][j] - max_input);
            }
        }

        // Normalize to get probabilities
        for (size_t i = 0; i < num_samples; ++i) {
            NumpyFloat sum_exp = std::accumulate(exp_values[i].begin(), exp_values[i].end(), 0.0);
            for (size_t j = 0; j < num_classes; ++j) {
                output[i][j] = exp_values[i][j] / sum_exp;
            }
        }
    }

    // Backward pass
    void backward(const std::vector<std::vector<NumpyFloat>>& dvalues) {
        size_t num_samples = output.size();
        size_t num_classes = output[0].size();
        
        // Initialize gradient array
        dinputs.resize(num_samples, std::vector<NumpyFloat>(num_classes, 0.0));

        for (size_t i = 0; i < num_samples; ++i) {
            // Flatten output array
            std::vector<NumpyFloat> single_output = output[i];

            // Calculate Jacobian matrix of the output
            for (size_t j = 0; j < num_classes; ++j) {
                NumpyFloat diag = single_output[j];
                for (size_t k = 0; k < num_classes; ++k) {
                    if (j == k) {
                        dinputs[i][j] += diag * (1 - diag) * dvalues[i][k]; // Diagonal elements
                    } else {
                        dinputs[i][j] -= diag * single_output[k] * dvalues[i][k]; // Off-diagonal elements
                    }
                }
            }
        }
    }

    // Get the output
    const std::vector<std::vector<NumpyFloat>>& getOutput() const {
        return output;
    }

private:
    std::vector<std::vector<NumpyFloat>> inputs;
    std::vector<std::vector<NumpyFloat>> output;
    std::vector<std::vector<NumpyFloat>> dinputs;
};






class Activation_Softmax_Loss_CategoricalCrossentropy {
public:
    vector<vector<NumpyFloat>> dinputs;
    Activation_Softmax *activation_ptr;
    Loss_CategoricalCrossentropy *loss_ptr;
    std::vector<std::vector<NumpyFloat>> output;

    Activation_Softmax_Loss_CategoricalCrossentropy(
        Activation_Softmax *activation,
        Loss_CategoricalCrossentropy *loss
    ) : activation_ptr(activation),
        loss_ptr(loss)
    {}

    // Forward pass
    NumpyFloat forward(const vector<vector<NumpyFloat>>& inputs,
                                   const vector<vector<NumpyFloat>>& y_true) {
        
        //
        activation_ptr->forward(inputs);
        output = activation_ptr->getOutput();
        return loss_ptr->calculate(output, y_true);
    }

    // Backward pass
    vector<vector<NumpyFloat>> backward(const vector<vector<NumpyFloat>>& dvalues, const vector<vector<NumpyFloat>>& y_true) {
        size_t samples = dvalues.size();
        size_t classes = dvalues[0].size();

        // Convert one-hot encoded y_true to class labels if necessary
        vector<size_t> y_true_labels(samples);
        if (y_true[0].size() == classes) { // One-hot encoded labels
            for (size_t i = 0; i < samples; ++i) {
                auto it = max_element(y_true[i].begin(), y_true[i].end());
                y_true_labels[i] = distance(y_true[i].begin(), it);
            }
        } else { // Labels are already discrete
            for (size_t i = 0; i < samples; ++i) {
                y_true_labels[i] = static_cast<size_t>(y_true[i][0]);
            }
        }

        // Copy dvalues to dinputs
        dinputs = dvalues;

        // Calculate gradient: subtract 1 from the predicted class and normalize
        for (size_t i = 0; i < samples; ++i) {
            dinputs[i][y_true_labels[i]] -= 1;
        }

        // Normalize gradient by the number of samples
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < classes; ++j) {
                dinputs[i][j] /= samples;
            }
        }

        return dinputs;
    }
};


class Optimizer_RMSprop {
public:
    NumpyFloat learning_rate;
    NumpyFloat current_learning_rate;
    NumpyFloat decay;
    NumpyFloat epsilon;
    NumpyFloat rho;
    int iterations;

    // Constructor
    Optimizer_RMSprop(NumpyFloat learning_rate = 0.001, NumpyFloat decay = 0.0, NumpyFloat epsilon = 1e-7, NumpyFloat rho = 0.9)
        : learning_rate(learning_rate), current_learning_rate(learning_rate), decay(decay), epsilon(epsilon), rho(rho), iterations(0) {}

    // Call once before any parameter updates
    void pre_update_params() {
        if (decay > 0.0) {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }

    // Update parameters
    void update_params(Layer_Dense& layer) {
        // If layer does not contain cache arrays, create them filled with zeros
        if (layer.weight_caches.empty()) {
            layer.weight_caches.resize(layer.weights.size(), vector<NumpyFloat>(layer.weights[0].size(), 0.0));
            layer.bias_caches.resize(layer.biases.size(), 0.0);
        }

        // Update cache with squared current gradients
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            for (size_t j = 0; j < layer.weights[0].size(); ++j) {
                layer.weight_caches[i][j] = rho * layer.weight_caches[i][j] + (1 - rho) * pow(layer.dweights[i][j], 2);
            }
        }

        for (size_t i = 0; i < layer.biases.size(); ++i) {
            layer.bias_caches[i] = rho * layer.bias_caches[i] + (1 - rho) * pow(layer.dbiases[i], 2);
        }

        // Vanilla SGD parameter update + normalization with square rooted cache
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            for (size_t j = 0; j < layer.weights[0].size(); ++j) {
                layer.weights[i][j] += -current_learning_rate * layer.dweights[i][j] / (sqrt(layer.weight_caches[i][j]) + epsilon);
            }
        }

        for (size_t i = 0; i < layer.biases.size(); ++i) {
            layer.biases[i] += -current_learning_rate * layer.dbiases[i] / (sqrt(layer.bias_caches[i]) + epsilon);
        }
    }

    // Call once after any parameter updates
    void post_update_params() {
        ++iterations;
    }
};

// Adam optimizer
// class Optimizer_Adam {
// public:
//     NumpyFloat learning_rate;
//     NumpyFloat beta_1;
//     NumpyFloat beta_2;
//     NumpyFloat epsilon;
//     int t;
//
//     Optimizer_Adam(NumpyFloat lr = 0.001, NumpyFloat beta1 = 0.9, NumpyFloat beta2 = 0.999, NumpyFloat eps = 1e-7) {
//         learning_rate = lr;
//         beta_1 = beta1;
//         beta_2 = beta2;
//         epsilon = eps;
//         t = 0;
//     }
//
//     void update_params(Layer_Dense& layer) {
//         t++;
//
//         for (size_t i = 0; i < layer.weights.size(); ++i) {
//             for (size_t j = 0; j < layer.weights[0].size(); ++j) {
//                 // Update weight momentums and caches
//                 layer.weight_momentums[i][j] = beta_1 * layer.weight_momentums[i][j] + (1 - beta_1) * layer.dweights[i][j];
//                 layer.weight_caches[i][j] = beta_2 * layer.weight_caches[i][j] + (1 - beta_2) * pow(layer.dweights[i][j], 2);
//
//                 // Corrected momentums and caches
//                 NumpyFloat corrected_weight_momentum = layer.weight_momentums[i][j] / (1 - pow(beta_1, t));
//                 NumpyFloat corrected_weight_cache = layer.weight_caches[i][j] / (1 - pow(beta_2, t));
//
//                 // Update weights
//                 layer.weights[i][j] -= learning_rate * corrected_weight_momentum / (sqrt(corrected_weight_cache) + epsilon);
//             }
//         }
//
//         // Update biases in a similar fashion
//         for (size_t j = 0; j < layer.biases.size(); ++j) {
//             // Update bias momentums and caches
//             layer.bias_momentums[j] = beta_1 * layer.bias_momentums[j] + (1 - beta_1) * layer.dbiases[j];
//             layer.bias_caches[j] = beta_2 * layer.bias_caches[j] + (1 - beta_2) * pow(layer.dbiases[j], 2);
//
//             // Corrected momentums and caches
//             NumpyFloat corrected_bias_momentum = layer.bias_momentums[j] / (1 - pow(beta_1, t));
//             NumpyFloat corrected_bias_cache = layer.bias_caches[j] / (1 - pow(beta_2, t));
//
//             // Update biases
//             layer.biases[j] -= learning_rate * corrected_bias_momentum / (sqrt(corrected_bias_cache) + epsilon);
//         }
//     }
// };

class Optimizer_Adam {
public:
    NumpyFloat learning_rate;
    NumpyFloat current_learning_rate;
    NumpyFloat decay;
    NumpyFloat epsilon;
    NumpyFloat beta_1;
    NumpyFloat beta_2;
    int iterations;

    Optimizer_Adam(NumpyFloat learning_rate = 0.001, NumpyFloat decay = 0., NumpyFloat epsilon = 1e-7,
                   NumpyFloat beta_1 = 0.9, NumpyFloat beta_2 = 0.999)
        : learning_rate(learning_rate), current_learning_rate(learning_rate),
          decay(decay), epsilon(epsilon), beta_1(beta_1), beta_2(beta_2), iterations(0) {}

    // Call once before any parameter updates
    void pre_update_params() {
        if (decay != 0.0) {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }

    // Update parameters
    void update_params(Layer_Dense& layer) {
        // Update momentum with current gradients
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                layer.weight_momentums[i][j] = beta_1 * layer.weight_momentums[i][j] + (1 - beta_1) * layer.dweights[i][j];
                layer.weight_caches[i][j] = beta_2 * layer.weight_caches[i][j] + (1 - beta_2) * std::pow(layer.dweights[i][j], 2);
            }
        }

        std::cout << "dBIAS: { ";
        for (size_t i = 0; i < layer.biases.size(); ++i) {
            layer.bias_momentums[i] = beta_1 * layer.bias_momentums[i] + (1 - beta_1) * layer.dbiases[i];
            layer.bias_caches[i] = beta_2 * layer.bias_caches[i] + (1 - beta_2) * std::pow(layer.dbiases[i], 2);
            std::cout << layer.dbiases[i] << ", ";
        }
        std::cout << " }" << std::endl;

        // Get corrected momentums and caches
        NumpyFloat momentum_sum = 0;
        NumpyFloat momentum_minValue = std::numeric_limits<NumpyFloat>::max();
        NumpyFloat momentum_maxValue = std::numeric_limits<NumpyFloat>::lowest();
        NumpyFloat cache_sum = 0;
        NumpyFloat cache_minValue = std::numeric_limits<NumpyFloat>::max();
        NumpyFloat cache_maxValue = std::numeric_limits<NumpyFloat>::lowest();
        std::size_t count = 0;

        std::cout << "WEIGHT CHANGE: {" << std::endl;
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            std::cout <<  "    { ";
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                NumpyFloat weight_momentum_corrected = layer.weight_momentums[i][j] / (1 - std::pow(beta_1, iterations + 1));
                NumpyFloat weight_cache_corrected = layer.weight_caches[i][j] / (1 - std::pow(beta_2, iterations + 1));
                NumpyFloat weight_change = -current_learning_rate * weight_momentum_corrected / (std::sqrt(weight_cache_corrected) + epsilon);
                layer.weights[i][j] += weight_change;
                std::cout << weight_change << ", ";

                // Calculate vector stats
                momentum_sum += weight_momentum_corrected;
                if (weight_momentum_corrected < momentum_minValue) {
                    momentum_minValue = weight_momentum_corrected;
                }
                if (weight_momentum_corrected > momentum_maxValue) {
                    momentum_maxValue = weight_momentum_corrected;
                }
                cache_sum += weight_cache_corrected;
                if (weight_cache_corrected < cache_minValue) {
                    cache_minValue = weight_cache_corrected;
                }
                if (weight_cache_corrected > cache_maxValue) {
                    cache_maxValue = weight_cache_corrected;
                }
                ++count;
            }
            std::cout << "}," << std::endl; 
        }
        std::cout << "}" << std::endl;
        NumpyFloat momentum_average = (count > 0) ? momentum_sum / count : 0;
        NumpyFloat cache_average = (count > 0) ? cache_sum / count : 0;

        // std::cout << "MOMENTUM" << std::endl;
        // std::cout << "  --Average: " << momentum_average << std::endl;
        // std::cout << "  --Min: " << momentum_minValue << std::endl;
        // std::cout << "  --Max: " << momentum_maxValue << std::endl;
        // std::cout << "CACHE" << std::endl;
        // std::cout << "  --Average: " << cache_average << std::endl;
        // std::cout << "  --Min: " << cache_minValue << std::endl;

        std::cout << "BIAS CHANGE: { ";
        for (size_t i = 0; i < layer.biases.size(); ++i) {
            NumpyFloat bias_momentum_corrected = layer.bias_momentums[i] / (1 - std::pow(beta_1, iterations + 1));
            NumpyFloat bias_cache_corrected = layer.bias_caches[i] / (1 - std::pow(beta_2, iterations + 1));
            NumpyFloat bias_change = -current_learning_rate * bias_momentum_corrected / (std::sqrt(bias_cache_corrected) + epsilon);
            layer.biases[i] += bias_change;
            std::cout << bias_change << ", ";
        }
        std::cout << "}" << std::endl;
    }

    // Call once after any parameter updates
    void post_update_params() {
        ++iterations;
    }
};

//extern vector<vector<NumpyFloat>> X;
//extern vector<vector<NumpyFloat>> y;
//#include "main_ch14_data.inc"

void load_2darray(std::vector<std::vector<NumpyFloat>> &var, std::string var_path) {

    auto d = npy::read_npy<NumpyFloat>(var_path);

    assert(d.fortran_order == false);
    
    if (d.shape.size() == 2) {
        size_t accum = 0;
        var.resize(d.shape[0]);
        for (unsigned int r = 0; r < d.shape[0]; r++) {
            var[r].resize(d.shape[1]);
            for (unsigned int c = 0; c < d.shape[1]; c++) {
                var[r][c] = d.data[accum++];
            }
        }
    }
}


void load_1darray(std::vector<NumpyFloat> &var, std::string var_path) {

    auto d = npy::read_npy<NumpyFloat>(var_path);

    assert(d.fortran_order == false);
    
    if (d.shape.size() == 1) {
        var = d.data;
    }
}


void test_dense() {
    vector< vector<NumpyFloat> > x {
        {1, 2, 3}, {4, 5, 6}
    };
    vector< vector<NumpyFloat> > w {
        {0.01, 0.02, 0.03, 0.04, 0.05},
        {0.1, 0.2, 0.3, 0.4, 0.5},
        {1, 2, 3, 4, 5}
    };
    vector<NumpyFloat> b { 0, 0, 0, 0, 0 };

    Layer_Dense dense(3, 5);
    dense.weights = w;
    dense.biases = b;
    dense.forward(x);
    cout << "Dense test finished." << endl;
}



int main() {

    test_dense();

    // Generate spiral data
    vector<vector<NumpyFloat>> X;
    vector<vector<NumpyFloat>> y;
    //generate_spiral_data(100, 3, X, y);  // 100 points per class, 3 classes
    //std::vector<std::vector<NumpyFloat>> X { {0, 0}, {1, 4}, {5, 6}, {7,2} };  // Example input
    // std::vector<std::vector<NumpyFloat>>  y { {0}, {1}, {1} ,{0}};  // Example labels

    // Load spiral data from numpy
    load_2darray(X, "data/X.npy");
    load_2darray(y, "data/y_2d.npy");

    // Create the first Dense layer with 2 inputs and 64 neurons, and L2 regularization
    Layer_Dense dense1(2, 64, 0.000, 5e-4, 0, 5e-4);

    // Create ReLU activation for first layer
    Activation_ReLU relu1;

    // Create the second Dense layer with 64 inputs and 3 neurons (3 output classes), and L2 regularization

    Layer_Dense dense2(64, 3);

    // Load weight init from numpy
    load_2darray(dense1.weights, "data/dense1.weights.npy");
    load_1darray(dense1.biases, "data/dense1.biases.npy");
    load_2darray(dense2.weights, "data/dense2.weights.npy");
    load_1darray(dense2.biases, "data/dense2.biases.npy");

    // Create Adam optimizer
    Optimizer_Adam adam_optimizer(0.02, 5e-7);

    //Optimizer_RMSprop rms_opt;

    // Create Softmax classifier's combined loss and activation
    Loss_CategoricalCrossentropy loss;
    Activation_Softmax activation;
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation(&activation, &loss);

    // Training for 1000 epochs
    for (int epoch = 0; epoch < 200; ++epoch) {
        // Forward pass through first dense layer
        dense1.forward(X);

        // Forward pass through ReLU activation for first layer
        relu1.forward(dense1.output);

        // Forward pass through second dense layer
        dense2.forward(relu1.output);


        // Calculate loss
        NumpyFloat data_loss = loss_activation.forward(dense2.output, y);

        // Calculate regularization penalty
        NumpyFloat regularization_loss = loss_activation.loss_ptr->regularization_loss(dense1) +
            loss_activation.loss_ptr->regularization_loss(dense2);

        NumpyFloat loss = data_loss + regularization_loss;

        // Backward pass through loss function
        vector<vector<NumpyFloat>> dloss = loss_activation.backward(loss_activation.output, y);
#if 0
        cout << "DLOSS: {" << endl;
        for (unsigned int i = 0; i < dloss.size(); i++) {
            cout << "    { ";
            for (unsigned int j = 0; j < dloss[i].size(); j++) {
                cout << dloss[i][j] << ", ";
            }
            cout << "}," << endl;
        }
#endif

        // Backward pass through second Dense layer
        dense2.backward(loss_activation.dinputs);
        relu1.backward(dense2.dinputs, dense1.output);
        dense1.backward(relu1.dinputs);


        // Update parameters using Adam optimizer for both layers
        adam_optimizer.pre_update_params();
        adam_optimizer.update_params(dense1);
        adam_optimizer.update_params(dense2);
        adam_optimizer.post_update_params();
        //rms_opt.update_params(dense1);
        //rms_opt.update_params(dense2);
        

        // Print loss every 100 epochs
        if (epoch % 1 == 0) {
            printf("epoch: %d, loss: %f (data_loss: %f, reg_loss: %f), lr: %f\n",
                epoch,
                loss,
                data_loss,
                regularization_loss,
                adam_optimizer.current_learning_rate
                //rms_opt.current_learning_rate
            );
        }
    }

    return 0;
}
