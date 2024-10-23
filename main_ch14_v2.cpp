#include <iostream>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

// Constants for dataset size
const size_t SAMPLES = 100;
const size_t CLASSES = 3;
const size_t INPUT_DIM = 2;
const size_t HIDDEN_NEURONS = 64;
const size_t OUTPUT_NEURONS = 3;

// Function to generate spiral data
template<size_t Samples, size_t Classes>
void spiral_data(std::array<std::array<double, INPUT_DIM>, Samples * Classes>& X,
                 std::array<size_t, Samples * Classes>& y) {
    size_t ix = 0;
    double radius, theta;
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (size_t class_number = 0; class_number < Classes; ++class_number) {
        for (size_t i = 0; i < Samples; ++i) {
            radius = static_cast<double>(i) / Samples;
            theta = (class_number * 4.0) + (radius * 4.0) + dist(gen) * 0.2;
            X[ix][0] = radius * std::sin(theta);
            X[ix][1] = radius * std::cos(theta);
            y[ix] = class_number;
            ++ix;
        }
    }
}

// Activation ReLU
template<size_t BatchSize, size_t InputSize>
struct ActivationReLU {
    std::array<std::array<double, InputSize>, BatchSize> inputs;
    std::array<std::array<double, InputSize>, BatchSize> output;
    std::array<std::array<double, InputSize>, BatchSize> dinputs;

    void forward(const std::array<std::array<double, InputSize>, BatchSize>& inputs_) {
        inputs = inputs_;
        for (size_t i = 0; i < BatchSize; ++i) {
            for (size_t j = 0; j < InputSize; ++j) {
                output[i][j] = std::max(0.0, inputs[i][j]);
            }
        }
    }

    void backward(const std::array<std::array<double, InputSize>, BatchSize>& dvalues) {
        dinputs = dvalues;
        for (size_t i = 0; i < BatchSize; ++i) {
            for (size_t j = 0; j < InputSize; ++j) {
                if (inputs[i][j] <= 0) {
                    dinputs[i][j] = 0.0;
                }
            }
        }
    }
};

// Dense Layer with Regularization
template<size_t InputSize, size_t Neurons>
struct LayerDense {
    std::array<std::array<double, Neurons>, InputSize> weights;
    std::array<double, Neurons> biases;

    // Adam optimizer parameters
    std::array<std::array<double, Neurons>, InputSize> weight_momentums;
    std::array<std::array<double, Neurons>, InputSize> weight_cache;
    std::array<double, Neurons> bias_momentums;
    std::array<double, Neurons> bias_cache;

    // Regularization parameters
    double weight_regularizer_l1;
    double weight_regularizer_l2;
    double bias_regularizer_l1;
    double bias_regularizer_l2;

    LayerDense(double weight_reg_l1 = 0.0, double weight_reg_l2 = 0.0,
               double bias_reg_l1 = 0.0, double bias_reg_l2 = 0.0)
        : weight_regularizer_l1(weight_reg_l1),
          weight_regularizer_l2(weight_reg_l2),
          bias_regularizer_l1(bias_reg_l1),
          bias_regularizer_l2(bias_reg_l2) {
        std::mt19937 gen(0);
        std::normal_distribution<> dist(0.0, 0.01);

        for (size_t i = 0; i < InputSize; ++i) {
            for (size_t j = 0; j < Neurons; ++j) {
                weights[i][j] = dist(gen);
                weight_momentums[i][j] = 0.0;
                weight_cache[i][j] = 0.0;
            }
        }
        biases.fill(0.0);
        bias_momentums.fill(0.0);
        bias_cache.fill(0.0);
    }

    template<size_t BatchSize>
    void forward(const std::array<std::array<double, InputSize>, BatchSize>& inputs_,
                 std::array<std::array<double, Neurons>, BatchSize>& output_) {
        for (size_t i = 0; i < BatchSize; ++i) {
            for (size_t j = 0; j < Neurons; ++j) {
                double sum = biases[j];
                for (size_t k = 0; k < InputSize; ++k) {
                    sum += inputs_[i][k] * weights[k][j];
                }
                output_[i][j] = sum;
            }
        }
    }

    template<size_t BatchSize>
    void backward(const std::array<std::array<double, Neurons>, BatchSize>& dvalues,
                  const std::array<std::array<double, InputSize>, BatchSize>& inputs_,
                  std::array<std::array<double, InputSize>, BatchSize>& dinputs_) {
        // Gradient on weights and biases
        dweights.fill({});
        dbiases.fill(0.0);

        for (size_t i = 0; i < BatchSize; ++i) {
            for (size_t j = 0; j < Neurons; ++j) {
                dbiases[j] += dvalues[i][j];
                for (size_t k = 0; k < InputSize; ++k) {
                    dweights[k][j] += inputs_[i][k] * dvalues[i][j];
                }
            }
        }

        // Regularization gradients
        // Weights
        for (size_t i = 0; i < InputSize; ++i) {
            for (size_t j = 0; j < Neurons; ++j) {
                if (weight_regularizer_l1 > 0.0) {
                    dweights[k][j] += weight_regularizer_l1 * ((weights[k][j] > 0) ? 1.0 : -1.0);
                }
                if (weight_regularizer_l2 > 0.0) {
                    dweights[k][j] += 2.0 * weight_regularizer_l2 * weights[k][j];
                }
            }
        }
        // Biases
        for (size_t j = 0; j < Neurons; ++j) {
            if (bias_regularizer_l1 > 0.0) {
                dbiases[j] += bias_regularizer_l1 * ((biases[j] > 0) ? 1.0 : -1.0);
            }
            if (bias_regularizer_l2 > 0.0) {
                dbiases[j] += 2.0 * bias_regularizer_l2 * biases[j];
            }
        }

        // Gradient on inputs
        for (size_t i = 0; i < BatchSize; ++i) {
            for (size_t j = 0; j < InputSize; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < Neurons; ++k) {
                    sum += dvalues[i][k] * weights[j][k];
                }
                dinputs_[i][j] = sum;
            }
        }
    }

    // Gradients
    std::array<std::array<double, Neurons>, InputSize> dweights;
    std::array<double, Neurons> dbiases;
};

// Softmax Activation
template<size_t BatchSize, size_t InputSize>
struct ActivationSoftmax {
    std::array<std::array<double, InputSize>, BatchSize> output;
    std::array<std::array<double, InputSize>, BatchSize> dinputs;

    void forward(const std::array<std::array<double, InputSize>, BatchSize>& inputs) {
        for (size_t i = 0; i < BatchSize; ++i) {
            double max_input = *std::max_element(inputs[i].begin(), inputs[i].end());
            std::array<double, InputSize> exp_values;
            double sum_exp = 0.0;

            for (size_t j = 0; j < InputSize; ++j) {
                exp_values[j] = std::exp(inputs[i][j] - max_input);
                sum_exp += exp_values[j];
            }
            for (size_t j = 0; j < InputSize; ++j) {
                output[i][j] = exp_values[j] / sum_exp;
            }
        }
    }

    void backward(const std::array<std::array<double, InputSize>, BatchSize>& dvalues) {
        for (size_t i = 0; i < BatchSize; ++i) {
            const std::array<double, InputSize>& single_output = output[i];
            for (size_t j = 0; j < InputSize; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < InputSize; ++k) {
                    double jacobian = (j == k) ? single_output[j] * (1.0 - single_output[k]) : -single_output[j] * single_output[k];
                    sum += jacobian * dvalues[i][k];
                }
                dinputs[i][j] = sum;
            }
        }
    }
};

// Cross-Entropy Loss
template<size_t BatchSize, size_t InputSize>
struct LossCategoricalCrossentropy {
    double data_loss;
    double regularization_loss;

    double forward(const std::array<std::array<double, InputSize>, BatchSize>& y_pred,
                   const std::array<size_t, BatchSize>& y_true) {
        data_loss = 0.0;
        for (size_t i = 0; i < BatchSize; ++i) {
            double correct_confidence = y_pred[i][y_true[i]];
            correct_confidence = std::max(correct_confidence, 1e-7); // Clipping
            data_loss += -std::log(correct_confidence);
        }
        data_loss /= BatchSize;
        return data_loss;
    }

    void backward(std::array<std::array<double, InputSize>, BatchSize>& dinputs,
                  const std::array<std::array<double, InputSize>, BatchSize>& y_pred,
                  const std::array<size_t, BatchSize>& y_true) {
        for (size_t i = 0; i < BatchSize; ++i) {
            dinputs[i] = y_pred[i];
            dinputs[i][y_true[i]] -= 1.0;
        }
        // Normalize
        for (size_t i = 0; i < BatchSize; ++i) {
            for (size_t j = 0; j < InputSize; ++j) {
                dinputs[i][j] /= BatchSize;
            }
        }
    }

    template<size_t InputSize_, size_t Neurons_>
    double calculate_regularization_loss(const LayerDense<InputSize_, Neurons_>& layer) {
        double regularization_loss = 0.0;

        // L1 regularization on weights
        if (layer.weight_regularizer_l1 > 0.0) {
            for (size_t i = 0; i < InputSize_; ++i) {
                for (size_t j = 0; j < Neurons_; ++j) {
                    regularization_loss += layer.weight_regularizer_l1 * std::abs(layer.weights[i][j]);
                }
            }
        }

        // L2 regularization on weights
        if (layer.weight_regularizer_l2 > 0.0) {
            for (size_t i = 0; i < InputSize_; ++i) {
                for (size_t j = 0; j < Neurons_; ++j) {
                    regularization_loss += layer.weight_regularizer_l2 * layer.weights[i][j] * layer.weights[i][j];
                }
            }
        }

        // L1 regularization on biases
        if (layer.bias_regularizer_l1 > 0.0) {
            for (size_t j = 0; j < Neurons_; ++j) {
                regularization_loss += layer.bias_regularizer_l1 * std::abs(layer.biases[j]);
            }
        }

        // L2 regularization on biases
        if (layer.bias_regularizer_l2 > 0.0) {
            for (size_t j = 0; j < Neurons_; ++j) {
                regularization_loss += layer.bias_regularizer_l2 * layer.biases[j] * layer.biases[j];
            }
        }

        return regularization_loss;
    }
};

// Optimizer Adam
struct OptimizerAdam {
    double learning_rate;
    double current_learning_rate;
    double decay;
    size_t iterations;
    double epsilon;
    double beta_1;
    double beta_2;

    OptimizerAdam(double lr = 0.001, double dec = 0.0, double eps = 1e-7,
                  double b1 = 0.9, double b2 = 0.999)
        : learning_rate(lr), current_learning_rate(lr), decay(dec),
          iterations(0), epsilon(eps), beta_1(b1), beta_2(b2) {}

    void pre_update_params() {
        if (decay) {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }

    template<size_t InputSize, size_t Neurons>
    void update_params(LayerDense<InputSize, Neurons>& layer) {
        // Update weights
        for (size_t i = 0; i < InputSize; ++i) {
            for (size_t j = 0; j < Neurons; ++j) {
                // Update momentum
                layer.weight_momentums[i][j] = beta_1 * layer.weight_momentums[i][j] + (1.0 - beta_1) * layer.dweights[i][j];
                // Update cache
                layer.weight_cache[i][j] = beta_2 * layer.weight_cache[i][j] + (1.0 - beta_2) * layer.dweights[i][j] * layer.dweights[i][j];

                // Corrected momentum and cache
                double weight_momentum_corrected = layer.weight_momentums[i][j] / (1.0 - std::pow(beta_1, iterations + 1));
                double weight_cache_corrected = layer.weight_cache[i][j] / (1.0 - std::pow(beta_2, iterations + 1));

                // Update weights
                layer.weights[i][j] += -current_learning_rate * weight_momentum_corrected / (std::sqrt(weight_cache_corrected) + epsilon);
            }
        }

        // Update biases
        for (size_t j = 0; j < Neurons; ++j) {
            // Update momentum
            layer.bias_momentums[j] = beta_1 * layer.bias_momentums[j] + (1.0 - beta_1) * layer.dbiases[j];
            // Update cache
            layer.bias_cache[j] = beta_2 * layer.bias_cache[j] + (1.0 - beta_2) * layer.dbiases[j] * layer.dbiases[j];

            // Corrected momentum and cache
            double bias_momentum_corrected = layer.bias_momentums[j] / (1.0 - std::pow(beta_1, iterations + 1));
            double bias_cache_corrected = layer.bias_cache[j] / (1.0 - std::pow(beta_2, iterations + 1));

            // Update biases
            layer.biases[j] += -current_learning_rate * bias_momentum_corrected / (std::sqrt(bias_cache_corrected) + epsilon);
        }
    }

    void post_update_params() {
        ++iterations;
    }
};

// Main function
int main() {
    // Initialize data arrays
    constexpr size_t TOTAL_SAMPLES = SAMPLES * CLASSES;
    std::array<std::array<double, INPUT_DIM>, TOTAL_SAMPLES> X;
    std::array<size_t, TOTAL_SAMPLES> y;

    // Generate dataset
    spiral_data<SAMPLES, CLASSES>(X, y);

    // Initialize layers with regularization parameters
    constexpr size_t BATCH_SIZE = TOTAL_SAMPLES;
    LayerDense<INPUT_DIM, HIDDEN_NEURONS> dense1(0.0, 5e-4, 0.0, 5e-4);
    ActivationReLU<BATCH_SIZE, HIDDEN_NEURONS> activation1;
    LayerDense<HIDDEN_NEURONS, OUTPUT_NEURONS> dense2;
    ActivationSoftmax<BATCH_SIZE, OUTPUT_NEURONS> activation_softmax;
    LossCategoricalCrossentropy<BATCH_SIZE, OUTPUT_NEURONS> loss_function;
    OptimizerAdam optimizer(0.02, 5e-7);

    // Arrays for forward and backward passes
    std::array<std::array<double, HIDDEN_NEURONS>, BATCH_SIZE> dense1_output;
    std::array<std::array<double, HIDDEN_NEURONS>, BATCH_SIZE> activation1_output;
    std::array<std::array<double, OUTPUT_NEURONS>, BATCH_SIZE> dense2_output;
    std::array<std::array<double, OUTPUT_NEURONS>, BATCH_SIZE> loss_dinputs;
    std::array<std::array<double, HIDDEN_NEURONS>, BATCH_SIZE> dense2_dinputs;
    std::array<std::array<double, HIDDEN_NEURONS>, BATCH_SIZE> activation1_dinputs;
    std::array<std::array<double, INPUT_DIM>, BATCH_SIZE> dense1_dinputs;

    // Training loop
    constexpr size_t EPOCHS = 10000;
    for (size_t epoch = 0; epoch <= EPOCHS; ++epoch) {
        // Forward pass
        dense1.forward<BATCH_SIZE>(X, dense1_output);
        activation1.forward(dense1_output);
        dense2.forward<BATCH_SIZE>(activation1.output, dense2_output);
        activation_softmax.forward(dense2_output);
        double data_loss = loss_function.forward(activation_softmax.output, y);

        // Calculate regularization loss
        double reg_loss = 0.0;
        reg_loss += loss_function.calculate_regularization_loss(dense1);
        reg_loss += loss_function.calculate_regularization_loss(dense2);

        double total_loss = data_loss + reg_loss;

        // Calculate accuracy
        size_t correct_predictions = 0;
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            auto max_iter = std::max_element(activation_softmax.output[i].begin(), activation_softmax.output[i].end());
            size_t prediction = std::distance(activation_softmax.output[i].begin(), max_iter);
            if (prediction == y[i]) {
                ++correct_predictions;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / BATCH_SIZE;

        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Accuracy: " << accuracy
                      << ", Loss: " << total_loss
                      << " (Data Loss: " << data_loss
                      << ", Reg Loss: " << reg_loss << ")"
                      << ", Learning Rate: " << optimizer.current_learning_rate << std::endl;
        }

        // Backward pass
        loss_function.backward(loss_dinputs, activation_softmax.output, y);
        activation_softmax.backward(loss_dinputs);
        dense2.backward<BATCH_SIZE>(activation_softmax.dinputs, activation1.output, dense2_dinputs);
        activation1.backward(dense2_dinputs);
        dense1.backward<BATCH_SIZE>(activation1.dinputs, X, dense1_dinputs);

        // Update parameters
        optimizer.pre_update_params();
        optimizer.update_params(dense2);
        optimizer.update_params(dense1);
        optimizer.post_update_params();
    }

    // Validation
    // Generate test data
    std::array<std::array<double, INPUT_DIM>, TOTAL_SAMPLES> X_test;
    std::array<size_t, TOTAL_SAMPLES> y_test;
    spiral_data<SAMPLES, CLASSES>(X_test, y_test);

    // Forward pass
    dense1.forward<BATCH_SIZE>(X_test, dense1_output);
    activation1.forward(dense1_output);
    dense2.forward<BATCH_SIZE>(activation1.output, dense2_output);
    activation_softmax.forward(dense2_output);
    double test_loss = loss_function.forward(activation_softmax.output, y_test);

    // Calculate accuracy
    size_t test_correct_predictions = 0;
    for (size_t i = 0; i < BATCH_SIZE; ++i) {
        auto max_iter = std::max_element(activation_softmax.output[i].begin(), activation_softmax.output[i].end());
        size_t prediction = std::distance(activation_softmax.output[i].begin(), max_iter);
        if (prediction == y_test[i]) {
            ++test_correct_predictions;
        }
    }
    double test_accuracy = static_cast<double>(test_correct_predictions) / BATCH_SIZE;

    std::cout << "Validation, Accuracy: " << test_accuracy << ", Loss: " << test_loss << std::endl;

    return 0;
}
