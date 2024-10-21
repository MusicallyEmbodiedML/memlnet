#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>


// Transpose of a matrix
std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

// Sum the rows of a matrix (used for calculating dbiases)
std::vector<std::vector<double>> sum_rows(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(1, std::vector<double>(matrix[0].size(), 0.0));
    for (const auto& row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            result[0][j] += row[j];
        }
    }
    return result;
}

// Matrix multiplication
std::vector<std::vector<double>> dot(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t common = A[0].size();

    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < common; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// Element-wise maximum
std::vector<std::vector<double>> relu(const std::vector<std::vector<double>>& inputs) {
    std::vector<std::vector<double>> result = inputs;
    for (auto& row : result) {
        for (auto& elem : row) {
            elem = std::max(0.0, elem);
        }
    }
    return result;
}

// Softmax function
std::vector<std::vector<double>> softmax(const std::vector<std::vector<double>>& inputs) {
    std::vector<std::vector<double>> result = inputs;
    for (auto& row : result) {
        double max_input = *std::max_element(row.begin(), row.end());
        double sum_exp = 0.0;
        for (auto& elem : row) {
            elem = std::exp(elem - max_input);
            sum_exp += elem;
        }
        for (auto& elem : row) {
            elem /= sum_exp;
        }
    }
    return result;
}

// Helper to create zero matrices
std::vector<std::vector<double>> zeros(size_t rows, size_t cols) {
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

class Loss_CategoricalCrossentropy {
public:
    double calculate(const std::vector<std::vector<double>>& y_pred, const std::vector<size_t>& y_true) {
        size_t samples = y_pred.size();
        std::vector<double> correct_confidences(samples);

        for (size_t i = 0; i < samples; ++i) {
            correct_confidences[i] = y_pred[i][y_true[i]];
        }

        double sum_loss = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            sum_loss += -std::log(std::max(correct_confidences[i], 1e-7));
        }

        return sum_loss / samples;
    }
};
class Layer_Dense {
public:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> output;
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> dweights;
    std::vector<std::vector<double>> dbiases;
    std::vector<std::vector<double>> dinputs;
    std::vector<std::vector<double>> weight_momentums; // Momentum for weights
    std::vector<std::vector<double>> bias_momentums;   // Momentum for biases

    Layer_Dense(size_t n_inputs, size_t n_neurons) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 1.0);

        weights = std::vector<std::vector<double>>(n_inputs, std::vector<double>(n_neurons));
        biases = zeros(1, n_neurons);

        for (size_t i = 0; i < n_inputs; ++i) {
            for (size_t j = 0; j < n_neurons; ++j) {
                weights[i][j] = 0.01 * dis(gen);
            }
        }
    }

    void forward(const std::vector<std::vector<double>>& inputs) {
        this->inputs = inputs;
        output = dot(inputs, weights);
        for (size_t i = 0; i < output.size(); ++i) {
            for (size_t j = 0; j < biases[0].size(); ++j) {
                output[i][j] += biases[0][j];
            }
        }
    }

    void backward(const std::vector<std::vector<double>>& dvalues) {
        dweights = dot(transpose(inputs), dvalues);
        dbiases = sum_rows(dvalues);
        dinputs = dot(dvalues, transpose(weights));
    }
};

class Activation_Softmax_Loss_CategoricalCrossentropy {
public:
    std::vector<std::vector<double>> output;
    std::vector<std::vector<double>> dinputs;

    void forward(const std::vector<std::vector<double>>& inputs) {
        output = softmax(inputs);
    }

    double calculate_loss(const std::vector<size_t>& y_true) {
        Loss_CategoricalCrossentropy loss;
        return loss.calculate(output, y_true);
    }

    void backward(const std::vector<size_t>& y_true) {
        size_t samples = output.size();
        size_t labels = output[0].size();

        dinputs = output;
        for (size_t i = 0; i < samples; ++i) {
            dinputs[i][y_true[i]] -= 1;
        }

        // Normalize the gradient
        for (auto& row : dinputs) {
            for (auto& elem : row) {
                elem /= samples;
            }
        }
    }
};

class Activation_ReLU {
public:
    std::vector<std::vector<double>> output;
    std::vector<std::vector<double>> dinputs;

    void forward(const std::vector<std::vector<double>>& inputs) {
        output = relu(inputs);
    }

    void backward(const std::vector<std::vector<double>>& dvalues) {
        dinputs = dvalues;
        for (size_t i = 0; i < dinputs.size(); ++i) {
            for (size_t j = 0; j < dinputs[0].size(); ++j) {
                if (output[i][j] <= 0) {
                    dinputs[i][j] = 0;
                }
            }
        }
    }
};

class Optimizer_SGD {
public:
    double learning_rate;
    double decay;
    double momentum;
    double current_learning_rate;
    size_t iterations;

    // Constructor to initialize learning rate, decay, and momentum
    Optimizer_SGD(double learning_rate = 1.0, double decay = 0.0, double momentum = 0.0)
        : learning_rate(learning_rate), decay(decay), momentum(momentum), iterations(0) {
        current_learning_rate = learning_rate;
    }

    // Call before any parameter updates (for learning rate decay)
    void pre_update_params() {
        if (decay) {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }

    // Update the parameters (weights and biases)
    void update_params(Layer_Dense& layer) {
        // If momentum is used, we initialize momentum arrays if they don't exist
        if (momentum) {
            if (layer.weight_momentums.empty()) {
                layer.weight_momentums = zeros(layer.weights.size(), layer.weights[0].size());
                layer.bias_momentums = zeros(layer.biases.size(), layer.biases[0].size());
            }

            // Update weights and biases using momentum
            for (size_t i = 0; i < layer.weights.size(); ++i) {
                for (size_t j = 0; j < layer.weights[0].size(); ++j) {
                    layer.weight_momentums[i][j] = momentum * layer.weight_momentums[i][j] - current_learning_rate * layer.dweights[i][j];
                    layer.weights[i][j] += layer.weight_momentums[i][j];
                }
            }

            for (size_t i = 0; i < layer.biases.size(); ++i) {
                for (size_t j = 0; j < layer.biases[0].size(); ++j) {
                    layer.bias_momentums[i][j] = momentum * layer.bias_momentums[i][j] - current_learning_rate * layer.dbiases[i][j];
                    layer.biases[i][j] += layer.bias_momentums[i][j];
                }
            }
        } else {
            // Standard gradient descent update (without momentum)
            for (size_t i = 0; i < layer.weights.size(); ++i) {
                for (size_t j = 0; j < layer.weights[0].size(); ++j) {
                    layer.weights[i][j] -= current_learning_rate * layer.dweights[i][j];
                }
            }

            for (size_t i = 0; i < layer.biases.size(); ++i) {
                for (size_t j = 0; j < layer.biases[0].size(); ++j) {
                    layer.biases[i][j] -= current_learning_rate * layer.dbiases[i][j];
                }
            }
        }
    }

    // Call after parameter updates (for learning rate decay and iterations)
    void post_update_params() {
        iterations++;
    }
};


int main()
{
    std::cout << "MEML Net Test!" << std::endl;
    // Simulated dataset (replace with real data)
    std::vector<std::vector<double>> X = { {0, 0, 0 }, {-1, 0, 0}, {0, 0, 1}, {-1,-1,1} };  // Example input
    std::vector<size_t> y = { 1, 0, 1 ,0};  // Example labels

    // Initialize layers
    Layer_Dense dense1(3, 4);
    Activation_ReLU activation1;

    Layer_Dense dense2(4, 3);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;

    // Initialize optimizer
    Optimizer_SGD optimizer(0.01, 1e-6, 0.9); // Learning rate, decay, momentum

    // Training loop
    for (size_t epoch = 0; epoch < 500; ++epoch) {
        // Forward pass
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        loss_activation.forward(dense2.output);

        // Calculate loss
        double loss = loss_activation.calculate_loss(y);

        // Calculate accuracy (predicted labels)
        size_t correct = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            if (std::distance(loss_activation.output[i].begin(),
                              std::max_element(loss_activation.output[i].begin(), loss_activation.output[i].end())) == y[i]) {
                ++correct;
                              }
        }
        double accuracy = (double)correct / y.size();

        // Print progress every 1000 epochs
        if (epoch % 50 == 0) {
            std::cout << "Epoch: " << epoch << " Loss: " << loss << " Accuracy: " << accuracy * 100.0 << "%" << std::endl;
        }

        // Backward pass
        loss_activation.backward(y);
        dense2.backward(loss_activation.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        // Update parameters
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }
    return 0;
}
