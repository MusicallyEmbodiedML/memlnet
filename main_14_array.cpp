#include <iostream>
#include <cmath>
#include <array>
#include <algorithm>

const int POINTS_PER_CLASS = 100;
const int NUM_CLASSES = 3;
const int INPUT_SIZE = 2;
const int LAYER_1_SIZE = 64;
const int OUTPUT_SIZE = 3;
const double LEARNING_RATE = 0.01;

// Fixed-size array for inputs and outputs
using DataArray = std::array<std::array<double, INPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES>;
using LabelArray = std::array<std::array<double, OUTPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES>;

// Utility function to generate spiral data
void generate_spiral_data(DataArray &X, LabelArray &y) {
    for (int class_number = 0; class_number < NUM_CLASSES; ++class_number) {
        double angle_start = class_number * 4.0;
        for (int i = 0; i < POINTS_PER_CLASS; ++i) {
            double t = (double)i / POINTS_PER_CLASS;
            double r = t * 1.0;
            double theta = angle_start + t * 4.0 * M_PI;
            double x1 = r * sin(theta);
            double x2 = r * cos(theta);
            X[class_number * POINTS_PER_CLASS + i] = {x1, x2};
            y[class_number * POINTS_PER_CLASS + i].fill(0.0);
            y[class_number * POINTS_PER_CLASS + i][class_number] = 1.0;
        }
    }
}

// Dense layer with fixed-size weights and biases arrays, forward and backward propagation
class Layer_Dense {
public:
    std::array<std::array<double, LAYER_1_SIZE>, INPUT_SIZE> weights;
    std::array<double, LAYER_1_SIZE> biases;
    std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> output;
    std::array<std::array<double, INPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> dinputs;
    std::array<std::array<double, LAYER_1_SIZE>, INPUT_SIZE> dweights;
    std::array<double, LAYER_1_SIZE> dbiases;

    Layer_Dense() {
        // Initialize weights and biases with small random values
        for (auto &row : weights) {
            for (auto &val : row) {
                val = ((double)rand() / RAND_MAX) * 2 - 1;  // Random between -1 and 1
            }
        }
        biases.fill(0.0);
    }

    // Forward pass
    void forward(const DataArray &inputs) {
        for (int i = 0; i < inputs.size(); ++i) {
            for (int j = 0; j < LAYER_1_SIZE; ++j) {
                output[i][j] = biases[j];
                for (int k = 0; k < INPUT_SIZE; ++k) {
                    output[i][j] += inputs[i][k] * weights[k][j];
                }
            }
        }
    }

    // Backward pass
    void backward(const std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &dvalues, const DataArray &inputs) {
        // Reset gradients
        dweights.fill({});
        dbiases.fill(0.0);

        for (int i = 0; i < inputs.size(); ++i) {
            for (int j = 0; j < LAYER_1_SIZE; ++j) {
                dbiases[j] += dvalues[i][j];
                for (int k = 0; k < INPUT_SIZE; ++k) {
                    dweights[k][j] += inputs[i][k] * dvalues[i][j];
                    dinputs[i][k] += dvalues[i][j] * weights[k][j];
                }
            }
        }

        // Update weights and biases
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < LAYER_1_SIZE; ++j) {
                weights[i][j] -= LEARNING_RATE * dweights[i][j];
            }
        }
        for (int i = 0; i < LAYER_1_SIZE; ++i) {
            biases[i] -= LEARNING_RATE * dbiases[i];
        }
    }
};

// ReLU activation with fixed-size output and backpropagation
class Activation_ReLU {
public:
    std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> output;
    std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> dinputs;

    // Forward pass
    void forward(const std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &inputs) {
        for (int i = 0; i < inputs.size(); ++i) {
            for (int j = 0; j < LAYER_1_SIZE; ++j) {
                output[i][j] = std::max(0.0, inputs[i][j]);
            }
        }
    }

    // Backward pass
    void backward(const std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &dvalues,
                  const std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &inputs) {
        for (int i = 0; i < inputs.size(); ++i) {
            for (int j = 0; j < LAYER_1_SIZE; ++j) {
                dinputs[i][j] = (inputs[i][j] > 0) ? dvalues[i][j] : 0.0;
            }
        }
    }
};

// Output layer with fixed-size weights, biases, and backpropagation
class Layer_Output {
public:
    std::array<std::array<double, OUTPUT_SIZE>, LAYER_1_SIZE> weights;
    std::array<double, OUTPUT_SIZE> biases;
    std::array<std::array<double, OUTPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> output;
    std::array<std::array<double, OUTPUT_SIZE>, LAYER_1_SIZE> dweights;
    std::array<double, OUTPUT_SIZE> dbiases;

    Layer_Output() {
        // Initialize weights and biases with random values
        for (auto &row : weights) {
            for (auto &val : row) {
                val = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
        biases.fill(0.0);
    }

    // Forward pass
    void forward(const std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &inputs) {
        for (int i = 0; i < inputs.size(); ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                output[i][j] = biases[j];
                for (int k = 0; k < LAYER_1_SIZE; ++k) {
                    output[i][j] += inputs[i][k] * weights[k][j];
                }
            }
        }
    }

    // Backward pass
    void backward(const std::array<std::array<double, OUTPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &dvalues,
                  const std::array<std::array<double, LAYER_1_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &inputs) {
        dweights.fill({});
        dbiases.fill(0.0);

        for (int i = 0; i < inputs.size(); ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                dbiases[j] += dvalues[i][j];
                for (int k = 0; k < LAYER_1_SIZE; ++k) {
                    dweights[k][j] += inputs[i][k] * dvalues[i][j];
                }
            }
        }

        // Update weights and biases
        for (int i = 0; i < LAYER_1_SIZE; ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                weights[i][j] -= LEARNING_RATE * dweights[i][j];
            }
        }
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            biases[i] -= LEARNING_RATE * dbiases[i];
        }
    }
};

// Mean Squared Error loss function with fixed-size, forward and backward propagation
class Loss_MeanSquaredError {
public:
    std::array<std::array<double, OUTPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> dinputs;

    // Forward pass
    double forward(const std::array<std::array<double, OUTPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &predictions,
                   const LabelArray &targets) {
        double loss = 0.0;
        for (int i = 0; i < predictions.size(); ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                loss += std::pow(predictions[i][j] - targets[i][j], 2);
            }
        }
        return loss / predictions.size();
    }

    // Backward pass
    void backward(const std::array<std::array<double, OUTPUT_SIZE>, POINTS_PER_CLASS * NUM_CLASSES> &predictions,
                  const LabelArray &targets) {
        for (int i = 0; i < predictions.size(); ++i) {
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                dinputs[i][j] = 2 * (predictions[i][j] - targets[i][j]) / predictions.size();
            }
        }
    }
};

// Main function demonstrating the usage
int main() {
    // Fixed-size arrays for inputs and labels
    DataArray X;
    LabelArray y;

    // Generate spiral data
    generate_spiral_data(X, y);

    // Create the first Dense layer with 2 inputs and 64 neurons
    Layer_Dense dense1;

    // Create ReLU activation
    Activation_ReLU relu1;

    // Create the second Dense layer with 64 inputs and 3 outputs
    Layer_Output output_layer;

    // Create Mean Squared Error loss function
    Loss_MeanSquaredError loss_function;

    // Training loop
    for (int epoch = 0; epoch < 10000; ++epoch) {
        // Forward pass
        dense1.forward(X);
        relu1.forward(dense1.output);
        output_layer.forward(relu1.output);

        // Loss calculation
        double loss = loss_function.forward(output_layer.output, y);

        // Backward pass
        loss_function.backward(output_layer.output, y);
        output_layer.backward(loss_function.dinputs, relu1.output);
        relu1.backward(output_layer.dinputs, dense1.output);
        dense1.backward(relu1.dinputs, X);

        // Print loss every 100 epochs
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }
    }

    return 0;
}
