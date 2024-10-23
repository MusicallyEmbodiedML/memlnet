#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

// Utility function for initializing random numbers
double rand_double() {
    return (static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0;
}

// Utility function to generate spiral data
void generate_spiral_data(int points_per_class, int num_classes, vector<vector<double>>& X, vector<vector<double>>& y) {
    for (int class_number = 0; class_number < num_classes; ++class_number) {
        double angle_start = class_number * 4.0;
        for (int i = 0; i < points_per_class; ++i) {
            double t = (double)i / points_per_class;
            double r = t * 1.0;
            double theta = angle_start + t * 4.0 * M_PI;
            double x1 = r * sin(theta);
            double x2 = r * cos(theta);
            X.push_back({x1, x2});
            vector<double> class_label(num_classes, 0.0);
            class_label[class_number] = 1.0;
            y.push_back(class_label);
        }
    }
}


class Layer_Input {
public:
    vector<vector<double>> output;

    // Forward pass for the input layer
    void forward(const vector<vector<double>>& inputs, bool training) {
        // Just set the inputs as the output for the input layer
        output = inputs;
    }
};

// Dense layer with L1 and L2 regularization
class Layer_Dense {
public:
    vector<vector<double>> weights;
    vector<double> biases;
    vector<vector<double>> inputs;
    vector<vector<double>> output;
    vector<vector<double>> dweights;
    vector<double> dbiases;
    vector<vector<double>> dinputs;

    // For Adam optimizer
    vector<vector<double>> weight_momentums;
    vector<vector<double>> weight_caches;
    vector<double> bias_momentums;
    vector<double> bias_caches;

    // Regularization parameters
    double weight_regularizer_l1;
    double weight_regularizer_l2;
    double bias_regularizer_l1;
    double bias_regularizer_l2;

    Layer_Dense(int n_inputs, int n_neurons,
                double weight_regularizer_l1 = 0.0, double weight_regularizer_l2 = 0.0,
                double bias_regularizer_l1 = 0.0, double bias_regularizer_l2 = 0.0)
        : weight_regularizer_l1(weight_regularizer_l1),
          weight_regularizer_l2(weight_regularizer_l2),
          bias_regularizer_l1(bias_regularizer_l1),
          bias_regularizer_l2(bias_regularizer_l2) {

        // Initialize weights with small random values and biases with zero
        weights.resize(n_inputs, vector<double>(n_neurons));
        for (int i = 0; i < n_inputs; ++i) {
            for (int j = 0; j < n_neurons; ++j) {
                weights[i][j] = rand_double();
            }
        }
        biases.resize(n_neurons, 0.0);

        // Initialize Adam optimizer terms
        weight_momentums.resize(n_inputs, vector<double>(n_neurons, 0.0));
        weight_caches.resize(n_inputs, vector<double>(n_neurons, 0.0));
        bias_momentums.resize(n_neurons, 0.0);
        bias_caches.resize(n_neurons, 0.0);
    }

    // Forward pass
    void forward(const vector<vector<double>>& inputs) {
        this->inputs = inputs;
        output.resize(inputs.size(), vector<double>(weights[0].size()));

        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < weights[0].size(); ++j) {
                output[i][j] = biases[j];
                for (size_t k = 0; k < weights.size(); ++k) {
                    output[i][j] += inputs[i][k] * weights[k][j];
                }
            }
        }
    }

    // Backward pass
    void backward(const vector<vector<double>>& dvalues) {
        dweights.resize(weights.size(), vector<double>(weights[0].size()));
        dbiases.resize(weights[0].size());
        dinputs.resize(inputs.size(), vector<double>(inputs[0].size()));

        // Calculate dweights and dbiases
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < weights[0].size(); ++j) {
                dbiases[j] += dvalues[i][j];
                for (size_t k = 0; k < weights.size(); ++k) {
                    dweights[k][j] += inputs[i][k] * dvalues[i][j];
                }
            }
        }

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
    vector<vector<double>> output;
    vector<vector<double>> dinputs;

    // Forward pass
    void forward(const vector<vector<double>>& inputs) {
        output.resize(inputs.size(), vector<double>(inputs[0].size()));

        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[0].size(); ++j) {
                output[i][j] = max(0.0, inputs[i][j]);
            }
        }
    }

    // Backward pass
    void backward(const vector<vector<double>>& dvalues, const vector<vector<double>>& inputs) {
        dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size()));

        for (size_t i = 0; i < dvalues.size(); ++i) {
            for (size_t j = 0; j < dvalues[0].size(); ++j) {
                dinputs[i][j] = inputs[i][j] > 0.0 ? dvalues[i][j] : 0.0;
            }
        }
    }
};

// Mean Squared Error loss function
class Loss_MeanSquaredError {
public:
    // Forward pass (returns loss value)
    double forward(const vector<vector<double>>& predictions, const vector<vector<double>>& targets) {
        double loss = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[0].size(); ++j) {
                loss += pow(predictions[i][j] - targets[i][j], 2);
            }
        }
        return loss / predictions.size();
    }

    // Backward pass (returns gradient with respect to predictions)
    vector<vector<double>> backward(const vector<vector<double>>& predictions, const vector<vector<double>>& targets) {
        vector<vector<double>> dinputs(predictions.size(), vector<double>(predictions[0].size()));

        for (size_t i = 0; i < predictions.size(); ++i) {
            for (size_t j = 0; j < predictions[0].size(); ++j) {
                dinputs[i][j] = 2 * (predictions[i][j] - targets[i][j]) / predictions.size();
            }
        }

        return dinputs;
    }
};

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

class Loss_CategoricalCrossentropy {
public:
    vector<vector<double>> dinputs;

    // Forward pass
    double forward(const vector<vector<double>>& y_pred, const vector<vector<double>>& y_true) {
        size_t samples = y_pred.size();
        vector<double> correct_confidences(samples);

        // Clip predictions to prevent division by 0 and dragging log towards infinity
        vector<vector<double>> y_pred_clipped = y_pred;
        for (size_t i = 0; i < samples; ++i) {
            for (size_t j = 0; j < y_pred[i].size(); ++j) {
                y_pred_clipped[i][j] = max(1e-7, min(1 - 1e-7, y_pred[i][j]));
            }
        }

        // Calculate confidence for the correct class
        for (size_t i = 0; i < samples; ++i) {
            if (y_true[i].size() == 1) {  // Sparse categorical labels
                size_t correct_class = distance(y_true[i].begin(), find(y_true[i].begin(), y_true[i].end(), 1.0));
                correct_confidences[i] = y_pred_clipped[i][correct_class];
            } else {  // One-hot encoded labels
                double sum = 0.0;
                for (size_t j = 0; j < y_true[i].size(); ++j) {
                    sum += y_pred_clipped[i][j] * y_true[i][j];
                }
                correct_confidences[i] = sum;
            }
        }

        // Calculate losses
        double loss = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            loss += -log(correct_confidences[i]);
        }
        return loss / samples;
    }

    // Backward pass
    vector<vector<double>> backward(const vector<vector<double>>& dvalues, const vector<vector<double>>& y_true) {
        size_t samples = dvalues.size();
        size_t labels = dvalues[0].size();

        // If labels are sparse, convert them to one-hot encoding
        vector<vector<double>> y_true_one_hot = y_true;
        if (y_true[0].size() == 1) {
            y_true_one_hot.resize(samples, vector<double>(labels, 0.0));
            for (size_t i = 0; i < samples; ++i) {
                size_t correct_class = distance(y_true[i].begin(), find(y_true[i].begin(), y_true[i].end(), 1.0));
                y_true_one_hot[i][correct_class] = 1.0;
            }
        }

        // Initialize gradients (dinputs) with zeros
        dinputs.resize(samples, vector<double>(labels, 0.0));

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

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

class Activation_Softmax_Loss_CategoricalCrossentropy {
public:
    vector<vector<double>> dinputs;

    // Backward pass
    vector<vector<double>> backward(const vector<vector<double>>& dvalues, const vector<vector<double>>& y_true) {
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

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

class Optimizer_RMSprop {
public:
    double learning_rate;
    double current_learning_rate;
    double decay;
    double epsilon;
    double rho;
    int iterations;

    // Constructor
    Optimizer_RMSprop(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7, double rho = 0.9)
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
            layer.weight_caches.resize(layer.weights.size(), vector<double>(layer.weights[0].size(), 0.0));
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
//     double learning_rate;
//     double beta_1;
//     double beta_2;
//     double epsilon;
//     int t;
//
//     Optimizer_Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-7) {
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
//                 double corrected_weight_momentum = layer.weight_momentums[i][j] / (1 - pow(beta_1, t));
//                 double corrected_weight_cache = layer.weight_caches[i][j] / (1 - pow(beta_2, t));
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
//             double corrected_bias_momentum = layer.bias_momentums[j] / (1 - pow(beta_1, t));
//             double corrected_bias_cache = layer.bias_caches[j] / (1 - pow(beta_2, t));
//
//             // Update biases
//             layer.biases[j] -= learning_rate * corrected_bias_momentum / (sqrt(corrected_bias_cache) + epsilon);
//         }
//     }
// };

class Optimizer_Adam {
public:
    double learning_rate;
    double current_learning_rate;
    double decay;
    double epsilon;
    double beta_1;
    double beta_2;
    int iterations;

    Optimizer_Adam(double learning_rate = 0.001, double decay = 0., double epsilon = 1e-7,
                   double beta_1 = 0.9, double beta_2 = 0.999)
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

        for (size_t i = 0; i < layer.biases.size(); ++i) {
            layer.bias_momentums[i] = beta_1 * layer.bias_momentums[i] + (1 - beta_1) * layer.dbiases[i];
            layer.bias_caches[i] = beta_2 * layer.bias_caches[i] + (1 - beta_2) * std::pow(layer.dbiases[i], 2);
        }

        // Get corrected momentums and caches
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                double weight_momentum_corrected = layer.weight_momentums[i][j] / (1 - std::pow(beta_1, iterations + 1));
                double weight_cache_corrected = layer.weight_caches[i][j] / (1 - std::pow(beta_2, iterations + 1));
                layer.weights[i][j] += -current_learning_rate * weight_momentum_corrected / (std::sqrt(weight_cache_corrected) + epsilon);
            }
        }

        for (size_t i = 0; i < layer.biases.size(); ++i) {
            double bias_momentum_corrected = layer.bias_momentums[i] / (1 - std::pow(beta_1, iterations + 1));
            double bias_cache_corrected = layer.bias_caches[i] / (1 - std::pow(beta_2, iterations + 1));
            layer.biases[i] += -current_learning_rate * bias_momentum_corrected / (std::sqrt(bias_cache_corrected) + epsilon);
        }
    }

    // Call once after any parameter updates
    void post_update_params() {
        ++iterations;
    }
};

  int main() {
      // Generate spiral data
      vector<vector<double>> X;
      vector<vector<double>> y;
      generate_spiral_data(100, 3, X, y);  // 100 points per class, 3 classes
      // std::vector<std::vector<double>> X { {0, 0}, {1, 4}, {5, 6}, {7,2} };  // Example input
      // std::vector<std::vector<double>>  y { {0}, {1}, {1} ,{0}};  // Example labels

      // Create the first Dense layer with 2 inputs and 64 neurons, and L2 regularization
      Layer_Dense dense1(2, 64, 0.000, 5e-4, 0, 5e-4);

      // Create ReLU activation for first layer
      Activation_ReLU relu1;

      // Create the second Dense layer with 64 inputs and 3 neurons (3 output classes), and L2 regularization

      Layer_Dense dense2(64, 3);

      // Create Adam optimizer
      // Optimizer_Adam adam_optimizer(0.02, 5e-7);

      Optimizer_RMSprop rms_opt;

      // Mean Squared Error loss
      Loss_MeanSquaredError loss_function;

      // Training for 1000 epochs
      for (int epoch = 0; epoch < 50000; ++epoch) {
          // Forward pass through first dense layer
          dense1.forward(X);

          // Forward pass through ReLU activation for first layer
          relu1.forward(dense1.output);

          // Forward pass through second dense layer
          dense2.forward(relu1.output);


          // Calculate loss
          double loss = loss_function.forward(dense2.output, y);

          // Backward pass through loss function
          vector<vector<double>> dloss = loss_function.backward(dense2.output, y);

          // Backward pass through second Dense layer
          dense2.backward(dloss);
          relu1.backward(dense2.dinputs, dense1.output);
          dense1.backward(relu1.dinputs);


          // Update parameters using Adam optimizer for both layers
          // adam_optimizer.pre_update_params();
          // adam_optimizer.update_params(dense1);
          // adam_optimizer.update_params(dense2);
          // adam_optimizer.update_params(dense3);
          // adam_optimizer.post_update_params();
          rms_opt.update_params(dense1);
          rms_opt.update_params(dense2);

          // Print loss every 100 epochs
          if (epoch % 10 == 0) {
              cout << "Epoch: " << epoch << " Loss: " << loss << endl;
          }
      }

     return 0;
 }
