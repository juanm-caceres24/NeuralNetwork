#pragma once

#include "nn/dataset.hpp"
#include "nn/network.hpp"
#include "nn/trainer.hpp"
#include "nn/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace nn {
namespace metrics {

constexpr int kReportFormatVersion = 1;

enum class TaskType {
    Classification,
    Regression,
};

TaskType infer_task_type(LossType loss) noexcept;
std::string to_string(TaskType task);

struct ClassMetrics {
    std::size_t class_index = 0;
    std::size_t support = 0;
    Scalar actual_frequency = Scalar{0};
    Scalar mean_predicted_prob = Scalar{0};
    Scalar precision = Scalar{0};
    Scalar recall = Scalar{0};
    Scalar f1 = Scalar{0};
};

struct ClassificationMetrics {
    std::size_t num_classes = 0;
    Scalar loss = Scalar{0};
    Scalar baseline_constant_loss = Scalar{0};
    Scalar loss_improvement_vs_baseline = Scalar{0};
    Scalar accuracy = Scalar{0};
    Scalar majority_baseline_accuracy = Scalar{0};
    Scalar macro_f1 = Scalar{0};
    // confusion_matrix[true_class][pred_class]
    std::vector<std::vector<std::size_t>> confusion_matrix;
    std::vector<ClassMetrics> per_class;
};

struct OutputRegressionMetrics {
    std::size_t output_index = 0;
    Scalar mae = Scalar{0};
    Scalar rmse = Scalar{0};
    Scalar r2 = Scalar{0};
};

struct RegressionMetrics {
    std::size_t num_outputs = 0;
    Scalar loss = Scalar{0};
    Scalar baseline_constant_loss = Scalar{0};
    Scalar loss_improvement_vs_baseline = Scalar{0};
    Scalar mae = Scalar{0};
    Scalar rmse = Scalar{0};
    Scalar r2 = Scalar{0};
    std::vector<OutputRegressionMetrics> per_output;
};

struct ModelSummary {
    std::size_t input_size = 0;
    std::size_t output_size = 0;
    std::size_t parameters = 0;
    std::vector<std::size_t> layer_sizes;
    std::vector<std::string> layer_activations;
};

struct EvaluationReport {
    int format_version = kReportFormatVersion;
    TaskType task = TaskType::Regression;
    std::string eval_split = "validation";
    double elapsed_ms = 0.0;
    std::size_t train_samples = 0;
    std::size_t val_samples = 0;

    ModelSummary model;
    TrainingConfig training_config;
    TrainingResult training_result;

    ClassificationMetrics classification;
    RegressionMetrics regression;
};

// Evaluates classification metrics on `eval_dataset` without modifying `network`.
// `baseline_train_dataset` provides the empirical class prior and majority class
// used to compute leakage-free constant-predictor baselines on `eval_dataset`.
ClassificationMetrics evaluate_classification(const Network& network,
                                              const Dataset& eval_dataset,
                                              const Dataset& baseline_train_dataset,
                                              LossType loss_type);

// Evaluates regression metrics on `eval_dataset` without modifying `network`.
// `baseline_train_dataset` provides the per-output mean target constant predictor
// used to compute the leakage-free baseline loss on `eval_dataset`.
RegressionMetrics evaluate_regression(const Network& network,
                                      const Dataset& eval_dataset,
                                      const Dataset& baseline_train_dataset,
                                      LossType loss_type,
                                      Scalar huber_delta = Scalar{1.0f});

// Builds a complete structured EvaluationReport for a finished training run.
// If `val_dataset` is non-null and non-empty, evaluation runs on `val_dataset`
// ("validation") with baselines fitted on `train_dataset`.
// Otherwise, evaluation runs on `train_dataset` ("train").
EvaluationReport build_evaluation_report(const Network& network,
                                         const Dataset& train_dataset,
                                         const Dataset* val_dataset,
                                         const TrainingConfig& training_config,
                                         const TrainingResult& training_result,
                                         double elapsed_ms = 0.0);

std::string report_to_json(const EvaluationReport& report, bool pretty = true);
void save_report_json(const EvaluationReport& report,
                      const std::string& path,
                      bool pretty = true);

}  // namespace metrics
}  // namespace nn
