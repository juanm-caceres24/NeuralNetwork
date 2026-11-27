#include "nn/metrics.hpp"

#include "nn/loss.hpp"
#include "nn/workspace.hpp"

#include "cJSON.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace nn {
namespace metrics {

namespace {

using CJsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
using CStringPtr = std::unique_ptr<char, decltype(&std::free)>;

constexpr Scalar kTargetTieTolerance = Scalar{1e-5f};

bool all_finite(const std::vector<Scalar>& vec) {
    for (Scalar v : vec) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

std::size_t argmax_index(const std::vector<Scalar>& vec) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < vec.size(); ++i) {
        if (vec[i] > vec[best]) {
            best = i;
        }
    }
    return best;
}

void validate_eval_inputs(const Network& network,
                          const Dataset& eval_dataset,
                          const Dataset& baseline_train_dataset,
                          const char* caller) {
    if (network.empty() || eval_dataset.empty() || baseline_train_dataset.empty()) {
        throw std::invalid_argument(std::string(caller) +
                                    " requires non-empty Network, eval Dataset, and baseline Dataset");
    }
    if (eval_dataset.input_size() != network.input_size() ||
        eval_dataset.target_size() != network.output_size()) {
        throw std::invalid_argument(std::string(caller) +
                                    " eval dataset dimensions do not match Network");
    }
    if (baseline_train_dataset.input_size() != network.input_size() ||
        baseline_train_dataset.target_size() != network.output_size()) {
        throw std::invalid_argument(std::string(caller) +
                                    " baseline dataset dimensions do not match Network");
    }
}

void add_json_number_or_null(cJSON* obj, const char* key, double value) {
    if (std::isfinite(value)) {
        if (!cJSON_AddNumberToObject(obj, key, value)) {
            throw std::runtime_error(std::string("Failed to add JSON number '") + key + "'");
        }
    } else {
        if (!cJSON_AddNullToObject(obj, key)) {
            throw std::runtime_error(std::string("Failed to add JSON null '") + key + "'");
        }
    }
}

void add_json_size(cJSON* obj, const char* key, std::size_t value) {
    if (!cJSON_AddNumberToObject(obj, key, static_cast<double>(value))) {
        throw std::runtime_error(std::string("Failed to add JSON size '") + key + "'");
    }
}

void add_json_bool(cJSON* obj, const char* key, bool value) {
    if (!cJSON_AddBoolToObject(obj, key, value ? 1 : 0)) {
        throw std::runtime_error(std::string("Failed to add JSON bool '") + key + "'");
    }
}

void add_json_string(cJSON* obj, const char* key, const std::string& value) {
    if (!cJSON_AddStringToObject(obj, key, value.c_str())) {
        throw std::runtime_error(std::string("Failed to add JSON string '") + key + "'");
    }
}

ModelSummary summarize_model(const Network& network) {
    ModelSummary summary;
    summary.input_size = network.input_size();
    summary.output_size = network.output_size();
    summary.parameters = network.parameter_count();
    summary.layer_sizes.reserve(network.num_layers() + 1);
    summary.layer_activations.reserve(network.num_layers());
    summary.layer_sizes.push_back(network.input_size());
    for (const DenseLayer& layer : network.layers()) {
        summary.layer_sizes.push_back(layer.output_size());
        summary.layer_activations.push_back(nn::to_string(layer.activation()));
    }
    return summary;
}

}  // namespace

TaskType infer_task_type(LossType loss) noexcept {
    switch (loss) {
        case LossType::BinaryCrossEntropy:
        case LossType::CategoricalCrossEntropy:
            return TaskType::Classification;
        case LossType::MSE:
        case LossType::Huber:
            return TaskType::Regression;
    }
    return TaskType::Regression;
}

std::string to_string(TaskType task) {
    switch (task) {
        case TaskType::Classification:
            return "classification";
        case TaskType::Regression:
            return "regression";
    }
    return "unknown";
}

ClassificationMetrics evaluate_classification(const Network& network,
                                              const Dataset& eval_dataset,
                                              const Dataset& baseline_train_dataset,
                                              LossType loss_type) {
    validate_eval_inputs(network, eval_dataset, baseline_train_dataset,
                         "evaluate_classification");

    const std::size_t out_dim = network.output_size();
    const bool is_binary_single_output = (out_dim == 1);
    const std::size_t num_classes = is_binary_single_output ? 2 : out_dim;
    const std::size_t n_train = baseline_train_dataset.size();
    const std::size_t n_eval = eval_dataset.size();

    // 1. Fit leakage-free constant prior and majority class strictly on baseline_train_dataset
    std::vector<double> prior_accum(out_dim, 0.0);
    std::vector<std::size_t> train_class_counts(num_classes, 0);
    for (std::size_t i = 0; i < n_train; ++i) {
        const std::vector<Scalar>& t = baseline_train_dataset.raw_target(i);
        for (std::size_t k = 0; k < out_dim; ++k) {
            prior_accum[k] += static_cast<double>(t[k]);
        }
        if (is_binary_single_output) {
            const std::size_t cls = (t[0] >= Scalar{0.5f}) ? 1 : 0;
            ++train_class_counts[cls];
        } else {
            ++train_class_counts[argmax_index(t)];
        }
    }

    std::vector<Scalar> constant_prior(out_dim, Scalar{0});
    for (std::size_t k = 0; k < out_dim; ++k) {
        constant_prior[k] = static_cast<Scalar>(prior_accum[k] / static_cast<double>(n_train));
    }

    std::size_t train_majority_class = 0;
    for (std::size_t c = 1; c < num_classes; ++c) {
        if (train_class_counts[c] > train_class_counts[train_majority_class]) {
            train_majority_class = c;
        }
    }

    // 2. Single inference pass over eval_dataset
    ClassificationMetrics out;
    out.num_classes = num_classes;
    out.confusion_matrix.assign(num_classes, std::vector<std::size_t>(num_classes, 0));
    out.per_class.resize(num_classes);

    NetworkWorkspace ws = network.create_workspace(false);
    double loss_sum = 0.0;
    double baseline_loss_sum = 0.0;
    std::size_t correct = 0;
    std::size_t majority_correct = 0;
    std::size_t finite_predictions = 0;
    std::vector<double> pred_prob_sum(num_classes, 0.0);

    for (std::size_t i = 0; i < n_eval; ++i) {
        const std::vector<Scalar>& x = eval_dataset.raw_input(i);
        const std::vector<Scalar>& t = eval_dataset.raw_target(i);
        const std::vector<Scalar>& pred = network.forward(x, ws);

        loss_sum += static_cast<double>(loss::compute(loss_type, pred, t));
        baseline_loss_sum += static_cast<double>(loss::compute(loss_type, constant_prior, t));

        if (is_binary_single_output) {
            const std::size_t true_cls = (t[0] >= Scalar{0.5f}) ? 1 : 0;
            if (true_cls == train_majority_class) {
                ++majority_correct;
            }
            if (!all_finite(pred)) {
                continue;
            }
            ++finite_predictions;
            const std::size_t pred_cls = (pred[0] >= Scalar{0.5f}) ? 1 : 0;
            if (pred_cls == true_cls) {
                ++correct;
            }
            ++out.confusion_matrix[true_cls][pred_cls];
            const double p1 = static_cast<double>(pred[0]);
            pred_prob_sum[0] += (1.0 - p1);
            pred_prob_sum[1] += p1;
        } else {
            const Scalar max_target = *std::max_element(t.begin(), t.end());
            if (t[train_majority_class] >= max_target - kTargetTieTolerance) {
                ++majority_correct;
            }
            if (!all_finite(pred)) {
                continue;
            }
            ++finite_predictions;
            const std::size_t pred_cls = argmax_index(pred);
            const bool is_tie_match = (t[pred_cls] >= max_target - kTargetTieTolerance);
            const std::size_t true_cls = is_tie_match ? pred_cls : argmax_index(t);
            if (is_tie_match) {
                ++correct;
            }
            ++out.confusion_matrix[true_cls][pred_cls];
            for (std::size_t c = 0; c < num_classes; ++c) {
                pred_prob_sum[c] += static_cast<double>(pred[c]);
            }
        }
    }

    const double inv_n_eval = 1.0 / static_cast<double>(n_eval);
    out.loss = static_cast<Scalar>(loss_sum * inv_n_eval);
    out.baseline_constant_loss = static_cast<Scalar>(baseline_loss_sum * inv_n_eval);
    out.loss_improvement_vs_baseline =
        static_cast<Scalar>((baseline_loss_sum - loss_sum) * inv_n_eval);
    out.accuracy = static_cast<Scalar>(static_cast<double>(correct) * inv_n_eval);
    out.majority_baseline_accuracy =
        static_cast<Scalar>(static_cast<double>(majority_correct) * inv_n_eval);

    double f1_sum = 0.0;
    for (std::size_t c = 0; c < num_classes; ++c) {
        std::size_t row_sum = 0;
        std::size_t col_sum = 0;
        for (std::size_t j = 0; j < num_classes; ++j) {
            row_sum += out.confusion_matrix[c][j];
            col_sum += out.confusion_matrix[j][c];
        }
        const std::size_t tp = out.confusion_matrix[c][c];

        const double precision =
            (col_sum > 0) ? (static_cast<double>(tp) / static_cast<double>(col_sum)) : 0.0;
        const double recall =
            (row_sum > 0) ? (static_cast<double>(tp) / static_cast<double>(row_sum)) : 0.0;
        const double f1 =
            (precision + recall > 0.0) ? (2.0 * precision * recall / (precision + recall)) : 0.0;

        ClassMetrics cm;
        cm.class_index = c;
        cm.support = row_sum;
        cm.actual_frequency = static_cast<Scalar>(static_cast<double>(row_sum) * inv_n_eval);
        cm.mean_predicted_prob =
            (finite_predictions > 0)
                ? static_cast<Scalar>(pred_prob_sum[c] / static_cast<double>(finite_predictions))
                : Scalar{0};
        cm.precision = static_cast<Scalar>(precision);
        cm.recall = static_cast<Scalar>(recall);
        cm.f1 = static_cast<Scalar>(f1);

        out.per_class[c] = cm;
        f1_sum += f1;
    }

    out.macro_f1 = static_cast<Scalar>(f1_sum / static_cast<double>(num_classes));
    return out;
}

RegressionMetrics evaluate_regression(const Network& network,
                                      const Dataset& eval_dataset,
                                      const Dataset& baseline_train_dataset,
                                      LossType loss_type,
                                      Scalar huber_delta) {
    validate_eval_inputs(network, eval_dataset, baseline_train_dataset, "evaluate_regression");

    const std::size_t out_dim = network.output_size();
    const std::size_t n_train = baseline_train_dataset.size();
    const std::size_t n_eval = eval_dataset.size();

    // 1. Fit constant mean-target baseline strictly on baseline_train_dataset
    std::vector<double> train_sum(out_dim, 0.0);
    for (std::size_t i = 0; i < n_train; ++i) {
        const std::vector<Scalar>& t = baseline_train_dataset.raw_target(i);
        for (std::size_t k = 0; k < out_dim; ++k) {
            train_sum[k] += static_cast<double>(t[k]);
        }
    }
    std::vector<Scalar> train_mean(out_dim, Scalar{0});
    for (std::size_t k = 0; k < out_dim; ++k) {
        train_mean[k] = static_cast<Scalar>(train_sum[k] / static_cast<double>(n_train));
    }

    // 2. Compute evaluation target means for numerically stable two-pass SS_tot
    std::vector<double> eval_target_mean(out_dim, 0.0);
    double eval_global_sum = 0.0;
    for (std::size_t i = 0; i < n_eval; ++i) {
        const std::vector<Scalar>& t = eval_dataset.raw_target(i);
        for (std::size_t k = 0; k < out_dim; ++k) {
            const double tv = static_cast<double>(t[k]);
            eval_target_mean[k] += tv;
            eval_global_sum += tv;
        }
    }
    for (std::size_t k = 0; k < out_dim; ++k) {
        eval_target_mean[k] /= static_cast<double>(n_eval);
    }
    const double eval_global_mean =
        eval_global_sum / static_cast<double>(n_eval * out_dim);

    // 3. Single inference pass over eval_dataset
    NetworkWorkspace ws = network.create_workspace(false);
    double loss_sum = 0.0;
    double baseline_loss_sum = 0.0;
    std::vector<double> abs_err_sum(out_dim, 0.0);
    std::vector<double> sq_err_sum(out_dim, 0.0);
    std::vector<double> ss_tot_per_output(out_dim, 0.0);
    double ss_tot_global = 0.0;

    for (std::size_t i = 0; i < n_eval; ++i) {
        const std::vector<Scalar>& x = eval_dataset.raw_input(i);
        const std::vector<Scalar>& t = eval_dataset.raw_target(i);
        const std::vector<Scalar>& pred = network.forward(x, ws);

        loss_sum += static_cast<double>(loss::compute(loss_type, pred, t, huber_delta));
        baseline_loss_sum +=
            static_cast<double>(loss::compute(loss_type, train_mean, t, huber_delta));

        for (std::size_t k = 0; k < out_dim; ++k) {
            const double pv = static_cast<double>(pred[k]);
            const double tv = static_cast<double>(t[k]);
            const double err = pv - tv;
            abs_err_sum[k] += std::abs(err);
            sq_err_sum[k] += err * err;

            const double diff_k = tv - eval_target_mean[k];
            ss_tot_per_output[k] += diff_k * diff_k;

            const double diff_global = tv - eval_global_mean;
            ss_tot_global += diff_global * diff_global;
        }
    }

    RegressionMetrics out;
    out.num_outputs = out_dim;
    const double inv_n_eval = 1.0 / static_cast<double>(n_eval);
    out.loss = static_cast<Scalar>(loss_sum * inv_n_eval);
    out.baseline_constant_loss = static_cast<Scalar>(baseline_loss_sum * inv_n_eval);
    out.loss_improvement_vs_baseline =
        static_cast<Scalar>((baseline_loss_sum - loss_sum) * inv_n_eval);

    double total_abs_err = 0.0;
    double total_sq_err = 0.0;
    out.per_output.resize(out_dim);
    for (std::size_t k = 0; k < out_dim; ++k) {
        total_abs_err += abs_err_sum[k];
        total_sq_err += sq_err_sum[k];

        OutputRegressionMetrics om;
        om.output_index = k;
        om.mae = static_cast<Scalar>(abs_err_sum[k] * inv_n_eval);
        om.rmse = static_cast<Scalar>(std::sqrt(sq_err_sum[k] * inv_n_eval));
        om.r2 = (ss_tot_per_output[k] > 0.0)
                    ? static_cast<Scalar>(1.0 - sq_err_sum[k] / ss_tot_per_output[k])
                    : Scalar{0};
        out.per_output[k] = om;
    }

    const double total_elems = static_cast<double>(n_eval * out_dim);
    out.mae = static_cast<Scalar>(total_abs_err / total_elems);
    out.rmse = static_cast<Scalar>(std::sqrt(total_sq_err / total_elems));
    out.r2 = (ss_tot_global > 0.0) ? static_cast<Scalar>(1.0 - total_sq_err / ss_tot_global)
                                   : Scalar{0};
    return out;
}

EvaluationReport build_evaluation_report(const Network& network,
                                         const Dataset& train_dataset,
                                         const Dataset* val_dataset,
                                         const TrainingConfig& training_config,
                                         const TrainingResult& training_result,
                                         double elapsed_ms) {
    const bool use_validation = (val_dataset != nullptr && !val_dataset->empty());
    const Dataset& eval_ds = use_validation ? *val_dataset : train_dataset;

    EvaluationReport report;
    report.format_version = kReportFormatVersion;
    report.task = infer_task_type(training_config.loss);
    report.eval_split = use_validation ? "validation" : "train";
    report.elapsed_ms = elapsed_ms;
    report.train_samples = train_dataset.size();
    report.val_samples = use_validation ? val_dataset->size() : 0;
    report.model = summarize_model(network);
    report.training_config = training_config;
    report.training_result = training_result;

    if (report.task == TaskType::Classification) {
        report.classification =
            evaluate_classification(network, eval_ds, train_dataset, training_config.loss);
    } else {
        report.regression = evaluate_regression(network, eval_ds, train_dataset,
                                                training_config.loss, training_config.huber_delta);
    }

    return report;
}

std::string report_to_json(const EvaluationReport& report, bool pretty) {
    CJsonPtr root(cJSON_CreateObject(), &cJSON_Delete);
    if (!root) {
        throw std::runtime_error("Failed to create root cJSON object for report");
    }

    add_json_size(root.get(), "format_version", static_cast<std::size_t>(report.format_version));
    add_json_string(root.get(), "task", to_string(report.task));
    add_json_string(root.get(), "eval_split", report.eval_split);

    // 1. Model section
    cJSON* model_obj = cJSON_AddObjectToObject(root.get(), "model");
    if (!model_obj) {
        throw std::runtime_error("Failed to create report 'model' object");
    }
    add_json_size(model_obj, "input_size", report.model.input_size);
    add_json_size(model_obj, "output_size", report.model.output_size);
    add_json_size(model_obj, "parameters", report.model.parameters);
    cJSON* topo_arr = cJSON_AddArrayToObject(model_obj, "topology");
    if (!topo_arr) {
        throw std::runtime_error("Failed to create report 'model.topology' array");
    }
    for (std::size_t i = 0; i < report.model.layer_sizes.size(); ++i) {
        cJSON* entry = cJSON_CreateObject();
        if (!entry) {
            throw std::runtime_error("Failed to create report topology entry");
        }
        cJSON_AddItemToArray(topo_arr, entry);
        add_json_size(entry, "size", report.model.layer_sizes[i]);
        if (i > 0 && (i - 1) < report.model.layer_activations.size()) {
            add_json_string(entry, "activation", report.model.layer_activations[i - 1]);
        }
    }

    // 2. Training summary section
    cJSON* train_obj = cJSON_AddObjectToObject(root.get(), "training");
    if (!train_obj) {
        throw std::runtime_error("Failed to create report 'training' object");
    }
    const TrainingConfig& cfg = report.training_config;
    const TrainingResult& res = report.training_result;
    add_json_string(train_obj, "loss", nn::to_string(cfg.loss));
    add_json_number_or_null(train_obj, "huber_delta", static_cast<double>(cfg.huber_delta));
    add_json_string(train_obj, "optimizer", nn::to_string(cfg.optimizer));
    add_json_size(train_obj, "epochs_configured", cfg.epochs);
    add_json_size(train_obj, "epochs_completed", res.epochs_completed);
    add_json_bool(train_obj, "early_stopped", res.early_stopped);
    add_json_size(train_obj, "batch_size", cfg.batch_size);
    add_json_number_or_null(train_obj, "learning_rate", static_cast<double>(cfg.learning_rate));
    add_json_string(train_obj, "lr_schedule", nn::to_string(cfg.lr_schedule));
    add_json_number_or_null(train_obj, "min_learning_rate",
                            static_cast<double>(cfg.min_learning_rate));
    add_json_number_or_null(train_obj, "weight_decay", static_cast<double>(cfg.weight_decay));
    add_json_number_or_null(train_obj, "dropout_rate", static_cast<double>(cfg.dropout_rate));
    add_json_number_or_null(train_obj, "grad_clip_norm", static_cast<double>(cfg.grad_clip_norm));
    add_json_size(train_obj, "early_stopping_patience", cfg.early_stopping_patience);
    add_json_number_or_null(train_obj, "early_stopping_min_delta",
                            static_cast<double>(cfg.early_stopping_min_delta));
    add_json_number_or_null(train_obj, "validation_split",
                            static_cast<double>(cfg.validation_split));
    add_json_bool(train_obj, "validation_shuffle", cfg.validation_shuffle);
    add_json_bool(train_obj, "keep_best_validation_model", cfg.keep_best_validation_model);
    add_json_size(train_obj, "seed", static_cast<std::size_t>(cfg.seed));
    add_json_size(train_obj, "num_threads", cfg.num_threads);
    add_json_size(train_obj, "train_samples", report.train_samples);
    add_json_size(train_obj, "val_samples", report.val_samples);
    add_json_number_or_null(train_obj, "elapsed_ms", report.elapsed_ms);
    add_json_number_or_null(train_obj, "initial_train_loss", static_cast<double>(res.initial_loss));
    add_json_number_or_null(train_obj, "final_train_loss", static_cast<double>(res.final_loss));
    add_json_bool(train_obj, "has_validation", res.has_validation);
    if (res.has_validation) {
        add_json_size(train_obj, "best_val_epoch", res.best_val_epoch);
        add_json_number_or_null(train_obj, "best_val_loss", static_cast<double>(res.best_val_loss));
        add_json_number_or_null(train_obj, "final_val_loss",
                                static_cast<double>(res.final_val_loss));
    } else {
        cJSON_AddNullToObject(train_obj, "best_val_epoch");
        cJSON_AddNullToObject(train_obj, "best_val_loss");
        cJSON_AddNullToObject(train_obj, "final_val_loss");
    }

    // 3. Evaluation metrics section
    cJSON* eval_obj = cJSON_AddObjectToObject(root.get(), "evaluation");
    if (!eval_obj) {
        throw std::runtime_error("Failed to create report 'evaluation' object");
    }

    if (report.task == TaskType::Classification) {
        const ClassificationMetrics& cm = report.classification;
        add_json_size(eval_obj, "num_classes", cm.num_classes);
        add_json_number_or_null(eval_obj, "loss", static_cast<double>(cm.loss));
        add_json_number_or_null(eval_obj, "baseline_constant_loss",
                                static_cast<double>(cm.baseline_constant_loss));
        add_json_number_or_null(eval_obj, "loss_improvement_vs_baseline",
                                static_cast<double>(cm.loss_improvement_vs_baseline));
        add_json_number_or_null(eval_obj, "accuracy", static_cast<double>(cm.accuracy));
        add_json_number_or_null(eval_obj, "majority_baseline_accuracy",
                                static_cast<double>(cm.majority_baseline_accuracy));
        add_json_number_or_null(eval_obj, "macro_f1", static_cast<double>(cm.macro_f1));

        cJSON* matrix_arr = cJSON_AddArrayToObject(eval_obj, "confusion_matrix");
        if (!matrix_arr) {
            throw std::runtime_error("Failed to create report 'confusion_matrix' array");
        }
        for (const auto& row : cm.confusion_matrix) {
            cJSON* row_arr = cJSON_CreateArray();
            if (!row_arr) {
                throw std::runtime_error("Failed to create confusion_matrix row array");
            }
            cJSON_AddItemToArray(matrix_arr, row_arr);
            for (std::size_t count : row) {
                cJSON* num = cJSON_CreateNumber(static_cast<double>(count));
                if (!num) {
                    throw std::runtime_error("Failed to create confusion_matrix cell");
                }
                cJSON_AddItemToArray(row_arr, num);
            }
        }

        cJSON* per_class_arr = cJSON_AddArrayToObject(eval_obj, "per_class");
        if (!per_class_arr) {
            throw std::runtime_error("Failed to create report 'per_class' array");
        }
        for (const ClassMetrics& item : cm.per_class) {
            cJSON* cls_obj = cJSON_CreateObject();
            if (!cls_obj) {
                throw std::runtime_error("Failed to create report per_class object");
            }
            cJSON_AddItemToArray(per_class_arr, cls_obj);
            add_json_size(cls_obj, "class_index", item.class_index);
            add_json_size(cls_obj, "support", item.support);
            add_json_number_or_null(cls_obj, "actual_frequency",
                                    static_cast<double>(item.actual_frequency));
            add_json_number_or_null(cls_obj, "mean_predicted_prob",
                                    static_cast<double>(item.mean_predicted_prob));
            add_json_number_or_null(cls_obj, "precision", static_cast<double>(item.precision));
            add_json_number_or_null(cls_obj, "recall", static_cast<double>(item.recall));
            add_json_number_or_null(cls_obj, "f1", static_cast<double>(item.f1));
        }
    } else {
        const RegressionMetrics& rm = report.regression;
        add_json_size(eval_obj, "num_outputs", rm.num_outputs);
        add_json_number_or_null(eval_obj, "loss", static_cast<double>(rm.loss));
        add_json_number_or_null(eval_obj, "baseline_constant_loss",
                                static_cast<double>(rm.baseline_constant_loss));
        add_json_number_or_null(eval_obj, "loss_improvement_vs_baseline",
                                static_cast<double>(rm.loss_improvement_vs_baseline));
        add_json_number_or_null(eval_obj, "mae", static_cast<double>(rm.mae));
        add_json_number_or_null(eval_obj, "rmse", static_cast<double>(rm.rmse));
        add_json_number_or_null(eval_obj, "r2", static_cast<double>(rm.r2));

        cJSON* per_out_arr = cJSON_AddArrayToObject(eval_obj, "per_output");
        if (!per_out_arr) {
            throw std::runtime_error("Failed to create report 'per_output' array");
        }
        for (const OutputRegressionMetrics& item : rm.per_output) {
            cJSON* out_obj = cJSON_CreateObject();
            if (!out_obj) {
                throw std::runtime_error("Failed to create report per_output object");
            }
            cJSON_AddItemToArray(per_out_arr, out_obj);
            add_json_size(out_obj, "output_index", item.output_index);
            add_json_number_or_null(out_obj, "mae", static_cast<double>(item.mae));
            add_json_number_or_null(out_obj, "rmse", static_cast<double>(item.rmse));
            add_json_number_or_null(out_obj, "r2", static_cast<double>(item.r2));
        }
    }

    // 4. Epoch history section
    cJSON* hist_arr = cJSON_AddArrayToObject(root.get(), "history");
    if (!hist_arr) {
        throw std::runtime_error("Failed to create report 'history' array");
    }
    for (std::size_t i = 0; i < res.loss_history.size(); ++i) {
        cJSON* entry = cJSON_CreateObject();
        if (!entry) {
            throw std::runtime_error("Failed to create report history entry");
        }
        cJSON_AddItemToArray(hist_arr, entry);
        add_json_size(entry, "epoch", i + 1);
        add_json_number_or_null(entry, "train_loss", static_cast<double>(res.loss_history[i]));
        if (i < res.val_loss_history.size()) {
            add_json_number_or_null(entry, "val_loss",
                                    static_cast<double>(res.val_loss_history[i]));
        }
        if (i < res.lr_history.size()) {
            add_json_number_or_null(entry, "lr", static_cast<double>(res.lr_history[i]));
        }
    }

    CStringPtr printed(pretty ? cJSON_Print(root.get()) : cJSON_PrintUnformatted(root.get()),
                       &std::free);
    if (!printed) {
        throw std::runtime_error("Failed to render evaluation report JSON");
    }
    return std::string(printed.get());
}

void save_report_json(const EvaluationReport& report, const std::string& path, bool pretty) {
    const std::string json_text = report_to_json(report, pretty);

    const std::filesystem::path file_path(path);
    if (file_path.has_parent_path()) {
        std::filesystem::create_directories(file_path.parent_path());
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open report file for writing: '" + path + "'");
    }
    out << json_text << '\n';
    if (!out.good()) {
        throw std::runtime_error("Failed while writing report file: '" + path + "'");
    }
}

}  // namespace metrics
}  // namespace nn
