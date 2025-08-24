import json
import os
import pandas as pd
from scipy import stats
import warnings


"""
Experiment structure:
    - experiment_dir
        - dataset_name
            - model_name
                - results.json

Results structure:
    Json file containing each metric and its value
"""


def generate_latex_row(
    results,
    all_metrics,
    all_model_names,
    dataset_name,
    dataset_name_map=None,
    experiment_dir=None,
):
    if dataset_name_map is not None:
        dataset_latex_name = dataset_name_map[dataset_name]
    else:
        dataset_latex_name = dataset_name
    dataset_latex_name = dataset_latex_name.replace("_", "\\_")
    row = f"\t\t{dataset_latex_name} & "

    for metric in all_metrics:
        all_scores = [
            results[model_name][dataset_name][metric]
            if dataset_name in results[model_name]
            and metric in results[model_name][dataset_name]
            else 0
            for model_name in all_model_names
        ]
        best_score = max(all_scores)
        best_score_index = all_scores.index(best_score)
        best_model_name = all_model_names[best_score_index]
        try:
            second_best_score = max(
                [score for score in all_scores if score != best_score]
            )
        except ValueError:
            # If all scores are the same, there is no second best score
            second_best_score = best_score
        second_best_score_index = all_scores.index(second_best_score)
        second_best_model_name = all_model_names[second_best_score_index]

        for model_name in all_model_names:
            if (
                dataset_name in results[model_name]
                and metric in results[model_name][dataset_name]
            ):
                if results[model_name][dataset_name][metric] == best_score:
                    pvalue = calculate_significance(
                        os.path.join(
                            experiment_dir,
                            dataset_name,
                            model_name,
                            f"{metric}.json",
                        ),
                        os.path.join(
                            experiment_dir,
                            dataset_name,
                            second_best_model_name,
                            f"{metric}.json",
                        ),
                    )
                    # row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}} & "
                    if pvalue < 0.01:
                        # Add \rlap{\textsuperscript{**}} after the score
                        row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}}\\rlap{{\\textsuperscript{{**}}}} & "
                    elif pvalue < 0.05:
                        # Add \rlap{\textsuperscript{*}} after the score
                        row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}}\\rlap{{\\textsuperscript{{*}}}} & "
                    else:
                        row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}} & "
                elif (
                    model_name == second_best_model_name
                    and best_model_name != second_best_model_name
                    and len(all_scores) > 2
                ):
                    row += f"\\textit{{{results[model_name][dataset_name][metric]:.3f}}} & "

                else:
                    row += f"{results[model_name][dataset_name][metric]:.3f} & "
            else:
                row += "~ & "
        row += "~ & "
    row = row[:-7] + "\\\\\n"
    return row


def generate_flipped_latex_row(
    results,
    all_metrics,
    all_dataset_names,
    model_name,
    model_name_map=None,
    experiment_dir=None,
):
    if model_name_map is not None:
        model_name_to_use = model_name_map[model_name] if model_name_map else model_name
    else:
        model_name_to_use = model_name
    model_name_to_use = model_name_to_use.replace("_", "\\_")
    row = f"\t\t{model_name_to_use} & "

    for metric in all_metrics:
        all_scores = [
            results[model_name][dataset_name][metric]
            if dataset_name in results[model_name]
            and metric in results[model_name][dataset_name]
            else 0
            for dataset_name in all_dataset_names
        ]
        best_score = max(all_scores)
        best_score_index = all_scores.index(best_score)
        best_dataset_name = all_dataset_names[best_score_index]
        try:
            second_best_score = max(
                [score for score in all_scores if score != best_score]
            )
        except ValueError:
            # If all scores are the same, there is no second best score
            second_best_score = best_score
        second_best_score_index = all_scores.index(second_best_score)
        second_best_dataset_name = all_dataset_names[second_best_score_index]

        for dataset_name in all_dataset_names:
            if (
                dataset_name in results[model_name]
                and metric in results[model_name][dataset_name]
            ):
                if results[model_name][dataset_name][metric] == best_score:
                    pvalue = calculate_significance(
                        os.path.join(
                            experiment_dir,
                            dataset_name,
                            model_name,
                            f"{metric}.json",
                        ),
                        os.path.join(
                            experiment_dir,
                            second_best_dataset_name,
                            model_name,
                            f"{metric}.json",
                        ),
                    )
                    if isinstance(results[model_name][dataset_name][metric], float):
                        row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}} & "
                    else:
                        row += f"\\textbf{{{results[model_name][dataset_name][metric]}}} & "

                    # if pvalue < 0.01:
                    #     # Add \rlap{\textsuperscript{**}} after the score
                    #     row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}}\\rlap{{\\textsuperscript{{**}}}} & "
                    # elif pvalue < 0.05:
                    #     # Add \rlap{\textsuperscript{*}} after the score
                    #     row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}}\\rlap{{\\textsuperscript{{*}}}} & "
                    # else:
                    #     row += f"\\textbf{{{results[model_name][dataset_name][metric]:.3f}}} & "
                elif (
                    dataset_name == second_best_dataset_name
                    and best_dataset_name != second_best_dataset_name
                    and len(all_scores) > 2
                ):
                    row += f"\\textit{{{results[model_name][dataset_name][metric]:.3f}}} & "

                else:
                    row += f"{results[model_name][dataset_name][metric]:.3f} & "
            else:
                row += "~ & "

        row += "~ & "
    row = row[:-7] + "\\\\\n"
    return row


def generate_latex_table(
    results,
    all_metrics,
    all_model_names,
    dataset_names,
    caption="Experiment results.",
    model_name_map=None,
    dataset_name_map=None,
    metric_name_map=None,
    experiment_dir=None,
    table_label="",
    table_star=False,
    fontsize="small",
    tabcolsep="1.5pt",
    dataset_order=None,
):
    n_models = len(all_model_names)
    # Generate the header row with metrics as main columns and models as sub-columns
    header = "\\toprule \n\t\t ~ & "
    for metric in all_metrics:
        if metric_name_map is not None and metric in metric_name_map:
            metric = metric_name_map[metric]
        metric = metric.replace("_", "\\_")
        header += f"\\multicolumn{{{n_models}}}{{c}}{{{metric}}} & ~ & "

    header = header[:-7] + "\\\\\n"

    # Seperate metrics from models
    metric_start_col = 2
    header_sep_line = "\t\t"
    for metric in all_metrics:
        header_sep_line += (
            f"\\cmidrule{{{metric_start_col}-{metric_start_col + n_models - 1}}}"
        )
        metric_start_col += n_models + 1
    header_sep_line += "\n"

    header += header_sep_line

    # Dataset and model name header
    header += "\t\tDataset & "
    for _ in range(len(all_metrics)):
        for model_name in all_model_names:
            if model_name_map is not None and model_name in model_name_map:
                model_name = model_name_map[model_name]
            model_name = model_name.replace("_", "\\_")
            header += f"{model_name} & "
        header += "~ & "

    header = header[:-7] + "\\\\\n" + "\t\t\\midrule\n"

    if dataset_order:
        if set(dataset_order) != set(dataset_names):
            raise ValueError(
                f"dataset_order ({dataset_order}) must contain the same datasets as dataset_names ({dataset_names})"
            )
        dataset_names = dataset_order
    else:
        # Sort datasets by name
        dataset_names = sorted(dataset_names)

    # Generate rows
    rows = ""
    for dataset_name in dataset_names:
        rows += generate_latex_row(
            results,
            all_metrics,
            all_model_names,
            dataset_name,
            dataset_name_map,
            experiment_dir,
        )

    # Generate bottom line
    bottom_line = "\\bottomrule\n"

    # Skip wins for now

    # Generate table
    table = "\\begin{table*}[t]\n" if table_star else "\\begin{table}[t]\n"
    table += f"\t\\{fontsize}\n"
    table += f"\t\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
    table += "\t\\centering\n"
    table += f"\t\\caption{{{caption}}}\n"

    if table_label != "":
        table += f"\t\\label{{{table_label}}}\n"

    table += "\t\\begin{tabular}{l"
    model_columns = "c" * len(all_model_names)
    for _ in range(len(all_metrics)):
        table += f" {model_columns} l"

    table = table[:-2] + "}\n"

    table += "\t\t" + header
    table += rows
    table += "\t\t" + bottom_line
    table += "\t\\end{tabular}\n"
    table += "\\end{table*}\n" if table_star else "\\end{table}\n"

    return table


def generate_flipped_latex_table(
    results,
    all_metrics,
    all_model_names,
    dataset_names,
    caption="Experiment results.",
    model_name_map=None,
    dataset_name_map=None,
    metric_name_map=None,
    experiment_dir=None,
    table_label="",
    table_star=False,
    fontsize="small",
    tabcolsep="1.5pt",
):
    n_datasets = len(dataset_names)
    # Generate the header row with metrics as main columns and datasets as sub-columns
    header = "\\toprule \n\t\t ~ & "
    for metric in all_metrics:
        if metric_name_map is not None and metric in metric_name_map:
            metric = metric_name_map[metric]
        metric = metric.replace("_", "\\_")
        header += f"\\multicolumn{{{n_datasets}}}{{c}}{{{metric}}} & ~ & "

    header = header[:-7] + "\\\\\n"

    # Seperate metrics from datasets
    metric_start_col = 2
    header_sep_line = "\t\t"
    for metric in all_metrics:
        header_sep_line += (
            f"\\cmidrule{{{metric_start_col}-{metric_start_col + n_datasets - 1}}}"
        )
        metric_start_col += n_datasets + 1
    header_sep_line += "\n"

    header += header_sep_line

    # Dataset and model name header
    header += "\t\tModel & "
    for _ in range(len(all_metrics)):
        for dataset_name in dataset_names:
            if dataset_name_map is not None and dataset_name in dataset_name_map:
                dataset_name = dataset_name_map[dataset_name]
            dataset_name = dataset_name.replace("_", "\\_")
            header += f"{dataset_name} & "
        header += "~ & "

    header = header[:-7] + "\\\\\n" + "\t\t\\midrule\n"

    # Sort datasets by name
    dataset_names = sorted(dataset_names)

    # Generate rows
    rows = ""
    for model_name in all_model_names:
        rows += generate_flipped_latex_row(
            results,
            all_metrics,
            dataset_names,
            model_name,
            model_name_map,
            experiment_dir,
        )

    # Generate bottom line
    bottom_line = "\\bottomrule\n"

    # Skip wins for now

    # Generate table
    table = "\\begin{table*}[t]\n" if table_star else "\\begin{table}[t]\n"
    table += f"\t\\{fontsize}\n"
    table += f"\t\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
    table += "\t\\centering\n"
    table += f"\t\\caption{{{caption}}}\n"

    if table_label != "":
        table += f"\t\\label{{{table_label}}}\n"

    table += "\t\\begin{tabular}{l"
    dataset_columns = "c" * len(dataset_names)
    for _ in range(len(all_metrics)):
        table += f" {dataset_columns} l"

    table = table[:-2] + "}\n"

    table += "\t\t" + header
    table += rows
    table += "\t\t" + bottom_line
    table += "\t\\end{tabular}\n"

    table += "\\end{table*}\n" if table_star else "\\end{table}\n"

    return table


def analyze_experiment(
    experiment_dir,
    model_name_map=None,
    dataset_name_map=None,
    skip_models=None,
    keep_named_only=False,
    metrics_to_show=None,
    metric_name_map=None,
    table_label="",
    table_caption="",
    table_star=False,
    fontsize="small",
    tabcolsep="1.5pt",
    dataset_order=None,
    flip_table=False,  # Added flag for flipping table
):
    dataset_names = os.listdir(experiment_dir)
    all_model_names = set()

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(experiment_dir, dataset_name)
        model_names = os.listdir(dataset_dir)
        all_model_names.update(model_names)

    all_model_names = sorted(list(all_model_names))

    if keep_named_only:
        all_model_names = [
            model_name for model_name in all_model_names if model_name in model_name_map
        ]
    elif skip_models is not None:
        all_model_names = [
            model_name
            for model_name in all_model_names
            if model_name not in skip_models
        ]

    all_metrics = set()
    results = {}
    for model_name in all_model_names:
        model_results = {}
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(experiment_dir, dataset_name)
            model_dir = os.path.join(dataset_dir, model_name)
            results_file = os.path.join(model_dir, "results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    model_results[dataset_name] = json.load(f)
                    all_metrics.update(model_results[dataset_name].keys())
        results[model_name] = model_results

    all_metrics = sorted(list(all_metrics))

    if metrics_to_show is not None:
        all_metrics = [metric for metric in all_metrics if metric in metrics_to_show]
        # Same order as metrics_to_show
        all_metrics = sorted(all_metrics, key=lambda x: metrics_to_show.index(x))

    if flip_table:
        latex_table = generate_flipped_latex_table(
            results,
            all_metrics,
            all_model_names,
            dataset_names,
            caption=table_caption,
            model_name_map=model_name_map,
            dataset_name_map=dataset_name_map,
            metric_name_map=metric_name_map,
            experiment_dir=experiment_dir,
            table_label=table_label,
            table_star=table_star,
            fontsize=fontsize,
            tabcolsep=tabcolsep,
        )
    else:
        latex_table = generate_latex_table(
            results,
            all_metrics,
            all_model_names,
            dataset_names,
            model_name_map=model_name_map,
            dataset_name_map=dataset_name_map,
            metric_name_map=metric_name_map,
            experiment_dir=experiment_dir,
            caption=table_caption,
            table_label=table_label,
            table_star=table_star,
            fontsize=fontsize,
            tabcolsep=tabcolsep,
            dataset_order=dataset_order,
        )

    return results, latex_table


def get_per_metric_results(
    experiment_dir,
    model_name_map=None,
    dataset_name_map=None,
    skip_models=None,
    keep_named_only=False,
    metrics_to_show=None,
    metric_name_map=None,
):
    dataset_names = os.listdir(experiment_dir)
    all_model_names = set()

    for dataset_name in dataset_names:
        dataset_dir = os.path.join(experiment_dir, dataset_name)
        model_names = os.listdir(dataset_dir)
        all_model_names.update(model_names)

    all_model_names = sorted(list(all_model_names))

    if keep_named_only:
        all_model_names = [
            model_name for model_name in all_model_names if model_name in model_name_map
        ]
    elif skip_models is not None:
        all_model_names = [
            model_name
            for model_name in all_model_names
            if model_name not in skip_models
        ]

    all_metrics = set()
    results = {}
    for model_name in all_model_names:
        model_results = {}
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(experiment_dir, dataset_name)
            model_dir = os.path.join(dataset_dir, model_name)
            results_file = os.path.join(model_dir, "results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    model_results[dataset_name] = json.load(f)
                    all_metrics.update(model_results[dataset_name].keys())
        results[model_name] = model_results

    all_metrics = sorted(list(all_metrics))

    if metrics_to_show is not None:
        all_metrics = [metric for metric in all_metrics if metric in metrics_to_show]
        # Same order as metrics_to_show
        all_metrics = sorted(all_metrics, key=lambda x: metrics_to_show.index(x))

    per_metric_dicts = {}
    for metric in all_metrics:
        per_metric_results = {}
        for model_name in all_model_names:
            model_results = {}
            for dataset_name in dataset_names:
                if dataset_name in results[model_name] and metric in results[model_name][dataset_name]:
                    model_results[dataset_name] = results[model_name][dataset_name][metric]
                else:
                    model_results[dataset_name] = 0
            per_metric_results[model_name] = model_results
        per_metric_dicts[metric] = per_metric_results

    per_metrics_dfs = {}
    for metric in all_metrics:
        per_metric_df = pd.DataFrame(per_metric_dicts[metric])
        per_metric_df = per_metric_df.T

        # Rename columns (datasets)
        if dataset_name_map is not None:
            per_metric_df.rename(columns=dataset_name_map, inplace=True)

        # Rename rows (models)
        if model_name_map is not None:
            per_metric_df.rename(index=model_name_map, inplace=True)

        # Add Avg column
        per_metric_df["Avg"] = per_metric_df.mean(axis=1)

        # Round to 3 decimal places
        per_metric_df = per_metric_df.round(3)

        metric = metric_name_map.get(metric, metric) if metric_name_map else metric
        per_metrics_dfs[metric] = per_metric_df

    return per_metrics_dfs


def calculate_significance(run_a, run_b):
    try:
        with open(run_a, "r") as f:
            run_a_results = json.load(f)

        with open(run_b, "r") as f:
            run_b_results = json.load(f)
    except FileNotFoundError:
        warnings.warn(f"File not found: {run_a} or {run_b}")
        return 1

    run_a_scores = []
    run_b_scores = []

    # check if dictionary keys are identical
    try:
        assert set(run_a_results.keys()) == set(run_b_results.keys())
    except AssertionError:
        warnings.warn(
            f"Keys in {run_a} and {run_b} are not identical. Returning p-value of 1."
        )
        return 1

    for query_id in run_a_results:
        run_a_scores.append(run_a_results[query_id])
        run_b_scores.append(run_b_results[query_id])

    _, pvalue = stats.ttest_rel(run_a_scores, run_b_scores)

    return pvalue


def convert_trec_queries_to_beir_format(
    trec_queries_path, beir_queries_path=None, save=True
):
    trec_queries_df = pd.read_csv(trec_queries_path, sep="\t", names=["_id", "text"])
    trec_queries_df["_id"] = trec_queries_df["_id"].astype(str)
    trec_queries_df["text"] = trec_queries_df["text"].astype(str)
    if beir_queries_path is None:
        beir_queries_path = os.path.join(
            os.path.dirname(trec_queries_path),
            os.path.basename(trec_queries_path).replace(".tsv", ".jsonl"),
        )
    if save:
        trec_queries_df.to_json(beir_queries_path, orient="records", lines=True)

    return trec_queries_df


def convert_trec_qrels_to_beir_format(trec_qrels_path, beir_qrels_path=None, save=True):
    trec_qrel_df = pd.read_csv(
        trec_qrels_path, sep=" ", names=["query-id", "Q0", "corpus-id", "score"]
    )
    trec_qrel_df["query-id"] = trec_qrel_df["query-id"].astype(int)
    trec_qrel_df["corpus-id"] = trec_qrel_df["corpus-id"].astype(int)
    trec_qrel_df["score"] = trec_qrel_df["score"].astype(int)

    if beir_qrels_path is None:
        # Extension for trec_qrels_path can be .txt or .tsv
        beir_qrels_path = os.path.join(
            os.path.dirname(trec_qrels_path),
            os.path.basename(trec_qrels_path)
            .replace(".txt", ".tsv")
            .replace(".tsv", ".tsv"),
        )

    beir_qrels_df = trec_qrel_df[["query-id", "corpus-id", "score"]]

    if save:
        beir_qrels_df.to_csv(beir_qrels_path, sep="\t", index=False)

    return beir_qrels_df
