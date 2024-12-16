#! /usr/bin/env python3 
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import json


def read_performance_json(inputs):
    PERF_FILENAME="performance.json"
    performance_data = []
    for inp in inputs:
        inp = pathlib.Path(inp)
        if inp.is_file() and inp.name == PERF_FILENAME:
            performance_data.append(json.load(inp))
        elif inp.is_dir():
            for file in pathlib.Path(inp).rglob("performance.json"):
                with open(file, 'r') as f:
                    performance_data.append(json.load(f))
    df = pd.DataFrame.from_dict(performance_data)
    return df

def main():
    parser = argparse.ArgumentParser(description="Plotting script for runtime/performance data")
    parser.add_argument("inputs", type=pathlib.Path, nargs="+", help="Json files to plot")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Path to output image location")
    args = parser.parse_args()
    print(args)

    df = read_performance_json(args.inputs)

    sns.set_palette("Dark2")
    sns.set_context("talk")
    sns.set_style("darkgrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)

    g0 = sns.lineplot(df, ax=axes[0], x="n_total",y="totalProgram", style="device_name")
    axes[0].set_title("Total Runtime (s) vs population")
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)


    # Copy some columns from the dataframe
    dfm =df[["device_name", "n_total", "configParsing", "simulate", "preSimulate", "postSimulate", "flamegpuSimulateElapsed"]].copy()
    # Drop some columns pre-melt
    dfm = dfm.melt(id_vars=["device_name", "n_total"], var_name="metric", value_name = "count")
    print(dfm)


    # for y in y_cols:
    g1 = sns.lineplot(dfm, ax=axes[1], x="n_total", y="count", hue="metric", style="device_name")
    axes[1].set_title("Split timing information")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    
    if (args.output):
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()