#! /usr/bin/env python3 
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib


def main():
    parser = argparse.ArgumentParser(description="Plotting script for time series data")
    parser.add_argument("csv", type=pathlib.Path, help="Time series CSV to plot")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Path to output image location")
    args = parser.parse_args()
    

    df = pd.read_csv(args.csv)

    max_time = df["time"].max()

    df = df.drop("total_n", axis=1)

    mdf = df.copy().drop("n_infected", axis=1)
    mdf = mdf.melt("time", var_name="metric", value_name = "count")

    sns.set_palette("Dark2")
    sns.set_context("talk")
    sns.set_style("darkgrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharex=True)

    g0 = sns.lineplot(df, ax=axes[0], x="time",y="n_infected")
    axes[0].set_title("Total infected")
    axes[0].set_xlim(left=0, right=max_time)
    axes[0].set_ylim(bottom=0)


    g1 = sns.lineplot(mdf, ax=axes[1], x="time", y="count", hue="metric")
    axes[1].set_title("Infected per age demographic")
    axes[1].set_xlim(left=0, right=max_time)
    axes[1].set_ylim(bottom=0)
    
    if (args.output):
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()