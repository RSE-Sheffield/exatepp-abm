#! /usr/bin/env python3 
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib


def parse_cli():
    parser = argparse.ArgumentParser(description="Plotting script for time series data")
    parser.add_argument("csv", type=pathlib.Path, help="Time series CSV to plot")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Path to output image location")
    parser.add_argument("--per-demographic", action="store_true", help="per-demographic cumulative infected stacked area plot")
    args = parser.parse_args()
    return args

def read_inputs(args):
    return pd.read_csv(args.csv)

def save_or_show(args):
    # Either save the current plot to disk, or show it.
    if (args.output):
        plt.savefig(args.output)
    else:
        plt.show()

def demographic_area_plot(args, df):
    # Get the maximum time value for x axis limiting
    max_time = df["time"].max()

    # Set some theming values
    sns.set_palette("muted", 9)
    sns.set_context("talk")
    sns.set_style("darkgrid")

    # Do an area plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=True)

    stacked_df = df[[
        "time",
        "total_infected_0_9",
        "total_infected_10_19",
        "total_infected_20_29",
        "total_infected_30_39",
        "total_infected_40_49",
        "total_infected_50_59",
        "total_infected_60_69",
        "total_infected_70_79",
        "total_infected_80",
    ]].copy()
    
    stacked_df.rename(columns={
        "total_infected_0_9": "0-9",
        "total_infected_10_19": "10-19",
        "total_infected_20_29": "20-29",
        "total_infected_30_39": "30-39",
        "total_infected_40_49": "40-49",
        "total_infected_50_59": "50-59",
        "total_infected_60_69": "60-69",
        "total_infected_70_79": "70-79",
        "total_infected_80": "80+",
    }, inplace=True)
    stacked_df.plot.area(x="time", ax=ax, legend='reverse')

    ax.set_title("Cumulative Infected per age demographic")
    ax.set_xlim(left=0, right=max_time)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Cumulative Infections")

    save_or_show(args)

def lineplots(args, df):


    max_time = df["time"].max()

    sns.set_palette("muted")
    sns.set_context("talk")
    sns.set_style("darkgrid")

    # Get the current palette
    palette = sns.color_palette(palette=None)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True)

    # g0 = sns.lineplot(df, ax=axes[0], x="time",y="n_infected")#, hue=palette[0])
    # axes[0].set_title("Current SEIR")
    # axes[0].set_xlim(left=0, right=max_time)
    # axes[0].set_ylim(bottom=0)

# Cumulative Infected
    cumulative_infected = df[[
        "time",
        "n_susceptible",
        "n_exposed",
        "n_infected",
        "n_recovered",
       ]].copy()
    
    cumulative_infected.rename(columns={
        "n_susceptible": "Susceptible",
        "n_exposed": "Exposed",
        "n_infected": "Infected",
        "n_recovered": "Recovered",
    }, inplace=True)

    # Melt the dataframe
    cumulative_infected = cumulative_infected.melt("time", var_name="State", value_name = "count")

    g0 = sns.lineplot(cumulative_infected, ax=axes[0], x="time", y="count", hue="State")
    axes[0].set_title("Infection Status")
    axes[0].set_xlim(left=0, right=max_time)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel("Day")

    # ----------------------

    # Cumulative Infected
    cumulative_infected = df[[
        "time",
        "total_infected_0_9",
        "total_infected_0_9",
        "total_infected_10_19",
        "total_infected_20_29",
        "total_infected_30_39",
        "total_infected_40_49",
        "total_infected_50_59",
        "total_infected_60_69",
        "total_infected_70_79",
        "total_infected_80",
        "total_infected",
    ]].copy()
    
    cumulative_infected.rename(columns={
        "total_infected_0_9": "0-9",
        "total_infected_10_19": "10-19",
        "total_infected_20_29": "20-29",
        "total_infected_30_39": "30-39",
        "total_infected_40_49": "40-49",
        "total_infected_50_59": "50-59",
        "total_infected_60_69": "60-69",
        "total_infected_70_79": "70-79",
        "total_infected_80": "80+",
        "total_infected": "total",
    }, inplace=True)

    # Melt the dataframe
    cumulative_infected = cumulative_infected.melt("time", var_name="Demographic", value_name = "count")

    g1 = sns.lineplot(cumulative_infected, ax=axes[1], x="time", y="count", hue="Demographic")
    axes[1].set_title("Cumulative Infected per age demographic")
    axes[1].set_xlim(left=0, right=max_time)
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Day")


    save_or_show(args)


def main():
    # CLI parsing
    args = parse_cli()
    # Read the single input file 
    df = read_inputs(args)

    if(args.per_demographic):
        # do an area plot per demographic
        demographic_area_plot(args, df)
    else:
        # Do some line plots
        lineplots(args, df)


if __name__ == "__main__":
    main()