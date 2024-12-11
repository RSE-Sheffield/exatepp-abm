#! /usr/bin/env python3 
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

def parse_cli():
    parser = argparse.ArgumentParser(description="Plotting script for per individual data")
    parser.add_argument("csv", type=pathlib.Path, help="per individual CSV to plot")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Path to output image location")
    args = parser.parse_args()
    return args

def read_inputs(args):
    df = pd.read_csv(args.csv)
    
    expected_columns = ["ID", "age_group", "occupation_network", "house_no", "infection_count"]
    missing_columns = [c for c in expected_columns if c not in df.columns]
    if len(missing_columns) > 0:            
        raise Exception(f"expected columns missing from csv: {missing_columns}")

    # Add string version of age group
    AGE_GROUP_MAP = {
        0: "0-9",
        1: "10-19",
        2: "20-29",
        3: "30-39",
        4: "40-49",
        5: "50-59",
        6: "60-69",
        7: "70-79",
        8: "80+",
    }
    df["age_group_str"] = pd.Categorical(df["age_group"].map(AGE_GROUP_MAP), list(AGE_GROUP_MAP.values()))

    # Add string version of occupation network group
    OCCUPATION_NETWORK_MAP = {
        0: "0-9",
        1: "10-19",
        2: "20-69",
        3: "70-79",
        4: "80+"
    }
    df["occupation_network_str"] = pd.Categorical(df["occupation_network"].map(OCCUPATION_NETWORK_MAP), list(OCCUPATION_NETWORK_MAP.values()))
    return df

def save_or_show(args):
    # Either save the current plot to disk, or show it.
    if (args.output):
        args.output.parent.mkdir(exist_ok=True)
        plt.savefig(args.output)
    else:
        plt.show()


def plot_age_histogram(args, df, ax):
    ax.set_title("Population per age demographic")
    sns.histplot(ax=ax, data=df, x="age_group_str", discrete=True)
    ax.tick_params(axis='x', rotation=90)

def plot_age_histogram_per_workplace(args, df, ax):
    ax.set_title("Age Demographic per workplace")
    sns.histplot(ax=ax, data=df, x="age_group_str", hue="occupation_network_str", discrete=True, element="step", palette="Dark2")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.tick_params(axis='x', rotation=90)



def plot_household_size_household_age(args, df, ax):
    ax.set_title("Household size against household age")

    # Group by "house_no" and calculate mean "age_group" and count
    df_mean_age_per_house = df.groupby('house_no').agg({'age_group': ['mean']}).reset_index()
    df_mean_age_per_house.columns = ["house_no", "mean_age_group"]
    count_per_house = df.groupby('house_no').size().reset_index(name="house_size")
    combined_df = df_mean_age_per_house
    combined_df["house_size"] = count_per_house["house_size"]
 
    # sns.scatterplot(ax=ax, data=combined_df, x="house_size", y="mean_age_group")
    # sns.violinplot(ax=ax, data=combined_df, x="house_size", y="mean_age_group", inner="quart")
    sns.boxplot(ax=ax, data=combined_df, x="house_size", y="mean_age_group")
    sns.stripplot(ax=ax, data=combined_df, x="house_size", y="mean_age_group")
    # sns.swarmplot(ax=ax, data=combined_df, x="house_size", y="mean_age_group")




def plot_per_house_infection_count(args, df, ax):
    ax.set_title("Total infection count per house")

    # Group by "house_no" and compute the sum of infections per hosue
    df_grouped = df.groupby('house_no').agg({'infection_count': ['sum']}).reset_index()
    df_grouped.columns = ["house_no", "sum_infection_count"]
    count_per_house = df.groupby('house_no').size().reset_index(name="house_size")
    df_grouped["style"] = 0
 
    markers = ["x"]
    sizes = [0.5]
    sns.scatterplot(ax=ax, data=df_grouped, x="house_no", y="sum_infection_count", style="style", markers=markers, sizes=sizes, legend=None)


def plots(args, df):
    sns.set_palette("muted")
    sns.set_context("talk")
    sns.set_style("darkgrid")

    # Get the current palette
    palette = sns.color_palette(palette=None)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False, sharey=False, layout="constrained")

    # plot different things on each axis
    plot_age_histogram(args, df, axes[0, 0])
    plot_age_histogram_per_workplace(args, df, axes[0, 1])
    plot_household_size_household_age(args, df, axes[1, 0])
    plot_per_house_infection_count(args, df, axes[1, 1])

    # Save to disk or show on screen
    save_or_show(args)

def main():
    # CLI parsing
    args = parse_cli()
    # Read the single input file 
    df = read_inputs(args)
    # Do some plots
    plots(args, df)

if __name__ == "__main__":
    main()