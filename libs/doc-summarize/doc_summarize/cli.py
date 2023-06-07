import click

from .dataset import import_dataset
from .optimize import optimize_mmr_double_greedy, optimize_mmr_greedy, optimize_covdiv_greedy

@click.group()
def cli():
    return None

@cli.command(name="mmr-double-greedy")
@click.option(
    "--path",
    "-p",
    help="Path to the text file",
)
@click.option(
    "--budget",
    "-b",
    default=10,
    help="Number of maximum sentences (budget)",
)
def mmr_double_greedy(path: str, budget: float):
    dataset = import_dataset(path)
    output_summary = optimize_mmr_double_greedy(dataset, budget)
    print(output_summary)

@cli.command(name="mmr-greedy")
@click.option(
    "--path",
    "-p",
    help="Path to the text file",
)
@click.option(
    "--budget",
    "-b",
    default=10,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--scaling_factor",
    "-r",
    default=10,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--is_lazy",
    "-l",
    default=True,
    help="Use of lazy algorithm or not",
)
def mmr_double_greedy(path: str, budget: float, scaling_factor: float, is_lazy: bool):
    dataset = import_dataset(path)
    output_summary = optimize_mmr_greedy(
        dataset,
        budget,
        r=scaling_factor,
        is_lazy=is_lazy
    )
    print(output_summary)

@cli.command(name="covdiv-greedy")
@click.option(
    "--path",
    "-p",
    help="Path to the text file",
)
@click.option(
    "--budget",
    "-b",
    default=10,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--scaling_factor",
    "-r",
    default=10,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--is_lazy",
    "-l",
    default=True,
    help="Use of lazy algorithm or not",
)
def covdiv_greedy(path: str, budget: float, scaling_factor: float, is_lazy: bool):
    dataset = import_dataset(path)
    output_summary = optimize_covdiv_greedy(
        dataset,
        budget,
        r=scaling_factor,
        is_lazy=is_lazy
    )
    print(output_summary)
