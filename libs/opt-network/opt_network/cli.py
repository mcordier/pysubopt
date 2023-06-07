import click

from .dataset import import_dataset, build_contagion_model
from .optimize import optimize_covdiv_greedy, optimize_expected_greedy, random_select

@click.group()
def cli():
    return None

@cli.command(name="expected-greedy")
@click.option(
    "--path",
    "-p",
    help="Path to the graph file",
)
@click.option(
    "--budget",
    "-b",
    default=10,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--beta",
    default=0.1,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--gamma",
    default=0.05,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--n_time",
    "-t",
    default=2,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--n-iter",
    "-n",
    default=100,
    help="Use of lazy algorithm or not",
)
@click.option(
    "--n-iter-opt",
    "-m",
    default=1,
    help="Use of lazy algorithm or not",
)
@click.option(
    "--scaling-factor",
    "-r",
    default=10,
    help="Number of maximum sentences (budget)",
)
@click.option(
    "--is-lazy",
    "-l",
    default=True,
    help="Use of lazy algorithm or not",
)
def expected_greedy(
    path: str,
    budget: float,
    beta: float,
    gamma: float,
    n_time: int,
    n_iter: int,
    n_iter_opt: int,
    scaling_factor: float,
    is_lazy: bool
):
    dataset = import_dataset(path)
    parameters = {'beta' : 0.1, 'gamma': 0.05, 'T': n_time, 'N': n_iter}
    model = build_contagion_model(dataset, parameters)
    output = optimize_expected_greedy(
        dataset=dataset,
        budget=budget,
        model=model,
        T=n_time,
        N=n_iter_opt,
        r=scaling_factor,
        is_lazy=is_lazy
    )
    print(output)

@cli.command(name="covdiv-greedy")
@click.option(
    "--path",
    "-p",
    help="Path to the graph file",
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
    output = optimize_covdiv_greedy(
        dataset,
        budget,
        r=scaling_factor,
        is_lazy=is_lazy
    )
    print(output)

@cli.command(name="random")
@click.option(
    "--path",
    "-p",
    help="Path to the graph file",
)
@click.option(
    "--budget",
    "-b",
    default=10,
    help="Number of maximum sentences (budget)",
)
def random(path: str, budget: float):
    dataset = import_dataset(path)
    output = random_select(
        dataset,
        budget,
    )
    print(output)
