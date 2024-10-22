import argparse
import gzip
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import yaml
from dask.distributed import Client
from rich import print
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.tree import Tree

from . import rucio_utils
from .preprocess import preprocess


def print_dataset_query(
    query: str,
    dataset_list: Dict[str, Dict[str, list[str]]],
    console: Console,
    selected: list[str] = [],
) -> None:
    """
    Pretty-print the results of a rucio query in a table.

    Parameters
    ----------
        query: str
            The query given to rucio
        dataset_list: dict[str, dict[str,list[str]]]
            The second output of a call to query_dataset with tree=True
        console: Console
            A Console object to print to
        selected: list[str], default []
            A list of selected datasets
    """
    table = Table(title=f"Query: [bold red]{query}")
    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Tag", style="magenta", no_wrap=True)
    table.add_column("Selected", justify="center")
    table.row_styles = ["dim", "none"]
    j = 1
    for name, conds in dataset_list.items():
        ic = 0
        ncond = len(conds)
        for c, tiers in conds.items():
            dataset = f"/{name}/{c}/{tiers[0]}"
            sel = dataset in selected
            if ic == 0:
                table.add_row(
                    name,
                    f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                    "[green bold]Y" if sel else "[red]N",
                    end_section=ic == ncond - 1,
                )
            else:
                table.add_row(
                    "",
                    f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                    "[green bold]Y" if sel else "[red]N",
                    end_section=ic == ncond - 1,
                )
            ic += 1
            j += 1

    console.print(table)


def get_indices_query(input_str: str, maxN: int) -> List[int]:
    tokens = input_str.strip().split(" ")
    final_tokens = []
    for t in tokens:
        if t.isdigit():
            if int(t) > maxN:
                print(
                    f"[red bold]Requested index {t} larger than available elements {maxN}"
                )
                return False
            final_tokens.append(int(t) - 1)  # index 0
        elif "-" in t:
            rng = t.split("-")
            try:
                for i in range(
                    int(rng[0]), int(rng[1]) + 1
                ):  # including the last index
                    if i > maxN:
                        print(
                            f"[red bold]Requested index {t} larger than available elements {maxN}"
                        )
                        return False
                    final_tokens.append(i - 1)
            except Exception:
                print(
                    "[red]Error! Bad formatting for selection string. Use e.g. 1 4 5-9"
                )
                return False
        elif t == "all":
            final_tokens = list(range(0, maxN))
        else:
            print("[red]Error! Bad formatting for selection string. Use e.g. 1 4 5-9")
            return False
    return final_tokens


class DataDiscoveryCLI:
    """
    Simplifies dataset query, replicas, filters, and uproot preprocessing with Dask.
    It can be accessed in a Python script or interpreter via this class, or from the
    command line (as in `python -m coffea.dataset_tools.dataset_query --help`).
    """

    def __init__(self):
        self.console = Console()
        self.rucio_client = None
        self.selected_datasets = []
        self.selected_datasets_metadata = []
        self.last_query = ""
        self.last_query_tree = None
        self.last_query_list = None
        self.sites_allowlist = None
        self.sites_blocklist = None
        self.sites_regex = None
        self.last_replicas_results = None
        self.final_output = None
        self.preprocessed_total = None
        self.preprocessed_available = None

        self.replica_results = defaultdict(list)
        self.replica_results_metadata = {}
        self.replica_results_bysite = {}

        self.commands = [
            "help",
            "login",
            "query",
            "query-results",
            "select",
            "list-selected",
            "replicas",
            "list-replicas",
            "save",
            "preprocess",
            "allow-sites",
            "block-sites",
            "regex-sites",
            "sites-filters",
            "quit",
        ]

    def start_cli(self):
        while True:
            command = Prompt.ask(">", choices=self.commands)
            if command == "help":
                print(
                    r"""[bold yellow]Welcome to the datasets discovery coffea CLI![/bold yellow]
Use this CLI tool to query the CMS datasets and to select interactively the grid sites to use for reading the files in your analysis.
Some basic commands:
  - [bold cyan]query[/]: Look for datasets with * wildcards (like in DAS)
  - [bold cyan]select[/]: Select datasets to process further from query results
  - [bold cyan]replicas[/]: Query rucio to look for files replica and then select the preferred sites
  - [bold cyan]query-results[/]: List the results of the last dataset query
  - [bold cyan]list-selected[/]: Print a list of the selected datasets
  - [bold cyan]list-replicas[/]: Print the selected files replicas for the selected dataset
  - [bold cyan]sites-filters[/]: Show the active sites filters and ask to clear them
  - [bold cyan]allow-sites[/]: Restrict the grid sites available for replicas query only to the requested list
  - [bold cyan]block-sites[/]: Exclude grid sites from the available sites for replicas query
  - [bold cyan]regex-sites[/]: Select sites with a regex for replica queries: e.g.  "T[123]_(FR|IT|BE|CH|DE)_\w+"
  - [bold cyan]save[/]: Save the replicas query results to file (json or yaml) for further processing
  - [bold cyan]preprocess[/]: Preprocess the replicas with dask and save the fileset for further processing with uproot/coffea
  - [bold cyan]help[/]: Print this help message
            """
                )
            elif command == "login":
                self.do_login()
            elif command == "quit":
                print("Bye!")
                break
            elif command == "query":
                self.do_query()
            elif command == "query-results":
                self.do_query_results()
            elif command == "select":
                self.do_select()
            elif command == "list-selected":
                self.do_list_selected()
            elif command == "replicas":
                self.do_replicas()
            elif command == "list-replicas":
                self.do_list_replicas()
            elif command == "save":
                self.do_save()
            elif command == "preprocess":
                self.do_preprocess()
            elif command == "allow-sites":
                self.do_allowlist_sites()
            elif command == "block-sites":
                self.do_blocklist_sites()
            elif command == "regex-sites":
                self.do_regex_sites()
            elif command == "sites-filters":
                self.do_sites_filters()
            else:
                break

    def do_login(self, proxy=None):
        """Login to the rucio client. Optionally a specific proxy file can be passed to the command.
        If the proxy file is not specified, `voms-proxy-info` is used"""
        if proxy:
            self.rucio_client = rucio_utils.get_rucio_client(proxy)
        else:
            self.rucio_client = rucio_utils.get_rucio_client()
        print(self.rucio_client)

    def do_whoami(self):
        # Your code here
        if not self.rucio_client:
            print("First [bold]login (L)[/] to the rucio server")
            return
        print(self.rucio_client.whoami())

    def do_query(self, query=None):
        """
        Look for datasets with * wildcards (like in DAS)

        Parameters
        ----------
            query: str | None, default None
                The query to pass to rucio. If None, will prompt the user for an input.
        """
        if query is None:
            query = Prompt.ask(
                "[yellow bold]Query for[/]",
            )
        with self.console.status(f"Querying rucio for: [bold red]{query}[/]"):
            outlist, outtree = rucio_utils.query_dataset(
                query,
                client=self.rucio_client,
                tree=True,
                scope="cms",  # TODO configure scope
            )
            # Now let's print the results as a tree
            print_dataset_query(query, outtree, self.console, self.selected_datasets)
            self.last_query = query
            self.last_query_list = outlist
            self.last_query_tree = outtree
        print("Use the command [bold red]select[/] to selected the datasets")

    def do_query_results(self):
        """List the results of the last dataset query"""
        if self.last_query_list:
            print_dataset_query(
                self.last_query,
                self.last_query_tree,
                self.console,
                self.selected_datasets,
            )
        else:
            print("First [bold red]query (Q)[/] for a dataset")

    def do_select(self, selection=None, metadata=None):
        """
        Selected the datasets from the list of query results. Input a list of indices
        also with range 4-6 or "all".

        Parameters
        ----------
            selection: list[str] | None, default None
                A list of indices corresponding to selected datasets. Should be a
                string, with indices separated by spaces. Can include ranges (like "4-6")
                or "all".
            metadata: dict[Hashable,Any], default None
                Metadata to store in associated with selected datasets.
        """
        if not self.last_query_list:
            print("First [bold red]query (Q)[/] for a dataset")
            return

        if selection is None:
            selection = Prompt.ask(
                "[yellow bold]Select datasets indices[/] (e.g 1 4 6-10)", default="all"
            )
        final_tokens = get_indices_query(selection, len(self.last_query_list))
        if not final_tokens:
            return

        Nresults = len(self.last_query_list)
        print("[cyan]Selected datasets:")

        for s in final_tokens:
            if s < Nresults:
                self.selected_datasets.append(self.last_query_list[s])
                if metadata:
                    self.selected_datasets_metadata.append(metadata)
                else:
                    self.selected_datasets_metadata.append({})
                print(f"- ({s+1}) {self.last_query_list[s]}")
            else:
                print(
                    f"[red]The requested dataset is not in the list. Please insert a position <={Nresults}"
                )

    def do_list_selected(self):
        """Print a list of the selected datasets"""
        print("[cyan]Selected datasets:")
        table = Table(title="Selected datasets")
        table.add_column("Index", justify="left", style="cyan", no_wrap=True)
        table.add_column("Dataset", style="magenta", no_wrap=True)
        table.add_column("Replicas selected", justify="center")
        table.add_column("N. of files", justify="center")
        for i, ds in enumerate(self.selected_datasets):
            table.add_row(
                str(i + 1),
                ds,
                "[green bold]Y" if ds in self.replica_results else "[red]N",
                (
                    str(len(self.replica_results[ds]))
                    if ds in self.replica_results
                    else "-"
                ),
            )
        self.console.print(table)

    def do_replicas(self, mode=None, selection=None):
        """
        Query Rucio for replicas.

        Parameters
        ----------
            mode: str, default None
                One of the following
                    - None:  ask the user about the mode
                    - round-robin (take files randomly from available sites),
                    - choose: ask the user to choose from a list of sites
                    - first: take the first site from the rucio query
            selection: str, default None
                list of indices or 'all' to select all the selected datasets for
                replicas query
        """
        if selection is None:
            selection = Prompt.ask(
                "[yellow bold]Select datasets indices[/] (e.g 1 4 6-10)", default="all"
            )
        indices = get_indices_query(selection, len(self.selected_datasets))
        if not indices:
            return
        datasets = [
            (self.selected_datasets[ind], self.selected_datasets_metadata[ind])
            for ind in indices
        ]

        for dataset, dataset_metadata in datasets:
            with self.console.status(
                f"Querying rucio for replicas: [bold red]{dataset}[/]"
            ):
                try:
                    (
                        outfiles,
                        outsites,
                        sites_counts,
                    ) = rucio_utils.get_dataset_files_replicas(
                        dataset,
                        allowlist_sites=self.sites_allowlist,
                        blocklist_sites=self.sites_blocklist,
                        regex_sites=self.sites_regex,
                        mode="full",
                        client=self.rucio_client,
                    )
                except Exception as e:
                    print(f"\n[red bold] Exception: {e}[/]")
                    return
                self.last_replicas_results = (outfiles, outsites, sites_counts)

            print(f"[cyan]Sites availability for dataset: [red]{dataset}")
            table = Table(title="Available replicas")
            table.add_column("Index", justify="center")
            table.add_column("Site", justify="left", style="cyan", no_wrap=True)
            table.add_column("Files", style="magenta", no_wrap=True)
            table.add_column("Availability", justify="center")
            table.row_styles = ["dim", "none"]
            Nfiles = len(outfiles)

            sorted_sites = dict(
                sorted(sites_counts.items(), key=lambda x: x[1], reverse=True)
            )
            for i, (site, stat) in enumerate(sorted_sites.items()):
                table.add_row(
                    str(i), site, f"{stat} / {Nfiles}", f"{stat*100/Nfiles:.1f}%"
                )

            self.console.print(table)
            if mode is None:
                mode = Prompt.ask(
                    "Select sites",
                    choices=["round-robin", "choose", "first", "quit"],
                    default="round-robin",
                )

            files_by_site = defaultdict(list)

            if mode == "choose":
                ind = list(
                    map(
                        int,
                        Prompt.ask(
                            "Enter list of sites index to be used", default="0"
                        ).split(" "),
                    )
                )
                sites_to_use = [list(sorted_sites.keys())[i] for i in ind]
                print(f"Filtering replicas with [green]: {' '.join(sites_to_use)}")

                output = []
                for ifile, (files, sites) in enumerate(zip(outfiles, outsites)):
                    random.shuffle(sites_to_use)
                    found = False
                    # loop on shuffled selected sites until one is found
                    for site in sites_to_use:
                        try:
                            iS = sites.index(site)
                            output.append(files[iS])
                            files_by_site[sites[iS]].append(files[iS])
                            found = True
                            break  # keep only one replica
                        except ValueError:
                            # if the index is not found just go to the next site
                            pass

                    if not found:
                        print(
                            f"[bold red]No replica found compatible with sites selection for file #{ifile}. The available sites are:"
                        )
                        for f, s in zip(files, sites):
                            print(f"\t- [green]{s} [cyan]{f}")
                        return

                self.replica_results[dataset] = output
                self.replica_results_metadata[dataset] = dataset_metadata

            elif mode == "round-robin":
                output = []
                for ifile, (files, sites) in enumerate(zip(outfiles, outsites)):
                    # selecting randomly from the sites
                    iS = random.randint(0, len(sites) - 1)
                    output.append(files[iS])
                    files_by_site[sites[iS]].append(files[iS])
                self.replica_results[dataset] = output
                self.replica_results_metadata[dataset] = dataset_metadata

            elif mode == "first":
                output = []
                for ifile, (files, sites) in enumerate(zip(outfiles, outsites)):
                    output.append(files[0])
                    files_by_site[sites[0]].append(files[0])
                self.replica_results[dataset] = output
                self.replica_results_metadata[dataset] = dataset_metadata

            elif mode == "quit":
                print("[orange]Doing nothing...")
                return

            self.replica_results_bysite[dataset] = files_by_site

            # Now let's print the results
            tree = Tree(label=f"[bold orange]Replicas for [green]{dataset}")
            for site, files in files_by_site.items():
                T = tree.add(f"[green]{site}")
                for f in files:
                    T.add(f"[cyan]{f}")
            self.console.print(tree)

        # Building an uproot compatible output
        self.final_output = {}
        for fileset, files in self.replica_results.items():
            self.final_output[fileset] = {
                "files": {f: "Events" for f in files},
                "metadata": self.replica_results_metadata[fileset],
            }
        return self.final_output

    @property
    def as_dict(self):
        return self.final_output

    def do_allowlist_sites(self, sites=None):
        """
        Restrict the grid sites available for replicas query only to the requested list

        Parameters
        ----------
            sites: list[str] | None, default None
                The sites to allow the replicas query to look at. If passing in a list,
                elements of the list are sites. If passing in None, the prompt requires
                a single string containing a comma-separated listing.
        """
        if sites is None:
            sites = Prompt.ask(
                "[yellow]Restrict the available sites to (comma-separated list)"
            ).split(",")
        if self.sites_allowlist is None:
            self.sites_allowlist = sites
        else:
            self.sites_allowlist += sites
        print("[green]Allowlisted sites:")
        for s in self.sites_allowlist:
            print(f"- {s}")

    def do_blocklist_sites(self, sites=None):
        """
        Exclude grid sites from the available sites for replicas query

        Parameters
        ----------
            sites: list[str] | None, default None
                The sites to prevent the replicas query from looking at. If passing in a
                list elements of the list are sites. If passing in None, the prompt
                requires a single string containing a comma-separated listing.
        """
        if sites is None:
            sites = Prompt.ask(
                "[yellow]Exclude the sites (comma-separated list)"
            ).split(",")
        if self.sites_blocklist is None:
            self.sites_blocklist = sites
        else:
            self.sites_blocklist += sites
        print("[red]Blocklisted sites:")
        for s in self.sites_blocklist:
            print(f"- {s}")

    def do_regex_sites(self, regex=None):
        r"""
        Select sites with a regex for replica queries: e.g.  "T[123]_(FR|IT|BE|CH|DE)_\w+"

        Parameters
        ----------
            regex: str | None, default None
                Sites to use for replica queries, described with a regex string.
        """
        if regex is None:
            regex = Prompt.ask("[yellow]Regex to restrict the available sites")
        if len(regex):
            self.sites_regex = rf"{regex}"
            print(f"New sites regex: [cyan]{self.sites_regex}")

    def do_sites_filters(self, ask_clear=True):
        """
        Show the active sites filters (allowed, disallowed, and regex) and ask to clear
        them

        Parameters
        ----------
            ask_clear: bool, default True
                If True, ask the user via prompt if allow, disallow, and regex filters
                should be cleared.
        """
        print("[green bold]Allow-listed sites:")
        if self.sites_allowlist:
            for s in self.sites_allowlist:
                print(f"- {s}")

        print("[bold red]Block-listed sites:")
        if self.sites_blocklist:
            for s in self.sites_blocklist:
                print(f"- {s}")

        print(f"[bold cyan]Sites regex: [italics]{self.sites_regex}")

        if ask_clear:
            if Confirm.ask("Clear sites restriction?", default=False):
                self.sites_allowlist = None
                self.sites_blocklist = None
                self.sites_regex = None
                print("[bold green]Sites filters cleared")

    def do_list_replicas(self):
        """Print the selected files replicas for the selected dataset"""
        selection = Prompt.ask(
            "[yellow bold]Select datasets indices[/] (e.g 1 4 6-10)", default="all"
        )
        indices = get_indices_query(selection, len(self.selected_datasets))
        datasets = [self.selected_datasets[ind] for ind in indices]

        for dataset in datasets:
            if dataset not in self.replica_results:
                print(
                    f"[red bold]No replica info for dataset {dataset}. You need to selected the replicas with [cyan] replicas [/cyan] command[/]"
                )
                return
            tree = Tree(label=f"[bold orange]Replicas for [/][green]{dataset}[/]")
            for site, files in self.replica_results_bysite[dataset].items():
                T = tree.add(f"[green]{site}")
                for f in files:
                    T.add(f"[cyan]{f}")

            self.console.print(tree)

    def do_save(self, filename=None):
        """
        Save the replica information in yaml format

        Parameters:
            filename: str | None, default None
                The name of the file to save the information into
        """
        if not filename:
            filename = Prompt.ask(
                "[yellow bold]Output file name (.yaml or .json)", default="output.json"
            )
        format = os.path.splitext(filename)[1]
        if not format:
            raise Exception("[red] Please use a .json or .yaml filename for the output")
        with open(filename, "w") as file:
            if format == ".yaml":
                yaml.dump(self.final_output, file, default_flow_style=False)
            elif format == ".json":
                json.dump(self.final_output, file, indent=2)
        print(f"[green]File {filename} saved!")

    def do_preprocess(
        self,
        output_file=None,
        step_size=None,
        align_to_clusters=None,
        scheduler_url=None,
        recalculate_steps=None,
        files_per_batch=None,
        file_exceptions=(OSError,),
        save_form=None,
        uproot_options={},
        step_size_safety_factor=0.5,
        allow_empty_datasets=False,
    ):
        """
        Perform preprocessing for concrete fileset extraction into a file, compressed
        with gzip.

        Parameters
        ----------
            output_file: str | None, default None
                The name of the file to write the preprocessed file into
            step_size: int | None, default None
                The chunk size for file splitting
            align_to_clusters: bool | None, default None
                Whether or not round to the cluster size in a root file. See
                align_clusters parameter in coffea.dataset_tools.preprocess.
            scheduler_url: str | None, default None
                Dask scheduler URL where the preprocessing should take place
        """
        if not output_file:
            output_file = Prompt.ask(
                "[yellow bold]Output name", default="output_preprocessing"
            )
        if step_size is None:
            step_size = IntPrompt.ask("[yellow bold]Step size", default=None)
        if align_to_clusters is None:
            align_to_clusters = Confirm.ask(
                "[yellow bold]Align to clusters", default=True
            )
        if recalculate_steps is None:
            recalculate_steps = Confirm.ask(
                "[yellow bold]Recalculate steps", default=False
            )
        if files_per_batch is None:
            files_per_batch = IntPrompt.ask("[yellow bold]Files per batch", default=1)
        if save_form is None:
            save_form = Confirm.ask("[yellow bold]Save form", default=False)
        if step_size_safety_factor is None:
            step_size_safety_factor = FloatPrompt.ask(
                "[yellow bold]Step size safety factor", default=0.5
            )

        # init a local Dask cluster
        with self.console.status(
            "[red] Preprocessing files to extract available chunks with dask[/]"
        ):
            with Client(scheduler_url) as ddcsched:
                self.preprocessed_available, self.preprocessed_total = preprocess(
                    self.final_output,
                    step_size=step_size,
                    align_clusters=align_to_clusters,
                    skip_bad_files=True,
                    recalculate_steps=recalculate_steps,
                    files_per_batch=files_per_batch,
                    file_exceptions=file_exceptions,
                    save_form=save_form,
                    scheduler=ddcsched,
                    uproot_options=uproot_options,
                    step_size_safety_factor=step_size_safety_factor,
                    allow_empty_datasets=allow_empty_datasets,
                )

        with gzip.open(f"{output_file}_available.json.gz", "wt") as file:
            print(f"Saved available fileset chunks to {output_file}_available.json.gz")
            json.dump(self.preprocessed_available, file, indent=2)
        with gzip.open(f"{output_file}_all.json.gz", "wt") as file:
            print(f"Saved all fileset chunks to {output_file}_all.json.gz")
            json.dump(self.preprocessed_total, file, indent=2)
        return self.preprocessed_available, self.preprocessed_total

    def load_dataset_definition(
        self,
        dataset_definition,
        query_results_strategy="all",
        replicas_strategy="round-robin",
    ):
        """
        Initialize the DataDiscoverCLI by querying a set of datasets defined in `dataset_definitions`
        and selected results and replicas following the options.

        Parameters
        ----------
            dataset_definition: Dict[str,Dict[Hashable,Any]]
                Keys are dataset queries (ie: something that can be passed to do_query())
            query_results_strategy: str, default "all"
                How to decide which datasets to select. If "manual", user will be prompted
                for selection
            replicas_strategy: str, default "round-robin"
                Options are:
                    - "round-robin": select randomly from the available sites for each file
                    - "choose": filter the sites with a list of indices for all the files
                    - "first": take the first result returned by rucio
                    - "manual": to be prompt for manual decision dataset by dataset

        Returns
        -------
            out_replicas: FilesetSpecOptional
                An uproot-readable fileset. At this point, the fileset is not fully
                preprocessed, but this can be done with do_preprocess().
        """
        for dataset_query, dataset_meta in dataset_definition.items():
            print(f"\nProcessing query: {dataset_query}")
            # Adding queries
            self.do_query(dataset_query)
            # Now selecting the results depending on the interactive mode or not.
            # Metadata are passed to the selection function to associated them with the selected dataset.
            if query_results_strategy not in ["all", "manual"]:
                raise ValueError(
                    "Invalid query-results-strategy option: please choose between: manual|all"
                )
            elif query_results_strategy == "manual":
                self.do_select(selection=None, metadata=dataset_meta)
            else:
                self.do_select(selection="all", metadata=dataset_meta)

        # Now list all
        self.do_list_selected()

        # selecting replicas
        self.do_sites_filters(ask_clear=False)
        print("Getting replicas")
        if replicas_strategy == "manual":
            out_replicas = self.do_replicas(mode=None, selection="all")
        else:
            if replicas_strategy not in ["round-robin", "choose", "first"]:
                raise ValueError(
                    "Invalid replicas-strategy: please choose between manual|round-robin|choose|first"
                )
            out_replicas = self.do_replicas(mode=replicas_strategy, selection="all")
        # Now list all
        self.do_list_selected()
        return out_replicas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli", help="Start the dataset discovery CLI", action="store_true"
    )
    parser.add_argument(
        "-d",
        "--dataset-definition",
        help="Dataset definition file",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output name for dataset discovery output (no fileset preprocessing)",
        type=str,
        required=False,
        default="output_dataset.json",
    )
    parser.add_argument(
        "-fo",
        "--fileset-output",
        help="Output name for fileset",
        type=str,
        required=False,
        default="output_fileset",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        help="Preprocess with dask",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--step-size", help="Step size for preprocessing", type=int, default=500000
    )
    parser.add_argument(
        "--scheduler-url", help="Dask scheduler url", type=str, default=None
    )
    parser.add_argument(
        "-as",
        "--allow-sites",
        help="List of sites to be allowlisted",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-bs",
        "--block-sites",
        help="List of sites to be blocklisted",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-rs",
        "--regex-sites",
        help="Regex string to be used to filter the sites",
        type=str,
    )
    parser.add_argument(
        "--query-results-strategy",
        help="Mode for query results selection: [all|manual]",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--replicas-strategy",
        help="Mode for selecting replicas for datasets: [manual|round-robin|first|choose]",
        default="round-robin",
        required=False,
    )
    args = parser.parse_args()

    cli = DataDiscoveryCLI()

    if args.allow_sites:
        cli.sites_allowlist = args.allow_sites
    if args.block_sites:
        cli.sites_blocklist = args.block_sites
    if args.regex_sites:
        cli.sites_regex = args.regex_sites

    if args.dataset_definition:
        with open(args.dataset_definition) as file:
            dd = json.load(file)
        cli.load_dataset_definition(
            dd,
            query_results_strategy=args.query_results_strategy,
            replicas_strategy=args.replicas_strategy,
        )
        # Save
        if args.output:
            cli.do_save(filename=args.output)
        if args.preprocess:
            cli.do_preprocess(
                output_file=args.fileset_output,
                step_size=args.step_size,
                scheduler_url=args.scheduler_url,
                align_to_clusters=False,
            )

    if args.cli:
        cli.start_cli()
