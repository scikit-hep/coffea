# import getpass
import json
import os
import re
import subprocess
import time
from collections import defaultdict

from rucio.client import Client

# Rucio needs the default configuration --> taken from CMS cvmfs defaults
if "RUCIO_HOME" not in os.environ:
    os.environ["RUCIO_HOME"] = "/cvmfs/cms.cern.ch/rucio/current"


def get_proxy_path() -> str:
    """
    Checks if the VOMS proxy exists and if it is valid
    for at least 1 hour.
    If it exists, returns the path of it"""
    try:
        subprocess.run("voms-proxy-info -exists -hours 1", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception(
            "VOMS proxy expirend or non-existing: please run `voms-proxy-init -voms cms -rfc --valid 168:0`"
        )

    # Now get the path of the certificate
    proxy = subprocess.check_output(
        "voms-proxy-info -path", shell=True, text=True
    ).strip()
    return proxy


def get_rucio_client(proxy=None) -> Client:
    """
    Open a client to the CMS rucio server using x509 proxy.

    Parameters
    ----------
        proxy : str, optional
            Use the provided proxy file if given, if not use `voms-proxy-info` to get the current active one.

    Returns
    -------
        nativeClient: rucio.Client
            Rucio client
    """
    try:
        if not proxy:
            proxy = get_proxy_path()
        nativeClient = Client()
        return nativeClient

    except Exception as e:
        print("Wrong Rucio configuration, impossible to create client")
        raise e


def get_xrootd_sites_map():
    """
    The mapping between RSE (sites) and the xrootd prefix rules is read
    from `/cvmfs/cms/cern.ch/SITECONF/*site*/storage.json`.

    This function returns the list of xrootd prefix rules for each site.
    """
    sites_xrootd_access = defaultdict(dict)
    # Check if the cache file has been modified in the last 10 minutes
    cache_valid = False
    if os.path.exists(".sites_map.json"):
        file_time = os.path.getmtime(".sites_map.json")
        current_time = time.time()
        ten_minutes_ago = current_time - 600
        if file_time > ten_minutes_ago:
            cache_valid = True

    if not os.path.exists(".sites_map.json") or not cache_valid:
        print("Loading SITECONF info")
        sites = [
            (s, "/cvmfs/cms.cern.ch/SITECONF/" + s + "/storage.json")
            for s in os.listdir("/cvmfs/cms.cern.ch/SITECONF/")
            if s.startswith("T")
        ]
        for site_name, conf in sites:
            if not os.path.exists(conf):
                continue
            try:
                data = json.load(open(conf))
            except Exception:
                continue
            for site in data:
                if site["type"] != "DISK":
                    continue
                if site["rse"] is None:
                    continue
                for proc in site["protocols"]:
                    if proc["protocol"] == "XRootD":
                        if proc["access"] not in ["global-ro", "global-rw"]:
                            continue
                        if "prefix" not in proc:
                            if "rules" in proc:
                                for rule in proc["rules"]:
                                    sites_xrootd_access[site["rse"]][rule["lfn"]] = (
                                        rule["pfn"]
                                    )
                        else:
                            sites_xrootd_access[site["rse"]] = proc["prefix"]

        json.dump(sites_xrootd_access, open(".sites_map.json", "w"))

    return json.load(open(".sites_map.json"))


def _get_pfn_for_site(path, rules):
    """
    Utility function that converts the file path to a valid pfn matching
    the file path with the site rules (regexes).
    """
    if isinstance(rules, dict):
        for rule, pfn in rules.items():
            if m := re.match(rule, path):
                grs = m.groups()
                for i in range(len(grs)):
                    pfn = pfn.replace(f"${i+1}", grs[i])
                return pfn
    else:
        # not adding any slash as the path usually starts with it
        return rules + "/" + path.removeprefix("/")


def get_dataset_files_replicas(
    dataset,
    allowlist_sites=None,
    blocklist_sites=None,
    regex_sites=None,
    mode="full",
    partial_allowed=False,
    client=None,
    scope="cms",
):
    """
    This function queries the Rucio server to get information about the location
    of all the replicas of the files in a CMS dataset.

    The sites can be filtered in 3 different ways:
    - `allowlist_sites`: list of sites to select from. If the file is not found there, raise an Exception.
    - `blocklist_sites`: list of sites to avoid. If the file has no left site, raise an Exception
    - `regex_sites`: regex expression to restrict the list of sites.

    The fileset returned by the function is controlled by the `mode` parameter:
    - "full": returns the full set of replicas and sites (passing the filtering parameters)
    - "first": returns the first replica found for each file
    - "best": to be implemented (ServiceX..)
    - "roundrobin": try to distribute the replicas over different sites

    Parameters
    ----------

        dataset: str
            The dataset to search for.
        allowlist_sites: list
            List of sites to select from. If the file is not found there, raise an Exception.
        blocklist_sites: list
            List of sites to avoid. If the file has no left site, raise an Exception.
        regex_sites: list
            Regex expression to restrict the list of sites.
        mode:  str, default "full"
            One of "full", "first", "best", or "roundrobin". Behavior of each described above.
        client: rucio Client, optional
            The rucio client to use. If not provided, one will be generated for you.
        partial_allowed: bool, default False
            If False, throws an exception if any file in the dataset cannot be found. If True,
            will find as many files from the dataset as it can.
        scope:  rucio scope, "cms"
            The scope for rucio to search through.

    Returns
    -------
        files: list
           depending on the `mode` option.
           - If `mode=="full"`, returns the complete list of replicas for each file in the dataset
           - If `mode=="first"`, returns only the first replica for each file.

        sites: list
           depending on the `mode` option.
           - If `mode=="full"`, returns the list of sites where the file replica is available for each file in the dataset
           - If `mode=="first"`, returns a list of sites for the first replica of each file.

        sites_counts: dict
           Metadata counting the coverage of the dataset by site

    """
    sites_xrootd_prefix = get_xrootd_sites_map()
    client = client if client else get_rucio_client()
    outsites = []
    outfiles = []
    for filedata in client.list_replicas([{"scope": scope, "name": dataset}]):
        outfile = []
        outsite = []
        rses = filedata["rses"]
        found = False
        if allowlist_sites:
            for site in allowlist_sites:
                if site in rses:
                    # Check actual availability
                    meta = filedata["pfns"][rses[site][0]]
                    if (
                        meta["type"] != "DISK"
                        or meta["volatile"]
                        or filedata["states"][site] != "AVAILABLE"
                        or site not in sites_xrootd_prefix
                    ):
                        continue
                    outfile.append(
                        _get_pfn_for_site(filedata["name"], sites_xrootd_prefix[site])
                    )
                    outsite.append(site)
                    found = True

            if not found and not partial_allowed:
                raise Exception(
                    f"No SITE available in the allowlist for file {filedata['name']}"
                )
        else:
            possible_sites = list(rses.keys())
            if blocklist_sites:
                possible_sites = list(
                    filter(lambda key: key not in blocklist_sites, possible_sites)
                )

            if len(possible_sites) == 0 and not partial_allowed:
                raise Exception(f"No SITE available for file {filedata['name']}")

            # now check for regex
            for site in possible_sites:
                if regex_sites:
                    if re.search(regex_sites, site):
                        # Check actual availability
                        meta = filedata["pfns"][rses[site][0]]
                        if (
                            meta["type"] != "DISK"
                            or meta["volatile"]
                            or filedata["states"][site] != "AVAILABLE"
                            or site not in sites_xrootd_prefix
                        ):
                            continue
                        outfile.append(
                            _get_pfn_for_site(
                                filedata["name"], sites_xrootd_prefix[site]
                            )
                        )
                        outsite.append(site)
                        found = True
                else:
                    # Just take the first one
                    # Check actual availability
                    meta = filedata["pfns"][rses[site][0]]
                    if (
                        meta["type"] != "DISK"
                        or meta["volatile"]
                        or filedata["states"][site] != "AVAILABLE"
                        or site not in sites_xrootd_prefix
                    ):
                        continue
                    outfile.append(
                        _get_pfn_for_site(filedata["name"], sites_xrootd_prefix[site])
                    )
                    outsite.append(site)
                    found = True

        if not found and not partial_allowed:
            raise Exception(f"No SITE available for file {filedata['name']}")
        else:
            if mode == "full":
                outfiles.append(outfile)
                outsites.append(outsite)
            elif mode == "first":
                outfiles.append(outfile[0])
                outsites.append(outsite[0])
            else:
                raise NotImplementedError(f"Mode {mode} not yet implemented!")

    # Computing replicas by site:
    sites_counts = defaultdict(int)
    if mode == "full":
        for sites_by_file in outsites:
            for site in sites_by_file:
                sites_counts[site] += 1
    elif mode == "first":
        for site_by_file in outsites:
            sites_counts[site] += 1

    return outfiles, outsites, sites_counts


def query_dataset(
    query: str, client=None, tree: bool = False, datatype="container", scope="cms"
):
    """
    This function uses the rucio client to query for containers or datasets.

    Parameters
    ---------
        query: str
            Query to filter datasets / containers with the rucio list_dids functions
        client: rucio Client
            The rucio client to use. If not provided, one will be generated for you
        tree: bool, default False
            If True, return the results splitting the dataset name in parts
        datatype: str, default "container"
            Options are "container", "dataset".  rucio terminology. "Container"==CMS dataset. "Dataset" == CMS block.
        scope: str, default "cms"
            Rucio instance

    Returns
    -------
       List of containers/datasets. If tree==True, returns the list of dataset and also a dictionary decomposing
       the datasets names in the 1st command part and a list of available 2nd parts.

    """
    client = client if client else get_rucio_client()
    out = list(
        client.list_dids(
            scope=scope, filters={"name": query, "type": datatype}, long=False
        )
    )
    if tree:
        outdict = {}
        for dataset in out:
            split = dataset[1:].split("/")
            if split[0] not in outdict:
                outdict[split[0]] = defaultdict(list)
            outdict[split[0]][split[1]].append(split[2])
        return out, outdict
    else:
        return out
