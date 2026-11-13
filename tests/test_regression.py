"""
Exercise sample-scaled matching on captured tsinfer and sc2ts workloads.

The stored outputs used node-scaled transitions and are not expectations
for this model. Compare sparse and full execution under the same fixed
haploid sample size; small-panel tests check the model against dense maths.
"""

import gzip
import json
import pathlib
import shutil

import numpy as np
import pytest
import tskit

import tsls

DATA_DIR = pathlib.Path(__file__).parent / "data" / "regression"
CASES = sorted(path.name for path in DATA_DIR.iterdir())


class Case:
    def __init__(self, name, tmp_dir):
        case_dir = DATA_DIR / name
        trees_path = tmp_dir / f"{name}.trees"
        gz_path = case_dir / "reference.trees.gz"
        with gzip.open(gz_path, "rb") as f_in, open(trees_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        self.ts = tskit.load(trees_path)
        # A fixed test cohort size, including for ancestor-only references.
        self.sample_size = 100
        with open(case_dir / "case.json") as f:
            self.alleles = json.load(f)["alleles"]
        with np.load(case_dir / "case.npz") as data:
            self.data = dict(data)
        self.reference = tsls.TreeSequence(self.ts, alleles=self.alleles)

    @property
    def num_queries(self):
        return len(self.data["haplotypes"])

    def query(self, j):
        d = self.data
        return d["haplotypes"][j], int(d["start"][j]), int(d["end"][j])

    def hmm(self, *, sparse=True):
        return tsls.Hmm(
            self.reference,
            self.data["recombination"],
            self.data["mismatch"],
            sparse=sparse,
            sample_size=self.sample_size,
        )

    def num_alleles(self):
        """
        The number of alleles at each site, as tsls counts them: the length
        of the allele list, or the number of distinct states, at least 2.
        """
        if self.alleles is not None:
            return np.full(self.ts.num_sites, len(self.alleles))
        num_alleles = np.zeros(self.ts.num_sites, dtype=int)
        for site in self.ts.sites():
            states = {site.ancestral_state}
            states |= {m.derived_state for m in site.mutations}
            num_alleles[site.id] = max(2, len(states))
        return num_alleles


@pytest.fixture(scope="module", params=CASES)
def case(request, tmp_path_factory):
    return Case(request.param, tmp_path_factory.mktemp("regression"))


def path_log_likelihood(case, h, start, end, path, matched):
    """
    Return the log likelihood of the specified path (in left-to-right
    order) over sites [start, end), leaving out the initial probability,
    which is the same for every path.
    """
    left = path[0]
    rho = case.data["recombination"]
    mu = case.data["mismatch"]
    n = case.sample_size
    num_alleles = case.num_alleles()
    sites = np.arange(start, end)
    switch = np.isin(case.ts.sites_position[sites], left[1:])
    switch[0] = False
    r = rho[sites]
    transition = np.where(switch, np.log(r / n), np.log(1 - r + r / n))
    transition[0] = 0
    hs = h[sites]
    match = (hs == -1) | (hs == matched[sites])
    m = mu[sites]
    emission = np.where(match, np.log(1 - (num_alleles[sites] - 1) * m), np.log(m))
    return np.sum(transition) + np.sum(emission)


def path_haplotypes(case, paths):
    """
    Return the (num_sites, len(paths)) array of the alleles of the nodes on
    each of the specified paths.
    """
    ts = case.ts
    nodes = np.unique(np.concatenate([parent for _, _, parent in paths]))
    alleles = None if case.alleles is None else tuple(case.alleles)
    G = ts.genotype_matrix(samples=nodes, alleles=alleles, isolated_as_missing=False)
    positions = ts.sites_position
    H = np.zeros((ts.num_sites, len(paths)), dtype=np.int8)
    for k, (left, _, parent) in enumerate(paths):
        segment = np.searchsorted(left, positions, side="right") - 1
        segment = np.clip(segment, 0, len(parent) - 1)
        column = np.searchsorted(nodes, parent[segment])
        H[:, k] = G[np.arange(ts.num_sites), column]
    return H


def find_path(case, j, *, sparse=True):
    h, start, end = case.query(j)
    matched = np.zeros(case.ts.num_sites, dtype=np.int8)
    left, right, parent = case.hmm(sparse=sparse).find_path(h, start, end, matched)
    path = left[::-1], right[::-1], parent[::-1]
    return path, matched


class TestViterbi:
    def test_log_likelihood(self, case):
        for j in range(case.num_queries):
            h, start, end = case.query(j)
            scores = []
            for sparse in [True, False]:
                path, matched = find_path(case, j, sparse=sparse)
                scores.append(path_log_likelihood(case, h, start, end, path, matched))
            assert scores[0] == pytest.approx(scores[1], rel=1e-9)

    def test_matched_haplotype_is_path_haplotype(self, case):
        results = [find_path(case, j) for j in range(case.num_queries)]
        H = path_haplotypes(case, [path for path, _ in results])
        for j, (_, matched) in enumerate(results):
            _, start, end = case.query(j)
            np.testing.assert_array_equal(matched[start:end], H[start:end, j])

    def test_path_covers_window(self, case):
        positions = case.ts.sites_position
        for j in range(case.num_queries):
            _, start, end = case.query(j)
            (left, right, _), _ = find_path(case, j)
            assert left[0] <= positions[start]
            assert right[-1] > positions[end - 1]
            np.testing.assert_array_equal(left[1:], right[:-1])


class TestForwardBackward:
    def test_posteriors(self, case):
        # Check a few queries from long captured workloads.
        step = 4 if case.ts.num_sites > 10000 else 1
        for j in range(0, case.num_queries, step):
            h, _, _ = case.query(j)
            posteriors = []
            scores = []
            for sparse in [True, False]:
                hmm = case.hmm(sparse=sparse)
                forward = hmm.forward_matrix(h)
                backward = hmm.backward_matrix(h, forward.normalisation_factor)
                posteriors.append(forward.decode() * backward.decode())
                scores.append(forward.log_likelihood)
            assert scores[0] == pytest.approx(scores[1], rel=1e-10)
            np.testing.assert_allclose(posteriors[0], posteriors[1], atol=1e-10)
