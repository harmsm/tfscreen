
import numpy as np


def build_mut_geno_matrix(genotypes, skip_pairs=False):
    """
    Build binary indicator matrices mapping mutations and mutation pairs to
    genotypes, from an ordered list of genotype strings.

    Genotype strings use slash-separated notation (e.g., "wt", "M42I",
    "M42I/K84L"). "wt" and entries that do not match the single-mutation
    pattern are treated as the wild-type reference (no mutations).

    Parameters
    ----------
    genotypes : array-like of str
        Ordered list of genotype strings. The column order of the output
        matrices follows this order.
    skip_pairs : bool, optional
        If True, skip building the pair_geno_matrix entirely and return an
        empty ``(0, num_genotype)`` array. Use this when epistasis is disabled
        to avoid allocating a potentially enormous matrix. Default is False.

    Returns
    -------
    mut_labels : list of str
        Unique single-mutation labels in first-seen order (e.g., ["M42I", "K84L"]).
    pair_labels : list of str
        Unique pairwise-mutation labels in canonical "A/B" form (A < B
        alphabetically), one per observed pair in the data.
    mut_geno_matrix : np.ndarray, shape (num_mutation, num_genotype)
        Binary float32 matrix. ``mut_geno_matrix[m, g] == 1`` iff mutation
        ``m`` is present in genotype ``g``.
    pair_geno_matrix : np.ndarray, shape (num_pair, num_genotype)
        Binary float32 matrix. ``pair_geno_matrix[p, g] == 1`` iff both
        mutations of pair ``p`` are present in genotype ``g``. Empty
        (shape ``(0, num_genotype)``) when no multi-mutation genotypes exist
        or when ``skip_pairs=True``.
    """

    genotypes = list(genotypes)
    num_genotype = len(genotypes)

    # Parse each genotype string into its constituent single mutations.
    # "wt" and bare "wt"-like entries contribute no mutations.
    def _parse(g):
        parts = g.split("/")
        return [p for p in parts if p and p.lower() != "wt"]

    geno_muts = [_parse(g) for g in genotypes]

    # Collect unique single mutations in first-seen order.
    mut_seen = {}
    for muts in geno_muts:
        for m in muts:
            if m not in mut_seen:
                mut_seen[m] = len(mut_seen)

    mut_labels = list(mut_seen.keys())
    num_mutation = len(mut_labels)

    # Build mut_geno_matrix [num_mutation, num_genotype].
    mut_geno_matrix = np.zeros((num_mutation, num_genotype), dtype=np.float32)
    for g_idx, muts in enumerate(geno_muts):
        for m in muts:
            mut_geno_matrix[mut_seen[m], g_idx] = 1.0

    # Collect unique observed mutation pairs (from genotypes with >= 2 mutations).
    # Pairs are stored in canonical form: the two mutation strings sorted
    # alphabetically and joined with "/".
    if skip_pairs:
        pair_labels = []
        pair_geno_matrix = np.zeros((0, num_genotype), dtype=np.float32)
        return mut_labels, pair_labels, mut_geno_matrix, pair_geno_matrix

    pair_seen = {}
    for muts in geno_muts:
        if len(muts) < 2:
            continue
        for i in range(len(muts)):
            for j in range(i + 1, len(muts)):
                a, b = sorted([muts[i], muts[j]])
                label = f"{a}/{b}"
                if label not in pair_seen:
                    pair_seen[label] = len(pair_seen)

    pair_labels = list(pair_seen.keys())
    num_pair = len(pair_labels)

    # Build pair_geno_matrix [num_pair, num_genotype].
    pair_geno_matrix = np.zeros((num_pair, num_genotype), dtype=np.float32)
    for g_idx, muts in enumerate(geno_muts):
        if len(muts) < 2:
            continue
        for i in range(len(muts)):
            for j in range(i + 1, len(muts)):
                a, b = sorted([muts[i], muts[j]])
                label = f"{a}/{b}"
                pair_geno_matrix[pair_seen[label], g_idx] = 1.0

    return mut_labels, pair_labels, mut_geno_matrix, pair_geno_matrix
