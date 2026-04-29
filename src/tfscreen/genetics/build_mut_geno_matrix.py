
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
        If True, skip building the pair index arrays entirely and return empty
        arrays of shape ``(0,)``. Use this when epistasis is disabled to avoid
        iterating over all multi-mutation genotypes. Default is False.

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
    pair_nnz_pair_idx : np.ndarray, shape (nnz,), dtype int32
        Row (pair) index of each nonzero entry in the logical
        ``(num_pair, num_genotype)`` binary pair-genotype matrix, stored in
        COO format.  Empty (shape ``(0,)``) when there are no multi-mutation
        genotypes or when ``skip_pairs=True``.
    pair_nnz_geno_idx : np.ndarray, shape (nnz,), dtype int32
        Column (genotype) index of each nonzero entry, matching
        ``pair_nnz_pair_idx`` element-for-element.
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
        pair_nnz_pair_idx = np.zeros(0, dtype=np.int32)
        pair_nnz_geno_idx = np.zeros(0, dtype=np.int32)
        return mut_labels, pair_labels, mut_geno_matrix, pair_nnz_pair_idx, pair_nnz_geno_idx

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

    # Build COO representation of the (num_pair, num_genotype) binary matrix.
    # For a library of double mutants each genotype contributes exactly one
    # nonzero; triples contribute C(k,2) nonzeros.  This avoids the O(P*G)
    # dense allocation (which can exceed 100 GiB for large libraries).
    rows = []
    cols = []
    for g_idx, muts in enumerate(geno_muts):
        if len(muts) < 2:
            continue
        for i in range(len(muts)):
            for j in range(i + 1, len(muts)):
                a, b = sorted([muts[i], muts[j]])
                label = f"{a}/{b}"
                rows.append(pair_seen[label])
                cols.append(g_idx)

    pair_nnz_pair_idx = np.array(rows, dtype=np.int32)
    pair_nnz_geno_idx = np.array(cols, dtype=np.int32)

    return mut_labels, pair_labels, mut_geno_matrix, pair_nnz_pair_idx, pair_nnz_geno_idx


def apply_pair_matrix(epi, pair_nnz_pair_idx, pair_nnz_geno_idx, num_genotype):
    """Scatter epistasis values to genotypes via COO pair-genotype indices.

    This is the memory-efficient equivalent of ``epi @ pair_geno_matrix`` where
    ``pair_geno_matrix`` is the dense ``(num_pair, num_genotype)`` binary matrix
    stored instead in COO format as ``(pair_nnz_pair_idx, pair_nnz_geno_idx)``.

    Parameters
    ----------
    epi : jnp.ndarray, shape (..., num_pair)
        Epistasis values, one per pair. Leading dimensions are broadcast.
    pair_nnz_pair_idx : array-like, shape (nnz,)
        Row (pair) index of each nonzero.
    pair_nnz_geno_idx : array-like, shape (nnz,)
        Column (genotype) index of each nonzero.
    num_genotype : int
        Size of the genotype dimension in the output.

    Returns
    -------
    jnp.ndarray, shape (..., num_genotype)
        ``result[..., g] = sum_k epi[..., pair_nnz_pair_idx[k]]``
        for all ``k`` where ``pair_nnz_geno_idx[k] == g``.
    """
    import jax.numpy as jnp
    leading = epi.shape[:-1]
    result = jnp.zeros((*leading, num_genotype))
    return result.at[..., pair_nnz_geno_idx].add(epi[..., pair_nnz_pair_idx])
