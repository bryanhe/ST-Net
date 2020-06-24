import stnet.utils.util

def print_spatial_patients(args):
    """
    Prints the name of the patients.
    """
    patient = stnet.utils.util.get_spatial_patients()
    for p in sorted(patient.keys()):
        print(p)

def print_spatial_sections(args):
    """
    Prints the name of the sections.
    """
    patient = stnet.utils.util.get_spatial_patients()
    for p in sorted(patient.keys()):
        for s in sorted(patient[p]):
            print("{}_{}".format(p, s))

def print_ensg_name(args):
    """
    Prints the ENGS name of the gene with the index-th highest mean expression (0-indexed).
    """
    dataset = stnet.datasets.Spatial(gene_filter=None)
    if isinstance(args.index, int):
        args.index = [args.index]

    x = sorted(zip(dataset.mean_expression, dataset.ensg_names, dataset.gene_names))[::-1]
    for i in args.index:
        print(x[i][1])
