# Name conversion between Ensembl ID and approved symbol/names
# Table obtained from https://www.genenames.org/download/custom/
# Selected:
#     Curated by the HGNC:
#         - Approved symbol
#         - Approved name
#     Downloaded from external sources
#         - Ensembl ID
#     Select status:
#         - Approved
#         - Entry and symbol withdrawn
#
# Direct download from
# https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit

import os
import pandas
import pickle
import collections

class IdentityDict(dict):
    """This variant of a dict defaults to the identity function if a key has
    no corresponding value.

    https://stackoverflow.com/questions/6229073/how-to-make-a-python-dictionary-that-returns-key-for-keys-missing-from-the-dicti
    """
    def __missing__(self, key):
        return key

root = os.path.dirname(os.path.realpath(__file__))
try:
    with open(os.path.join(root, "ensembl.pkl"), "rb") as f:
        symbol = pickle.load(f)
except FileNotFoundError:
    ensembl = pandas.read_csv(os.path.join(root, "ensembl.tsv"), sep="\t")

    # TODO: should just return Ensembl ID if no name available
    symbol = IdentityDict()

    for (index, row) in ensembl.iterrows():
        symbol[row["Ensembl ID(supplied by Ensembl)"]] = row["Approved symbol"]

    with open(os.path.join(root, "ensembl.pkl"), "wb") as f:
        pickle.dump(symbol, f)
