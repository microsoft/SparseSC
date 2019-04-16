"""
Warnings used throughout the module.  Exported and inhereited from a commmon
warning class so as to facilitate the filtering of warnings
"""
# pylint: disable=too-few-public-methods, missing-docstring
class SparseSCWarning(RuntimeWarning):pass
class UnpenalizedRecords(SparseSCWarning):pass


