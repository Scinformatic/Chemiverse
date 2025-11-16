from typing import Self, Literal

import pandas as pd
import scicoda

_PERIODIC_TABLE = scicoda.atom.periodic_table()
_Z_TO_SYMBOL: dict[int, str] = _PERIODIC_TABLE.set_index("z")["symbol"].to_dict()
_SYMBOL_TO_Z: dict[str, int] = _PERIODIC_TABLE.set_index("symbol")["z"].to_dict()
_Z_TO_EN: dict[int, float] = _PERIODIC_TABLE.set_index("z")["en_pauling"].to_dict()


class MolecularFormula:
    """Molecular formula.

    Parameters
    ----------
    atom_count
        Molecular formula as a dictionary
        mapping atomic numbers or symbols to their counts.
    """

    def __init__(
        self,
        atom_count: dict[int | str, int],
        charge: int = 0,
    ):
        self._z_to_count = {}
        self._symbol_to_count = {}
        for key, count in atom_count.items():
            if isinstance(key, int):
                z = key
                symbol = _Z_TO_SYMBOL[z]
            elif isinstance(key, str):
                symbol = key.strip().capitalize()
                z = _SYMBOL_TO_Z[symbol]
            else:
                raise TypeError(f"Invalid key type: {type(key)}")
            self._z_to_count[z] = count
            self._symbol_to_count[symbol] = count
        self._charge = charge
        return

    @property
    def latex(self) -> str:
        """Molecular formula formatted for LaTeX.

        The elements are sorted following the standard IUPAC convention:
        - If carbon is present, it is listed first, followed by hydrogen,
          and then the other elements in alphabetical order.
        - Otherwise, from least to most electronegative element.

        The formula also includes the charge of the system as a superscript.
        """
        parts = []
        for symbol, count in self._symbol_to_count.items():
            if count == 1:
                parts.append(f"{symbol}")
            else:
                parts.append(rf"{symbol}\textsubscript{{{count}}}")
        charge = self._charge
        if charge != 0:
            abs_charge = abs(charge)
            charge_str = abs_charge if abs_charge != 1 else ""
            parts.append(rf"\textsuperscript{{{charge_str}{"+" if charge > 0 else "–"}}}")
        return "".join(parts)

    def sort(self, method: Literal["iupac"]) -> Self:
        """Sort the molecular formula.

        Parameters
        ----------
        method
            The sorting method to use:
            - "iupac": Sort following the standard IUPAC convention:
              - If carbon is present, it is listed first, followed by hydrogen,
                and then the other elements in alphabetical order.
              - Otherwise, atoms are sorted from least to most electronegative element.
        """
        if method == "iupac":
            return self._sort_iupac()
        raise ValueError(f"Invalid sorting method: {method}")

    def _sort_iupac(self) -> Self:
        def en_sorter(symbol: str) -> float:
            z = _SYMBOL_TO_Z[symbol]
            en = _Z_TO_EN[z]
            return float("inf") if pd.isna(en) else en

        counts = self._symbol_to_count
        if "C" in counts:
            sorted_elements = ["C", "H"] + sorted(e for e in counts if e not in ("C", "H"))
        else:
            # Sort by electronegativity ascending (lowest first = most electropositive)
            sorted_elements = sorted(counts.keys(), key=en_sorter)
        formula = {symbol: counts[symbol] for symbol in sorted_elements}
        return MolecularFormula(formula, charge=self._charge)


def from_counts(
    counts: dict[str | int, int],
    charge: int = 0,
    sort: Literal["iupac"] | None = "iupac",
) -> MolecularFormula:
    """Create a MolecularFormula from a dictionary of element counts.

    Parameters
    ----------
    counts
        A dictionary mapping element symbols or atomic numbers to their counts.
    charge
        The overall charge of the molecule.
    sort
        The sorting method to use for the elements in the formula.
        If None, the elements are not sorted.
    """
    formula = MolecularFormula(counts, charge=charge)
    if sort:
        formula = formula.sort(method=sort)
    return formula
