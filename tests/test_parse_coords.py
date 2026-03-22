"""Tests for MOPAC output coordinate parsing."""
# ruff: noqa: PLR2004

import pytest

from opensqm.mopac.parse_output import _extract_coords


def test_extract_coords_uses_last_cartesian_block() -> None:
    """Test that _extract_coords uses the last cartesian block correctly."""
    fake_out = """
          CARTESIAN COORDINATES

     1         C         1.0000     0.0000     0.0000
     2         H         2.0000     0.0000     0.0000

Some parameters for the elements used in the calculation
General Reference for PM6: (early citation block — before optimisation output)

     GEOMETRY OPTIMISED

          CARTESIAN COORDINATES

     1         C         3.0000     0.0000     0.0000
     2         H         4.5000     0.0000     0.0000

General Reference for PM6: (end matter)
"""
    df = _extract_coords(fake_out)
    assert len(df) == 2
    assert float(df.iloc[0]["x"]) == 3.0
    assert float(df.iloc[1]["x"]) == 4.5


def test_extract_coords_single_block() -> None:
    """Test that _extract_coords handles a single cartesian block."""
    fake_out = """
          CARTESIAN COORDINATES

     1         O         .0000     .0000     .0000
     2         H         .9613     .0000     .0000

General Reference for PM6:
"""
    df = _extract_coords(fake_out)
    assert len(df) == 2
    assert float(df.iloc[1]["x"]) == pytest.approx(0.9613)


def test_extract_coords_pm6_d3h4x_citation_stops_before_second_table() -> None:
    """PM6-D3H4X uses a different citation line than plain PM6."""
    fake_out = """
          CARTESIAN COORDINATES
     1         C         1.0     0.0     0.0
     2         H         2.0     0.0     0.0
General Reference for PM6-D3H4X: Stewart, J. Mol. Model.

          CARTESIAN COORDINATES
     1         C         9.0     0.0     0.0
     2         H         9.0     0.0     0.0
"""
    df = _extract_coords(fake_out)
    assert len(df) == 2
    assert float(df.iloc[0]["x"]) == 9.0


def test_extract_coords_stops_before_empirical_formula_mozyme_tail() -> None:
    """MOZYME often omits ``General Reference`` right after the table; charge rows mimic xyz."""
    fake_out = """
          CARTESIAN COORDINATES

     1         C        22.17     6.91    21.13
     2         O        21.50     7.99    21.56


           Empirical Formula: C1 H1  =     2 atoms


              NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS

    ATOM NO.   TYPE          CHARGE      No. of ELECS.   s-Pop       p-Pop
      1          C           0.537155        3.4628     1.07278     2.39007
      2          O          -0.505213        6.5052     1.81708     4.68813
"""
    df = _extract_coords(fake_out)
    assert len(df) == 2
    assert float(df.iloc[0]["x"]) == pytest.approx(22.17)
    assert float(df.iloc[1]["x"]) == pytest.approx(21.50)


def test_extract_coords_duplicate_numbered_runs_keeps_last() -> None:
    """If two tables appear before ATOMIC ORBITAL…, keep the final run 1..N."""
    lines = [
        "          CARTESIAN COORDINATES",
        "",
        "     1         C         1.0     0.0     0.0",
        "     2         H         2.0     0.0     0.0",
        "     1         C         7.0     0.0     0.0",
        "     2         H         8.0     0.0     0.0",
        "General Reference for PM6-D3H4X: x",
        "",
        "          ATOMIC ORBITAL ELECTRON POPULATIONS",
    ]
    df = _extract_coords("\n".join(lines))
    assert len(df) == 2
    assert float(df.iloc[0]["x"]) == 7.0
