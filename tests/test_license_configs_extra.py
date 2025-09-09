from __future__ import annotations

import pytest

from glopt.license_config import LicenseConfigFactory


def test_roman_p_variants_shape_and_costs() -> None:
    lts = LicenseConfigFactory.get_config("roman_p_2_5")
    assert len(lts) == 2
    solo, group = lts
    assert solo.name.lower().startswith("solo")
    assert solo.cost == pytest.approx(1.0)
    assert solo.min_capacity == 1 and solo.max_capacity == 1
    assert group.name.lower().startswith("group")
    assert group.cost == pytest.approx(2.5)
    assert group.min_capacity == 2 and group.max_capacity >= 1000  # unbounded model


def test_duolingo_p_variants_shape_and_caps() -> None:
    lts = LicenseConfigFactory.get_config("duolingo_p_3_0")
    assert len(lts) == 2
    indiv, fam = lts
    assert indiv.cost == pytest.approx(1.0)
    assert indiv.min_capacity == 1 and indiv.max_capacity == 1
    assert fam.cost == pytest.approx(3.0)
    assert fam.min_capacity == 2 and fam.max_capacity == 6


def test_invalid_dynamic_config_raises() -> None:
    with pytest.raises(ValueError):
        LicenseConfigFactory.get_config("roman_p_notanumber")
    with pytest.raises(ValueError):
        LicenseConfigFactory.get_config("duolingo_p_notanumber")

