from __future__ import annotations

from typing import TYPE_CHECKING

from mxlpy import units

from mxlbricks import names as n
from mxlbricks.utils import (
    default_km,
    default_name,
    default_par,
    default_vmax,
)

if TYPE_CHECKING:
    from mxlpy import Model


def _rate_translocator(
    pi: float,
    pga: float,
    gap: float,
    dhap: float,
    k_pxt: float,
    p_ext: float,
    k_pi: float,
    k_pga: float,
    k_gap: float,
    k_dhap: float,
) -> float:
    return 1 + (1 + k_pxt / p_ext) * (
        pi / k_pi + pga / k_pga + gap / k_gap + dhap / k_dhap
    )


def _rate_out(
    s1: float,
    n_total: float,
    vmax_efflux: float,
    k_efflux: float,
) -> float:
    return vmax_efflux * s1 / (n_total * k_efflux)

def _rate_out_pga(
    s1: float,
    n_total: float,
    vmax_efflux: float,
    k_efflux: float,
):
    return 0.75 * _rate_out(s1, n_total, vmax_efflux, k_efflux)


def add_pga_exporter(
    model: Model,
    rxn: str,
    pga: str,
    n_translocator: str,
    vmax_export: str,
    km_pga: str,
) -> Model:
    rxn = default_name(rxn, n.ex_pga)

    model.add_reaction(
        name=rxn,
        fn=_rate_out,
        stoichiometry={
            pga: -1,
        },
        args=[
            pga,
            n_translocator,
            vmax_export,
            km_pga,
        ],
    )
    return model


def add_gap_exporter(
    model: Model,
    rxn: str,
    gap: str,
    n_translocator: str,
    vmax_export: str,
    km_gap: str,
) -> Model:
    rxn = default_name(rxn, n.ex_gap)

    model.add_reaction(
        name=rxn,
        fn=_rate_out,
        stoichiometry={
            gap: -1,
        },
        args=[
            gap,
            n_translocator,
            vmax_export,
            km_gap,
        ],
    )

    return model


def add_dhap_exporter(
    model: Model,
    *,
    rxn: str,
    dhap: str,
    n_translocator: str,
    vmax_export: str,
    km_dhap: str,
) -> Model:
    rxn = default_name(rxn, n.ex_dhap)

    model.add_reaction(
        name=rxn,
        fn=_rate_out,
        stoichiometry={
            dhap: -1,
        },
        args=[
            dhap,
            n_translocator,
            vmax_export,
            km_dhap,
        ],
    )
    return model


def add_triose_phosphate_exporters(
    model: Model,
    *,
    pga_rxn: str | None = None,
    gap_rxn: str | None = None,
    dhap_rxn: str | None = None,
    pi: str | None = None,
    pga: str | None = None,
    gap: str | None = None,
    dhap: str | None = None,
    pi_ext: str | None = None,
    e0: str | None = None,
    km_pga: str | None = None,
    km_gap: str | None = None,
    km_dhap: str | None = None,
    km_pi_ext: str | None = None,
    km_pi: str | None = None,
    kcat_export: str | None = None,
) -> Model:
    n_translocator = "N_translocator"
    pga_rxn = default_name(pga_rxn, n.ex_pga)
    gap_rxn = default_name(gap_rxn, n.ex_gap)
    dhap_rxn = default_name(dhap_rxn, n.ex_dhap)
    pi = default_name(pi, n.pi)
    pga = default_name(pga, n.pga)
    gap = default_name(gap, n.gap)
    dhap = default_name(dhap, n.dhap)

    pi_ext = default_par(model, par=pi_ext, name=n.pi_ext(), value=0.5, unit=units.mmol / units.liter)
    
    km_pga = default_km(model, par=km_pga, rxn=n_translocator, subs=pga, value=0.25, unit=units.mmol / units.liter, source="https://doi.org/10.1016/0005-2728(78)90045-2")
    km_gap = default_km(model, par=km_gap, rxn=n_translocator, subs=gap, value=0.075, unit=units.mmol / units.liter, source="https://doi.org/10.1016/0005-2728(78)90045-2")
    km_dhap = default_km(model, par=km_dhap, rxn=n_translocator, subs=dhap, value=0.077, unit=units.mmol / units.liter, source="https://doi.org/10.1016/0005-2728(78)90045-2")

    vmax_export = default_vmax(
        model,
        e0=e0,
        kcat=kcat_export,
        rxn=n_translocator,
        e0_value=1,
        kcat_value=0.25 * 8,
    )

    model.add_derived(
        name=n_translocator,
        fn=_rate_translocator,
        args=[
            pi,
            pga,
            gap,
            dhap,
            default_km(model, par=km_pi_ext, rxn=n_translocator, subs=pi_ext, value=0.74, unit=units.mmol / units.liter, source="https://doi.org/10.1016/0005-2728(78)90045-2"),
            pi_ext,
            default_km(model, par=km_pi, rxn=n_translocator, subs=pi, value=0.63, unit=units.mmol / units.liter, source="https://doi.org/10.1016/0005-2728(78)90045-2"),
            km_pga,
            km_gap,
            km_dhap,
        ],
    )
    add_pga_exporter(
        model=model,
        rxn=pga_rxn,
        pga=pga,
        n_translocator=n_translocator,
        vmax_export=vmax_export,
        km_pga=km_pga,
    )
    add_gap_exporter(
        model=model,
        rxn=gap_rxn,
        gap=gap,
        n_translocator=n_translocator,
        vmax_export=vmax_export,
        km_gap=km_gap,
    )
    add_dhap_exporter(
        model=model,
        rxn=dhap_rxn,
        dhap=dhap,
        n_translocator=n_translocator,
        vmax_export=vmax_export,
        km_dhap=km_dhap,
    )

    return model
