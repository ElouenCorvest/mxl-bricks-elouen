from __future__ import annotations

from typing import TYPE_CHECKING

from mxlbricks import names as n
from mxlbricks.fns import mass_action_1s
from mxlbricks.utils import static

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


def add_triose_phosphate_exporters(
    model: Model,
    *,
    chl_stroma: str = "",
    e0: str | None = None,
    km_pga: str | None = None,
    km_gap: str | None = None,
    km_dhap: str | None = None,
    km_pi_ext: str | None = None,
    km_pi: str | None = None,
    kcat_export: str | None = None,
) -> Model:
    n_translocator = "N_translocator"
    pga_name = n.ex_pga()
    gap_name = n.ex_gap()
    dhap_name = n.ex_dhap()

    pi_ext = static(model, n.pi_ext(), 0.5)

    km_pga = static(model, n.km(pga_name), 0.25) if km_pga is None else km_pga
    km_gap = static(model, n.km(gap_name), 0.075) if km_gap is None else km_gap
    km_dhap = static(model, n.km(dhap_name), 0.077) if km_dhap is None else km_dhap
    km_pi_ext = static(model, n.km(n_translocator, n.pi_ext()), 0.74)
    km_pi = static(model, n.km(n_translocator, n.pi()), 0.63)

    kcat_export = (
        static(model, n.kcat(n_translocator), 0.25 * 8)
        if kcat_export is None
        else kcat_export
    )

    e0 = static(model, n.e0(n_translocator), 1.0) if e0 is None else e0
    model.add_derived(
        vmax_export := n.vmax(pga_name), fn=mass_action_1s, args=[kcat_export, e0]
    )

    model.add_derived(
        name=n_translocator,
        fn=_rate_translocator,
        args=[
            n.pi(chl_stroma),
            n.pga(chl_stroma),
            n.gap(chl_stroma),
            n.dhap(chl_stroma),
            km_pi_ext,
            pi_ext,
            km_pi,
            km_pga,
            km_gap,
            km_dhap,
        ],
    )

    enzyme_name = pga_name
    model.add_reaction(
        name=enzyme_name,
        fn=_rate_out,
        stoichiometry={
            n.pga(chl_stroma): -1,
        },
        args=[
            n.pga(chl_stroma),
            n_translocator,
            vmax_export,
            km_pga,
        ],
    )

    enzyme_name = gap_name
    model.add_reaction(
        name=enzyme_name,
        fn=_rate_out,
        stoichiometry={
            n.gap(chl_stroma): -1,
        },
        args=[
            n.gap(chl_stroma),
            n_translocator,
            vmax_export,
            km_gap,
        ],
    )

    enzyme_name = dhap_name
    model.add_reaction(
        name=enzyme_name,
        fn=_rate_out,
        stoichiometry={
            n.dhap(chl_stroma): -1,
        },
        args=[
            n.dhap(chl_stroma),
            n_translocator,
            vmax_export,
            km_dhap,
        ],
    )
    return model
