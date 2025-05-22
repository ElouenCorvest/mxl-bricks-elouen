# mxlbricks

## Models

| Name                 | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| Ebenhöh 2011         | PSII & two-state quencher & ATP synthase                                    |
| Ebenhöh 2014         | PETC & state transitions & ATP synthase from Ebenhoeh 2011                  |
| Matuszyńska 2016 NPQ | 2011 + PSII & four-state quencher                                           |
| Matuszyńska 2016 PhD | ?                                                                           |
| Matuszyńska 2019     | Merges PETC (Ebenhöh 2014), NPQ (Matuszynska 2016) and CBB (Poolman 2000)   |
| Saadat 2021          | 2019 + Mehler (Valero ?) & Thioredoxin & extendend PSI states & consumption |
| van Aalst 2023       | Saadat 2021 & Yokota 1985 & Witzel 2010                                     |


## References

| Name         | Description                                           |
| ------------ | ----------------------------------------------------- |
| Poolman 2000 | CBB cycle, based on Pettersson & Ryde-Pettersson 1988 |
| Yokota 1985  | Photorespiration                                      |
| Valero ?     |                                                       |


```
    *,
    kcat: str | None = None,
    e0: str | None = None,
    kms: str | None = None,
    kmp: str | None = None,
    keq: str | None = None,
) -> Model:
    kms = (
        static(model, n.kms(ENZYME), None) if kms is None else kms
    )  # FIXME: source
    kmp = (
        static(model, n.kmp(ENZYME), None) if kmp is None else kmp
    )  # FIXME: source
    kcat = (
        static(model, n.kcat(ENZYME), None) if kcat is None else kcat
    )  # FIXME: source
    e0 = static(model, n.e0(ENZYME), 1.0) if e0 is None else e0  # FIXME: source
    keq = static(model, n.keq(ENZYME), None) if keq is None else keq # FIXME: source
    model.add_derived(vmax := n.vmax(ENZYME), fn=mass_action_1s, args=[kcat, e0])
```


```
    kf: str | None = None,
) -> Model:
    kf = static(model, n.kf(ENZYME), None) if kf is None else kf


    kre: str | None = None,
    keq: str | None = None,
) -> Model:
    kre = static(model, n.kre(ENZYME), 800000000.0) if kre is None else kre
    keq = static(model, n.keq(ENZYME), None) if keq is None else keq
```



## API

```python
def add_reaction_name(
    model: Model,
    e0: str,
    kcat: str,
    km: str,
    s1: str = n.substrate(),  # possibly inject compartment here
    s2: str = n.product(),  # possibly inject compartment here
) -> Model:
    model.add_derived(n.vmax(name), fn=..., args=[e0, kcat])
    model.add_reaction(
        name=name,
        args=[s1, s2, vmax, km, s1, s2]
        stoichiometry=filter_stoichiometry(
            model,
            {...}
        )

    )
    return model

def static(model: Model, name: str, value: float) -> str:
    model.add_parameter(name, value)
    return name

def thioredixon_regulated(model: Model, name: str, value: float) -> str:
    scaled_name = f"{name}_active"

    model.add_parameter(name, value)
    model.add_derived_compound(scaled_name, mul, [base, n.e_active])
    return scaled_name

add_reaction_name(
    model,
    e0=thioredixon_regulated(m, n.e0(rxn), 2.0,)
    kcat=arhennius(m, n.kcat(rxn), 1.6, to_kelvin(25), activation_energy=45_000,)
    km=static(m, n.km(rxn), 0.077),
)
```

