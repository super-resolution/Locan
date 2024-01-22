# Benchmarks

We use Airspeed Velocity (*asv*) to benchmark locan.

In order to use *asv* look up the details in the [ASV documentation](https://asv.readthedocs.io/), or install *asv* and check `asv --help` and `asv run --help`.

## Run benchmarks:

To get benchmarks in an existing conda environment
enter the locan/benchmarks directory with the appropriate configuration file 
and run:

```
asv run --config asv.conf_conda.json -E existing
```

```
asv publish --config asv.conf_conda.json
```

```
asv preview
```