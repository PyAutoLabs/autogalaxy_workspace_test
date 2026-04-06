"""
Integration test: aggregator Galaxies scrape.

Exercises GalaxiesAgg randomly_drawn_via_pdf_gen_from and
all_above_weight_gen_from with weight verification.
"""
import os
import shutil
from os import path

from autoconf import conf
from autoconf.conf import with_config
import autofit as af
import autogalaxy as ag
from autogalaxy import fixtures
from autofit.non_linear.samples import Sample

os.environ["PYAUTOFIT_TEST_MODE"] = "1"

directory = path.dirname(path.realpath(__file__))

conf.instance.push(
    new_path=path.join(directory, "config"),
    output_path=path.join(directory, "output"),
)

database_file = "db_galaxies"


def clean():
    database_sqlite = path.join(conf.instance.output_path, f"{database_file}.sqlite")
    if path.exists(database_sqlite):
        os.remove(database_sqlite)
    result_path = path.join(conf.instance.output_path, database_file)
    if path.exists(result_path):
        shutil.rmtree(result_path)


@with_config("general", "output", "samples_to_csv", value=True)
def aggregator_from(analysis, model, samples):
    result_path = path.join(conf.instance.output_path, database_file)
    clean()
    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=database_file)
    search.fit(model=model, analysis=analysis)
    analysis.visualize_before_fit(paths=search.paths, model=model)
    db_path = path.join(conf.instance.output_path, f"{database_file}.sqlite")
    agg = af.Aggregator.from_database(filename=db_path)
    agg.add_directory(directory=result_path)
    return agg


def make_model():
    dataset_model = af.Model(ag.DatasetModel)
    dataset_model.background_sky_level = af.UniformPrior(
        lower_limit=0.5, upper_limit=1.5
    )
    return af.Collection(
        dataset_model=dataset_model,
        galaxies=af.Collection(
            g0=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.Sersic),
            g1=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.Sersic),
        ),
    )


def make_samples(model):
    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]
    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )
    return ag.m.MockSamples(
        model=model,
        prior_means=[1.0] * model.prior_count,
        sample_list=sample_list,
    )


model = make_model()
samples = make_samples(model)
masked_imaging = fixtures.make_masked_imaging_7x7()
analysis = ag.AnalysisImaging(dataset=masked_imaging, use_jax=False)

# --- Test 1: randomly_drawn_via_pdf_gen_from ---

print("Test 1: galaxies_randomly_drawn_via_pdf_gen_from ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
galaxies_pdf_gen = galaxies_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for galaxies_gen in galaxies_pdf_gen:
    for galaxy_list in galaxies_gen:
        i += 1
        assert galaxy_list[0].g0.redshift == 0.5
        assert galaxy_list[0].g0.light.centre == (10.0, 10.0)
        assert galaxy_list[0].g1.redshift == 1.0
assert i == 2
clean()

print("PASSED")

# --- Test 2: all_above_weight_gen_from ---

print("Test 2: galaxies_all_above_weight_gen ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
galaxies_pdf_gen = galaxies_agg.all_above_weight_gen_from(minimum_weight=-1.0)
weight_pdf_gen = galaxies_agg.weights_above_gen_from(minimum_weight=-1.0)

i = 0
for galaxies_gen, weight_gen in zip(galaxies_pdf_gen, weight_pdf_gen):
    for galaxy_list in galaxies_gen:
        i += 1
        if i == 1:
            assert galaxy_list[0].g0.redshift == 0.5
            assert galaxy_list[0].g0.light.centre == (1.0, 1.0)
            assert galaxy_list[0].g1.redshift == 1.0
        if i == 2:
            assert galaxy_list[0].g0.redshift == 0.5
            assert galaxy_list[0].g0.light.centre == (10.0, 10.0)
            assert galaxy_list[0].g1.redshift == 1.0
    for weight in weight_gen:
        if i == 0:
            assert weight == 0.0
        if i == 1:
            assert weight == 1.0
assert i == 2
clean()

print("PASSED")

print("\nAll galaxies aggregator tests passed.")
