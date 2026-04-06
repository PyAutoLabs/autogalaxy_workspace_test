"""
Integration test: aggregator FitImaging scrape.

Exercises FitImagingAgg max_log_likelihood_gen_from, randomly_drawn_via_pdf_gen_from,
all_above_weight_gen_from, and adapt_images round-trip.
Also exercises ImagingAgg dataset_gen_from.
"""
import os
import shutil
from os import path

from autoconf import conf
from autoconf.conf import with_config
import autofit as af
import autogalaxy as ag
from autogalaxy import fixtures
from autoarray.fixtures import (
    make_image_7x7,
    make_psf_3x3,
    make_noise_map_7x7,
    make_mask_2d_7x7,
)
from autofit.non_linear.samples import Sample

os.environ["PYAUTOFIT_TEST_MODE"] = "1"

directory = path.dirname(path.realpath(__file__))

conf.instance.push(
    new_path=path.join(directory, "config"),
    output_path=path.join(directory, "output"),
)

database_file = "db_fit_imaging"


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
analysis = fixtures.make_analysis_imaging_7x7()
adapt_images = fixtures.make_adapt_images_7x7()

# --- Test 1: max_log_likelihood_gen_from ---

print("Test 1: fit_imaging_max_log_likelihood ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
fit_max_lh_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_max_lh_gen:
    assert fit_list[0].galaxies[0].redshift == 0.5
    assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
    assert fit_list[0].dataset_model.background_sky_level == 10.0
clean()

print("PASSED")

# --- Test 2: randomly_drawn_via_pdf_gen_from ---

print("Test 2: fit_imaging_randomly_drawn_via_pdf_gen_from ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=3)

i = 0
for fit_list_gen in fit_pdf_gen:
    for fit_list in fit_list_gen:
        i += 1
        assert fit_list[0].galaxies[0].redshift == 0.5
        assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
        assert fit_list[0].dataset_model.background_sky_level == 10.0
assert i == 3
clean()

print("PASSED")

# --- Test 3: all_above_weight_gen_from ---

print("Test 3: fit_imaging_all_above_weight_gen ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

i = 0
for fit_gen in fit_pdf_gen:
    for fit_list in fit_gen:
        i += 1
        if i == 1:
            assert fit_list[0].galaxies[0].redshift == 0.5
            assert fit_list[0].galaxies[0].light.centre == (1.0, 1.0)
        if i == 2:
            assert fit_list[0].galaxies[0].redshift == 0.5
            assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
assert i == 2
clean()

print("PASSED")

# --- Test 4: adapt_images round-trip ---

print("Test 4: fit_imaging_adapt_images ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for fit_gen in fit_pdf_gen:
    for fit_list in fit_gen:
        i += 1
        assert (
            list(fit_list[0].adapt_images.galaxy_image_dict.values())[0]
            == list(adapt_images.galaxy_name_image_dict.values())[0]
        ).all()
        assert (
            list(fit_list[0].adapt_images.galaxy_image_plane_mesh_grid_dict.values())[0]
            == list(adapt_images.galaxy_name_image_plane_mesh_grid_dict.values())[0]
        ).all()
assert i == 2
clean()

print("PASSED")

# --- Test 5: ImagingAgg dataset_gen_from ---

print("Test 5: imaging_dataset_gen_from ... ", end="")

db_file_imaging = "db_imaging"
imaging = ag.Imaging(
    data=make_image_7x7(),
    psf=make_psf_3x3(),
    noise_map=make_noise_map_7x7(),
    over_sample_size_lp=5,
    over_sample_size_pixelization=3,
)
masked_imaging_oversampled = imaging.apply_mask(mask=make_mask_2d_7x7())
analysis_oversampled = ag.AnalysisImaging(dataset=masked_imaging_oversampled, use_jax=False)

# Use a different database_file for this test
database_sqlite = path.join(conf.instance.output_path, f"{db_file_imaging}.sqlite")
if path.exists(database_sqlite):
    os.remove(database_sqlite)
result_path = path.join(conf.instance.output_path, db_file_imaging)
if path.exists(result_path):
    shutil.rmtree(result_path)

@with_config("general", "output", "samples_to_csv", value=True)
def _agg_imaging():
    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=db_file_imaging)
    search.fit(model=model, analysis=analysis_oversampled)
    analysis_oversampled.visualize_before_fit(paths=search.paths, model=model)
    db = path.join(conf.instance.output_path, f"{db_file_imaging}.sqlite")
    agg = af.Aggregator.from_database(filename=db)
    agg.add_directory(directory=path.join(conf.instance.output_path, db_file_imaging))
    return agg

agg = _agg_imaging()
dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_gen_from()

for dataset_list in dataset_gen:
    assert (dataset_list[0].data == masked_imaging_oversampled.data).all()
    assert dataset_list[0].grids.over_sample_size_lp.slim[0] == 5
    assert dataset_list[0].grids.over_sample_size_pixelization.slim[0] == 3

database_sqlite = path.join(conf.instance.output_path, f"{db_file_imaging}.sqlite")
if path.exists(database_sqlite):
    os.remove(database_sqlite)
result_path = path.join(conf.instance.output_path, db_file_imaging)
if path.exists(result_path):
    shutil.rmtree(result_path)

print("PASSED")

print("\nAll fit_imaging aggregator tests passed.")
