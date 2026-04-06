"""
Integration test: aggregator FitInterferometer scrape.

Exercises FitInterferometerAgg randomly_drawn_via_pdf_gen_from,
all_above_weight_gen_from, and adapt_images round-trip.
Also exercises InterferometerAgg dataset_gen_from.
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
    make_visibilities_7,
    make_visibilities_noise_map_7,
    make_uv_wavelengths_7x2,
    make_mask_2d_7x7,
)
from autofit.non_linear.samples import Sample

os.environ["PYAUTOFIT_TEST_MODE"] = "1"

directory = path.dirname(path.realpath(__file__))

conf.instance.push(
    new_path=path.join(directory, "config"),
    output_path=path.join(directory, "output"),
)

database_file = "db_fit_interferometer"


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
interferometer_7 = fixtures.make_interferometer_7()
adapt_images = fixtures.make_adapt_images_7x7()
analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

# --- Test 1: randomly_drawn_via_pdf_gen_from ---

print("Test 1: fit_interferometer_randomly_drawn_via_pdf_gen_from ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
fit_agg = ag.agg.FitInterferometerAgg(aggregator=agg)
fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for fit_gen in fit_pdf_gen:
    for fit_list in fit_gen:
        i += 1
        assert fit_list[0].galaxies[0].redshift == 0.5
        assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
        assert fit_list[0].dataset_model.background_sky_level == 10.0
assert i == 2
clean()

print("PASSED")

# --- Test 2: all_above_weight_gen_from ---

print("Test 2: fit_interferometer_all_above_weight_gen ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
fit_agg = ag.agg.FitInterferometerAgg(aggregator=agg)
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

# --- Test 3: adapt_images round-trip ---

print("Test 3: fit_interferometer_adapt_images ... ", end="")

analysis_adapt = ag.AnalysisInterferometer(
    dataset=interferometer_7, adapt_images=adapt_images, use_jax=False
)
analysis_adapt._adapt_images = adapt_images

agg = aggregator_from(analysis=analysis_adapt, model=model, samples=samples)
fit_agg = ag.agg.FitInterferometerAgg(aggregator=agg)
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

# --- Test 4: InterferometerAgg dataset_gen_from ---

print("Test 4: interferometer_dataset_gen_from ... ", end="")

db_file_inter = "db_interferometer"
visibilities_7 = make_visibilities_7()
visibilities_noise_map_7 = make_visibilities_noise_map_7()
uv_wavelengths_7x2 = make_uv_wavelengths_7x2()
mask_2d_7x7 = make_mask_2d_7x7()

interferometer_custom = ag.Interferometer(
    data=visibilities_7,
    noise_map=visibilities_noise_map_7,
    uv_wavelengths=uv_wavelengths_7x2,
    real_space_mask=mask_2d_7x7,
    transformer_class=ag.TransformerDFT,
)
analysis_custom = ag.AnalysisInterferometer(dataset=interferometer_custom, use_jax=False)

database_sqlite = path.join(conf.instance.output_path, f"{db_file_inter}.sqlite")
if path.exists(database_sqlite):
    os.remove(database_sqlite)
result_path = path.join(conf.instance.output_path, db_file_inter)
if path.exists(result_path):
    shutil.rmtree(result_path)

@with_config("general", "output", "samples_to_csv", value=True)
def _agg_inter():
    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=db_file_inter)
    search.fit(model=model, analysis=analysis_custom)
    analysis_custom.visualize_before_fit(paths=search.paths, model=model)
    db = path.join(conf.instance.output_path, f"{db_file_inter}.sqlite")
    agg = af.Aggregator.from_database(filename=db)
    agg.add_directory(directory=path.join(conf.instance.output_path, db_file_inter))
    return agg

agg = _agg_inter()
dataset_agg = ag.agg.InterferometerAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_gen_from()

for dataset_list in dataset_gen:
    assert (dataset_list[0].data == interferometer_custom.data).all()
    assert (dataset_list[0].real_space_mask == mask_2d_7x7).all()
    assert isinstance(dataset_list[0].transformer, ag.TransformerDFT)

database_sqlite = path.join(conf.instance.output_path, f"{db_file_inter}.sqlite")
if path.exists(database_sqlite):
    os.remove(database_sqlite)
result_path = path.join(conf.instance.output_path, db_file_inter)
if path.exists(result_path):
    shutil.rmtree(result_path)

print("PASSED")

print("\nAll fit_interferometer aggregator tests passed.")
