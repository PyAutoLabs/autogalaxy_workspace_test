"""
Integration test: aggregator Ellipse scrape.

Exercises EllipsesAgg, FitEllipseAgg, and MultipolesAgg
randomly_drawn_via_pdf_gen_from and all_above_weight_gen_from.
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

os.environ["PYAUTO_TEST_MODE"] = "1"

directory = path.dirname(path.realpath(__file__))

conf.instance.push(
    new_path=path.join(directory, "config"),
    output_path=path.join(directory, "output"),
)


def clean(database_file):
    database_sqlite = path.join(conf.instance.output_path, f"{database_file}.sqlite")
    if path.exists(database_sqlite):
        os.remove(database_sqlite)
    result_path = path.join(conf.instance.output_path, database_file)
    if path.exists(result_path):
        shutil.rmtree(result_path)


@with_config("general", "output", "samples_to_csv", value=True)
def aggregator_from(database_file, analysis, model, samples):
    result_path = path.join(conf.instance.output_path, database_file)
    clean(database_file)
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


# Ellipse model

ellipse_list = af.Collection(af.Model(ag.Ellipse) for _ in range(2))
for i, ellipse in enumerate(ellipse_list):
    ellipse.major_axis = i

multipole_list_model = []
for i in range(len(ellipse_list)):
    multipole_1 = af.Model(ag.EllipseMultipole)
    multipole_1.m = 1
    multipole_2 = af.Model(ag.EllipseMultipole)
    multipole_2.m = 2
    multipole_list_model.append([multipole_1, multipole_2])

model = af.Collection(ellipses=ellipse_list, multipoles=multipole_list_model)

parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]
sample_list = Sample.from_lists(
    model=model,
    parameter_lists=parameters,
    log_likelihood_list=[1.0, 2.0],
    log_prior_list=[0.0, 0.0],
    weight_list=[0.0, 1.0],
)
samples = ag.m.MockSamples(
    model=model,
    prior_means=[1.0] * model.prior_count,
    sample_list=sample_list,
)

masked_imaging = fixtures.make_masked_imaging_7x7()
analysis_ellipse = fixtures.make_analysis_ellipse_7x7()

# --- Test 1: EllipsesAgg randomly_drawn_via_pdf_gen_from ---

print("Test 1: ellipses_randomly_drawn_via_pdf_gen_from ... ", end="")

db_file = "db_ellipses"
analysis = ag.AnalysisEllipse(dataset=masked_imaging, use_jax=False)
agg = aggregator_from(database_file=db_file, analysis=analysis, model=model, samples=samples)
ellipses_agg = ag.agg.EllipsesAgg(aggregator=agg)
ellipses_pdf_gen = ellipses_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for ellipses_gen in ellipses_pdf_gen:
    for ellipses_lists_list in ellipses_gen:
        ellipse_list = ellipses_lists_list[0]
        i += 1
        assert ellipse_list[0].centre == (10.0, 10.0)
        assert ellipse_list[0].major_axis == 0
        assert ellipse_list[1].major_axis == 1
assert i == 2
clean(db_file)

print("PASSED")

# --- Test 2: EllipsesAgg all_above_weight_gen ---

print("Test 2: ellipses_all_above_weight_gen ... ", end="")

agg = aggregator_from(database_file=db_file, analysis=analysis, model=model, samples=samples)
ellipses_agg = ag.agg.EllipsesAgg(aggregator=agg)
ellipses_pdf_gen = ellipses_agg.all_above_weight_gen_from(minimum_weight=-1.0)
weight_pdf_gen = ellipses_agg.weights_above_gen_from(minimum_weight=-1.0)

i = 0
for ellipses_gen, weight_gen in zip(ellipses_pdf_gen, weight_pdf_gen):
    for ellipses_lists_list in ellipses_gen:
        ellipse_list = ellipses_lists_list[0]
        i += 1
        if i == 1:
            assert ellipse_list[0].centre == (1.0, 1.0)
        else:
            assert ellipse_list[0].centre == (10.0, 10.0)
        assert ellipse_list[0].major_axis == 0
        assert ellipse_list[1].major_axis == 1
    for weight in weight_gen:
        if i == 0:
            assert weight == 0.0
        if i == 1:
            assert weight == 1.0
assert i == 2
clean(db_file)

print("PASSED")

# --- Test 3: FitEllipseAgg randomly_drawn_via_pdf_gen_from ---

print("Test 3: fit_ellipse_randomly_drawn_via_pdf_gen_from ... ", end="")

db_file = "db_fit_ellipse"
agg = aggregator_from(database_file=db_file, analysis=analysis_ellipse, model=model, samples=samples)
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for fit_gen in fit_pdf_gen:
    for fit_lists_list in fit_gen:
        fit_list = fit_lists_list[0]
        i += 1
        assert fit_list[0].ellipse.major_axis == 0
        assert fit_list[1].ellipse.major_axis == 1
        assert fit_list[0].multipole_list[0].m == 1
        assert fit_list[0].multipole_list[1].m == 2
        assert fit_list[1].multipole_list[0].m == 1
        assert fit_list[1].multipole_list[1].m == 2
assert i == 2
clean(db_file)

print("PASSED")

# --- Test 4: FitEllipseAgg all_above_weight_gen ---

print("Test 4: fit_ellipse_all_above_weight_gen ... ", end="")

agg = aggregator_from(database_file=db_file, analysis=analysis_ellipse, model=model, samples=samples)
fit_agg = ag.agg.FitEllipseAgg(aggregator=agg)
fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

i = 0
for fit_gen in fit_pdf_gen:
    for fit_lists_list in fit_gen:
        fit_list = fit_lists_list[0]
        i += 1
        if i == 1:
            assert fit_list[0].ellipse.centre == (1.0, 1.0)
        if i == 2:
            assert fit_list[0].ellipse.centre == (10.0, 10.0)
assert i == 2
clean(db_file)

print("PASSED")

# --- Test 5: MultipolesAgg randomly_drawn_via_pdf_gen_from ---

print("Test 5: multipoles_randomly_drawn_via_pdf_gen_from ... ", end="")

db_file = "db_multipoles"
agg = aggregator_from(database_file=db_file, analysis=analysis, model=model, samples=samples)
multipoles_agg = ag.agg.MultipolesAgg(aggregator=agg)
multipoles_pdf_gen = multipoles_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for multipoles_gen in multipoles_pdf_gen:
    for multipoles_lists_list in multipoles_gen:
        multipole_list = multipoles_lists_list[0]
        i += 1
        assert multipole_list[0][0].m == 1
        assert multipole_list[0][1].m == 2
        assert multipole_list[1][0].m == 1
        assert multipole_list[1][1].m == 2
assert i == 2
clean(db_file)

print("PASSED")

# --- Test 6: MultipolesAgg all_above_weight_gen ---

print("Test 6: multipoles_all_above_weight_gen ... ", end="")

agg = aggregator_from(database_file=db_file, analysis=analysis, model=model, samples=samples)
multipoles_agg = ag.agg.MultipolesAgg(aggregator=agg)
multipoles_pdf_gen = multipoles_agg.all_above_weight_gen_from(minimum_weight=-1.0)
weight_pdf_gen = multipoles_agg.weights_above_gen_from(minimum_weight=-1.0)

i = 0
for multipoles_gen, weight_gen in zip(multipoles_pdf_gen, weight_pdf_gen):
    for multipoles_lists_list in multipoles_gen:
        multipole_list = multipoles_lists_list[0]
        i += 1
        assert multipole_list[0][0].m == 1
        assert multipole_list[0][1].m == 2
        assert multipole_list[1][0].m == 1
        assert multipole_list[1][1].m == 2
    for weight in weight_gen:
        if i == 0:
            assert weight == 0.0
        if i == 1:
            assert weight == 1.0
assert i == 2
clean(db_file)

print("PASSED")

print("\nAll ellipse aggregator tests passed.")
