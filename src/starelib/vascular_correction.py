import numpy as np
import pandas as pd

from .plotting import plot_before_and_after_tacs


def tac_vascular_correction(results):
    """ Find several options for fitting data to our model.

        :param StareResults results: An object storing pipeline data
        :return DataFrame: corrected tacs, same shape as input tacs
    """

    rpt_sect = results.report.begin_section("Regional TAC vascular correction")

    pct = results.args.vasc_corr_pct / 100.0
    fit_tac = results.fitted_tac.activity
    corrected_tacs = pd.DataFrame(
        data=np.zeros(results.tacs.shape),
        columns=results.regions,
    )
    for r in results.regions:
        corrected_tacs.loc[:, r] = (
            (1 / (1 - pct)) * (results.tacs.loc[:, r].values - pct * fit_tac)
        )

    # Write out the corrected tacs as a csv file
    corrected_tacs.to_csv(
        results.args.debug_path / "step-3_corrected_tacs.tsv",
        sep='\t', index=False,
    )
    fig = plot_before_and_after_tacs(
        results.tacs, corrected_tacs, results.mid_times,
    )
    fig.savefig(results.args.fig_path / "step-3_vascular_corrected_tacs.png")
    caption = "TACs before and after vascular correction"
    rpt_sect.add_figure(
        results.args.fig_path / "step-3_vascular_corrected_tacs.png",
        caption
    )

    results.corrected_tacs = corrected_tacs

    rpt_sect.end()
    return results
