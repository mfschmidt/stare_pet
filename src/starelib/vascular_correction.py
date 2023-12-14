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
        columns=results.tacs.columns,
        index=results.tacs.index,
    )
    for r in results.regions:
        corrected_tacs.loc[:, r] = (
            (1 / (1 - pct)) * (results.tacs.loc[:, r].values - pct * fit_tac)
        )

    # Write out the corrected tacs as a csv file
    if results.args.vasc_corr_pct == 0:
        report_line = (
            "No vascular correction was applied, so there is "
            "no before or after, just raw TACs."
        )
        caption = "Raw TACs, with no vascular correction"
    else:
        report_line = (
            f"Vascular correction of {pct:0.2%} was applied to the raw TACs."
        )
        caption = "TACs before and after vascular correction"
    fig = plot_before_and_after_tacs(
        results.tacs, corrected_tacs, results.mid_times,
    )
    corrected_tacs.to_csv(
        results.args.fig_path /
        f"sub-{results.args.subject}_step-3_vascular_corrected_tacs.tsv",
        sep='\t', index=False,
    )
    fig.savefig(
        results.args.fig_path /
        f"sub-{results.args.subject}_step-3_vascular_corrected_tacs.png"
    )
    rpt_sect.add_figure(
        results.args.fig_path /
        f"sub-{results.args.subject}_step-3_vascular_corrected_tacs.png",
        caption, css_class='right_fig',
    )

    rpt_sect.add_line(report_line)

    results.corrected_tacs = corrected_tacs

    rpt_sect.end()
    return results
