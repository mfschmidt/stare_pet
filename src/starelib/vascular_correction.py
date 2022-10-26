import numpy as np
import pandas as pd


def tac_vascular_correction(tacs, regions, vasc_corr_perc, pvc_vasc_tac_fit):
    """ Find several options for fitting data to our model.

        :param DataFrame tacs: a timepoints x regions dataframe of TACs
        :param list regions: column labels indicating tacs regions
        :param int vasc_corr_perc: The percentage of TAC due to vascular effect
                                   between 0 and 100
        :param pvc_vasc_tac_fit: Modeled vascular TAC
        :return DataFrame: corrected tacs, same shape as input tacs
    """

    pct = vasc_corr_perc / 100.0
    vtac = pvc_vasc_tac_fit.activity
    corrected_tacs = pd.DataFrame(
        data=np.zeros(tacs.shape), columns=regions,
    )
    for r in regions:
        corrected_tacs.loc[:, r] = (
                (1 / (1 - pct)) * (tacs.loc[:, r].values - pct * vtac)
        )
    return corrected_tacs
