import pandas as pd
from general.diagnostics.peak_detectors import PieceWiseLinearPeakDetector
from curvefit.core.utils import split_by_group

class PeakDetector:

    def __init__(self, df, col_log_derf_obs, col_group, col_t, peaked_groups, not_peaked_groups):
        self.df = df   
        self.col_log_derf_obs = col_log_derf_obs 
        self.col_group = col_group 
        self.col_t = col_t 
        self.peaked_groups = peaked_groups
        self.not_peaked_groups = not_peaked_groups

    def get_peak_detector(self):
        self.df_by_group = split_by_group(self.df, self.col_group)
        log_derf_obs = []
        times = []
        peaked = []
        self.groups = []
        for grp, df in self.df_by_group.items():
            if grp in self.peaked_groups or grp in self.not_peaked_groups:
                log_derf_obs.append(df[self.col_log_derf_obs].to_numpy())
                times.append(df[self.col_t].to_numpy())
                if grp in self.peaked_groups:
                    peaked.append(1)
                else:
                    peaked.append(0)
                self.groups.append(grp)
        
        self.peak_detector = PieceWiseLinearPeakDetector(log_derf_obs, self.groups, times, peaked)
        self.peak_detector.train_peak_classifier() 
        self.grp_to_pred = self.peak_detector.predicted

    def predict_peaked(self):
        for grp, df in self.df_by_group.items():
            if grp not in self.groups:
                self.grp_to_pred[grp] = self.peak_detector.has_peaked(
                    df[self.col_log_derf_obs].to_numpy(), 
                    grp,
                    df[self.col_t].to_numpy(),
                )
        return pd.DataFrame.from_dict(self.grp_to_pred, orient='index')



    


    

