from abc import ABC
from sys import stderr

import numpy as np
from tqdm import tqdm


class Reweighter(ABC):
    def __init__(self,
                 crosstab_df, # actual survey response crosstab
                 sample_df, # sample drawn for survey
                 pop_df, # population from which sample_df was drawn
                 cols, # crosstab demographic column names
                 ):
        self._ct_df = crosstab_df.copy()
        self._pop_df = pop_df.copy()
        self._sample_df = sample_df.copy()
        self._cols = cols


    def reweight(self):
        raise NotImplementedError


    def _apply_weights(self, weights):
        return self._sample_df[self._cols] * weights[self._cols]


    def f(self, weights):
        # TODO: I don't think this is right
        return 1 + ((np.std(weights) / np.mean(weights)) ** 2)


class CellReweighter(Reweighter):
    def reweight(self):
        weight_cols = [c + "_weight" for c in self._cols]
        
        # first generate weights for the crosstab vs. the sample
        ct_v_sample = self._ct_df.join(self._sample_df, rsuffix="_pop")
        for col in self._cols:
            ct_v_sample[col + "_weight"] = ct_v_sample[col + "_pop"] / ct_v_sample[col]

        print(f"Survey crosstab vs. survey sample F = {self.f(ct_v_sample[weight_cols].values)}", file=stderr)

        # next for sample vs. population
        sample_v_pop = self._sample_df.join(self._pop_df, rsuffix="_pop")
        for col in self._cols:
            sample_v_pop[col + "_weight"] = sample_v_pop[col + "_pop"] / sample_v_pop[col]

        print(f"Survey sample vs. population F = {self.f(sample_v_pop[weight_cols].values)}", file=stderr)

        # combine to get final weights
        weights = ct_v_sample[weight_cols] * sample_v_pop[weight_cols]
        weights = weights.rename(columns=dict(zip(weight_cols, self._cols)))

        # apply weights to sample
        # TODO: return the weights too?
        return self._apply_weights(weights)


class RakeReweighter(Reweighter):
    def reweight(self):
        # first generate weights for the crosstab vs. the sample
        ct_v_sample = self._rake(self._ct_df, self._sample_df)
        print(f"Survey crosstab vs. survey sample F = {self.f(ct_v_sample.values)}", file=stderr)

        # next for sample vs. population
        sample_v_pop = self._rake(self._sample_df, self._pop_df)
        print(f"Survey sample vs. population F = {self.f(sample_v_pop.values)}", file=stderr)

        # combine to get final weights
        weights = ct_v_sample * sample_v_pop
        
        # apply weights to sample
        # TODO: return the weights too?
        return self._apply_weights(weights)


    def _rake(self, current_df, target_df):
        expected_col = target_df.sum(axis=0).astype(float)
        expected_row = target_df[self._cols].sum(axis=1).astype(float)
        raked = current_df.copy() * (expected_row.sum() / current_df.sum(axis=0).sum())

        with tqdm() as pbar:
            while True:
                rake_rows = target_df.sum(axis=1) / raked.sum(axis=1)
                raked = raked.mul(rake_rows, axis=0)
                rake_cols = target_df.sum(axis=0) / raked.sum(axis=0)
                raked = raked.mul(rake_cols, axis=1)
                if (raked.sum(axis=1).round(2).equals(expected_row.round(2)) and
                    raked.sum(axis=0).round(2).equals(expected_col.round(2))):
                    break
                else:
                    pbar.update()

        raked = raked.div(current_df)
        return raked
