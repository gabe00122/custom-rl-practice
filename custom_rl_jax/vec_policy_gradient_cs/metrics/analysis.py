import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class Analysis:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def load(cls, path: Path) -> 'Analysis':
        return cls(pd.read_parquet(path))

    def plot(self):
        self.df.plot.line()
        plt.show()


def main():
    analysis = Analysis.load(Path("./multi_search/1/metrics.parquet"))
    analysis.plot()


if __name__ == '__main__':
    main()
