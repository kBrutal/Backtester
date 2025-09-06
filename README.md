# BACKTESTING ENGINE

This is version 1 of Backtesting Engine.

## How it works?

* Step 1: Clone the repo
```
git clone <repo_link>
```

* Step 2: Load the data in data folder
Load any security's data of any frequency in `.csv` file to the `data` folder.
For Example 3m and 4h frequency data of BTC-USD is updated currently in the folder.

Once the data is uploaded, its relative path to `data/` needs to be updated in the `config.yaml` file. Currently timestamps like `3m`, `5m` are used as tags to access the data. You can modify these tags to custom as per your convenience.

The access of data is controlled by these tags and decided in the `config.yaml` file itself under `backtester:`
`high_time` denotes the smaller frequency dataset used (here, 4h)
`low_time` denotes the larger frequency data used (here, 3m)

Update them with the respective tags.

Also input the dates in `backtester:` for which you want to backtest.


* Step 3: Code out the strategy:

The `BaseStrategy` class looks like:

```
    class BaseStrategy:
        def __init__(
            self,
            high_csv: pd.DataFrame,
            low_csv: pd.DataFrame,
            config: EasyDict,
            glob: EasyDict,
        ):
            self.high_csv = high_csv
            self.shifted_high_csv = high_csv.shift(1)
            self.low_csv = low_csv
            self.config = config
            self.glob = glob
            self.preprocessing()

        def preprocessing(self):
            pass

        def check_long_entry(self, high_pointer: int):
            pass

        def check_short_entry(self, high_pointer: int):
            pass

        def check_long_exit(self, high_pointer: int):
            pass

        def check_short_exit(self, high_pointer: int):
            pass 
```


You need to code out the logics of these functions. Sample stratgies are already given for reference.


* Step 3: Decide the parameters:

The major parameters are decided in `config.yaml` under `backtester`. All the parameter values are self-explanatory.





## 🚧 Work In Progress

1. Currently the position size remains fixed, based on capital, i.e., it takes position with all the capital it has. Variable position size based on capital exposure and amount of assets (fractional also) will be integrated soon.

2. Currently there are no optimiser for selecting best hyperparameters of a strategy. Plan is to add different optimisers that runs parallely (optimised) and any method can be selected based on the need.

3. Fit function needs to be implemented.

4. Forward Testing needs to be implemented.

5. A better way to handle the data through APIs.

6. A better way to present the backtesting results using dashboard maybe.




> This is the starting version of the Backtesting Engine. Will improve this further. Do drop your ideas and suggestions to us.

## 📬 Contact

| Name              | Email                               | LinkedIn                                               | GitHub                               |
|-------------------|-------------------------------------|--------------------------------------------------------|---------------------------------------|
| **Avinandan Sharma** | [avinandansh08@gmail.com](mailto:avinandansh08@gmail.com) | [linkedin.com/in/avinandan-sharma/](https://www.linkedin.com/in/avinandan-sharma/) | [github.com/kBrutal](https://github.com/kBrutal) |
| **Subarno Maji**     | [subarnomaji@gmail.com](mailto:subarnomaji@gmail.com)         | [linkedin.com/in/subarno-maji-6076a425b/](https://www.linkedin.com/in/subarno-maji-6076a425b/)         | [github.com/SubarnoMaji/](https://github.com/SubarnoMaji/)         |
