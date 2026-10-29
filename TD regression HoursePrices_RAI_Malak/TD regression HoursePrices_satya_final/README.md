# TD1 – House Prices Regression

This is a lab we did for our **Deep Learning course**.

**By:** Malak Rai & Ayoub Outahar

## What is it about?

The goal of this lab is to predict the price of a house (`SalePrice`) from its characteristics: the area, the neighborhood, the quality, the year it was built, etc. To do this we built a small neural network with **PyTorch**.

## Files

```
.
├── data/
│   ├── train_hp.csv                      # 1460 houses with their price
│   └── test_hp.csv                       # 1459 houses without the price (we have to predict it)
├── regression_house_prices_Exo_TD1.ipynb # our notebook
└── README.md
```

## What we did

1. **Merged train and test** with `pd.concat` so we can prepare both of them the same way.
2. **One-hot encoding**: the text columns are turned into 0/1 columns with `pd.get_dummies(..., dummy_na=True)`. For example `Street` becomes `Street_Pave` and `Street_Grvl`. We also kept a `_nan` column so the model knows when a value was missing.
3. **Missing values**: we replaced them with the **median** of the column. We chose the median instead of the mean because it is not really affected by extreme values.
4. **Normalization**: we used `MinMaxScaler` so all the features are between 0 and 1. We did not scale `SalePrice`.
5. **Split the data again**: the first 1460 rows are the train set (`iloc[:1460]`) and the rest is the test set. Then we split the train set into **1000 rows for training and 460 for validation** (`random_state=42` so we always get the same split).
6. **Log of the price**: the model predicts `log(SalePrice)` and not the price directly. This makes training easier and the error becomes relative to the price.
7. **Mini-batches**: we split the training data into 50 batches of 20 houses.

## Our model

| Layer   | Input | Output | Activation |
|---------|-------|--------|------------|
| hidden1 | 331   | 128    | ReLU       |
| hidden2 | 128   | 64     | ReLU       |
| output  | 64    | 1      | –          |

- 331 inputs because each house has 331 features after the encoding, and 1 output because we only predict the price.
- We used 2 hidden layers so the model can learn more complex relations between the features and the price.
- We used ReLU instead of sigmoid because it learns faster and does not saturate.
- Loss: `MSELoss` on the log prices
- Optimizer: Adam with `lr = 0.0005`
- 50 epochs
- seed = 42

## Results

We got a **validation RMSE of about 0.15 on the log prices**, which means around 15% error on the price on average. We think it is a good result for a simple model.

Some examples from the validation set:

| Predicted price ($) | Real price ($) |
|---------------------|----------------|
| 141,486             | 154,500        |
| 336,866             | 325,000        |
| 95,368              | 115,000        |
| 165,998             | 159,000        |
| 319,895             | 315,500        |

The predictions are pretty close to the real prices. The training and validation loss curves are in the notebook. For the test set, we use `torch.exp` on the output of the model to get back the price in dollars.

## How to run it

You need Python 3 and Jupyter:

```bash
pip install numpy pandas torch scikit-learn matplotlib jupyter
jupyter notebook regression_house_prices_Exo_TD1.ipynb
```

Then just run the cells in order. Make sure you open the notebook from this folder, otherwise the `data/...` paths won't work.

## Dataset

The data comes from the Kaggle competition [House Prices – Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (houses in Ames, Iowa).
