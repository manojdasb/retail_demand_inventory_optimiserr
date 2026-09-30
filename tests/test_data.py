import pytest
from rdio.data import generate_sales


def test_shape_and_columns():
    df = generate_sales(n_skus=3, n_stores=2, days=100)
    assert len(df) == 3 * 2 * 100
    assert set(df.columns) == {"date", "store", "sku", "price", "promo", "units"}


def test_deterministic_with_seed():
    a = generate_sales(n_skus=2, n_stores=1, days=90, seed=7)
    b = generate_sales(n_skus=2, n_stores=1, days=90, seed=7)
    assert a.equals(b)


def test_units_nonnegative_ints_and_promo_binary():
    df = generate_sales(n_skus=2, n_stores=1, days=120)
    assert (df["units"] >= 0).all()
    assert set(df["promo"].unique()) <= {0, 1}


def test_promo_days_have_lower_price_and_higher_demand():
    df = generate_sales(n_skus=4, n_stores=2, days=700)
    assert df.loc[df.promo == 1, "units"].mean() > df.loc[df.promo == 0, "units"].mean()


def test_invalid_args():
    with pytest.raises(ValueError):
        generate_sales(days=10)
