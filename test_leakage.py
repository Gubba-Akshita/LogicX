import pandas as pd
from evops.data_adapter import CrisisMMDAdapter
from evops.validation import check_event_leakage


def test_prevention_of_event_leakage():
    adapter = CrisisMMDAdapter(random_seed=42)
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event="hurricane_irma")

    # Should pass cleanly for valid split
    assert check_event_leakage(splits["train"], splits["ood"], splits["ood_event_name"])

    # Intentionally corrupt split by adding OOD event sample into train
    bad_train = pd.concat([splits["train"], pd.DataFrame([splits["ood"].iloc[0]])], ignore_index=True)
    try:
        check_event_leakage(bad_train, splits["ood"], splits["ood_event_name"])
        assert False, "Expected ValueError on event leakage, but none was raised."
    except ValueError:
        pass
