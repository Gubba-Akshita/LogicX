from evops.data_adapter import CrisisMMDAdapter


def test_event_split_correctness():
    adapter = CrisisMMDAdapter(random_seed=42)
    df = adapter.load_data()
    ood_event = "hurricane_irma"
    splits = adapter.create_event_splits(df, ood_event=ood_event)

    assert ood_event not in splits["train"]["event_name"].values, "OOD event found in training split."
    assert all(splits["ood"]["event_name"] == ood_event), "OOD split contains non-OOD events."
