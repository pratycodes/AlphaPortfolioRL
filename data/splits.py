from datetime import date


def _parse_iso_date(value):
    return date.fromisoformat(value)


def validate_research_dates(config):
    train_start = _parse_iso_date(config.TRAIN_START_DATE)
    train_end = _parse_iso_date(config.TRAIN_END_DATE)
    valid_start = _parse_iso_date(config.VALID_START_DATE)
    valid_end = _parse_iso_date(config.VALID_END_DATE)
    test_start = _parse_iso_date(config.TEST_START_DATE)
    test_end = _parse_iso_date(config.TEST_END_DATE)

    if not train_start < train_end < valid_start < valid_end < test_start < test_end:
        raise ValueError(
            "Expected chronological non-overlapping train/validation/test dates: "
            "TRAIN_START < TRAIN_END < VALID_START < VALID_END < TEST_START < TEST_END"
        )
