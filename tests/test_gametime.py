from nba_winprob import gametime


def test_period_lengths():
    assert gametime.period_length_seconds(1) == 720
    assert gametime.period_length_seconds(4) == 720
    assert gametime.period_length_seconds(5) == 300  # OT


def test_seconds_elapsed_regulation():
    assert gametime.seconds_elapsed(1, 720) == 0  # opening tip
    assert gametime.seconds_elapsed(1, 0) == 720  # end of Q1
    assert gametime.seconds_elapsed(3, 360) == 2 * 720 + 360
    assert gametime.seconds_elapsed(4, 0) == 2880  # end of regulation


def test_seconds_elapsed_overtime_is_monotonic():
    assert gametime.seconds_elapsed(5, 300) == 2880  # OT1 start
    assert gametime.seconds_elapsed(5, 0) == 3180
    assert gametime.seconds_elapsed(6, 300) == 3180  # OT2 start continues from OT1


def test_seconds_remaining():
    assert gametime.seconds_remaining(1, 720) == 2880
    assert gametime.seconds_remaining(4, 90) == 90
    # In OT only the current period clock counts
    assert gametime.seconds_remaining(5, 120) == 120
