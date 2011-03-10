from blockplayer import dataset
from blockplayer import table_calibration


def test_tablecalib():
    dataset.load_random_dataset()
    dataset.advance()
    table_calibration.finish_table_calib()
