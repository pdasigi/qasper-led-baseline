from allennlp.common.testing import ModelTestCase

import qasper_baselines.calibrator  # pylint: disable=unused-import
import qasper_baselines.dataset_reader  # pylint: disable=unused-import


class TestQasperCalibrator(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            "fixtures/qasper_calibrator.jsonnet", "fixtures/data/qasper_sample_small.json"
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
