import json
import unittest

import doctor_recommendations as doctor_rec


class DoctorRecommendationTests(unittest.TestCase):
    def test_make_and_format_cli_line_round_trip(self):
        rec = doctor_rec.make_doctor_recommendation(
            doctor_rec.SWITCH_TO_WORK,
            work_dir="C:/Games/Example/work",
        )
        line = doctor_rec.format_doctor_recommendation_cli_line(rec)
        parsed = doctor_rec.parse_doctor_recommendation_cli_line(line)

        self.assertEqual(parsed["code"], doctor_rec.SWITCH_TO_WORK)
        self.assertEqual(parsed["params"]["work_dir"], "C:/Games/Example/work")

    def test_legacy_string_maps_to_code(self):
        rec = doctor_rec.legacy_string_to_recommendation(
            "RAG store is enabled but empty; run bootstrap-rag before batch translation."
        )

        self.assertEqual(rec["code"], doctor_rec.BOOTSTRAP_RAG)

    def test_cli_line_json_contains_code_and_detail(self):
        rec = doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_WORK)
        payload = json.loads(doctor_rec.format_doctor_recommendation_cli_line(rec))

        self.assertEqual(payload["code"], doctor_rec.BOOTSTRAP_WORK)
        self.assertIn("detail", payload)
        self.assertIn("bootstrap-work", payload["detail"])


if __name__ == "__main__":
    unittest.main()