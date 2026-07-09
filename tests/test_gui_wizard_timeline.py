import unittest

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.wizard_timeline import WizardTimeline, resolve_timeline_step_state
except ImportError as exc:
    WizardTimeline = None  # type: ignore[assignment,misc]
    resolve_timeline_step_state = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(resolve_timeline_step_state is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class ResolveTimelineStepStateTests(unittest.TestCase):
    def test_ready_current_step_uses_ready_visual_state(self):
        state = resolve_timeline_step_state(2, 2, "ready")
        self.assertEqual(state, "ready")

    def test_waiting_current_step_uses_waiting_visual_state(self):
        state = resolve_timeline_step_state(1, 1, "waiting")
        self.assertEqual(state, "waiting")

    def test_warning_current_step_uses_warning_without_ready_ambiguity(self):
        state = resolve_timeline_step_state(1, 1, "warning")
        self.assertEqual(state, "warning")

    def test_stale_current_step_uses_warning_without_glow_pulse(self):
        state = resolve_timeline_step_state(1, 1, "stale")
        self.assertEqual(state, "warning")

    def test_done_marks_all_steps_success(self):
        self.assertEqual(resolve_timeline_step_state(0, -1, "done"), "success")
        self.assertEqual(resolve_timeline_step_state(3, -1, "done"), "success")


@unittest.skipIf(WizardTimeline is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class WizardTimelineAnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_ready_status_does_not_start_pulse_timer(self):
        timeline = WizardTimeline()
        timeline.set_steps([("build", "准备"), ("submit", "提交")])
        timeline.show()
        timeline.set_current_step("submit", "ready")
        self.assertFalse(timeline._anim_timer.isActive())

    def test_waiting_status_starts_pulse_timer(self):
        timeline = WizardTimeline()
        timeline.set_steps([("build", "准备"), ("submit", "提交")])
        timeline.show()
        timeline.set_current_step("submit", "waiting")
        self.assertTrue(timeline._anim_timer.isActive())

    def test_running_status_starts_pulse_timer(self):
        timeline = WizardTimeline()
        timeline.set_steps([("build", "准备"), ("submit", "提交")])
        timeline.show()
        timeline.set_current_step("submit", "running")
        self.assertTrue(timeline._anim_timer.isActive())

    def test_stale_status_does_not_start_pulse_timer(self):
        timeline = WizardTimeline()
        timeline.set_steps([("build", "准备"), ("submit", "提交")])
        timeline.show()
        timeline.set_current_step("submit", "stale")
        self.assertFalse(timeline._anim_timer.isActive())

    def test_hide_stops_pulse_timer(self):
        timeline = WizardTimeline()
        timeline.set_steps([("build", "准备"), ("submit", "提交")])
        timeline.show()
        timeline.set_current_step("submit", "running")
        self.assertTrue(timeline._anim_timer.isActive())
        timeline.hide()
        self.assertFalse(timeline._anim_timer.isActive())


if __name__ == "__main__":
    unittest.main()
