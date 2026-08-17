import json
import os
import tempfile
import unittest
from pathlib import Path

import translation_quality as quality


def subject(**overrides):
    value = {
        'item_id': 'item-1',
        'file_rel_path': 'script.rpy',
        'line': 4,
        'line_number': 5,
        'start': 10,
        'end': 20,
        'source': 'Hello Sir',
        'translation': '你好，Sir。',
        'speaker_id': 'ck',
        'speaker_name': 'Church Knight',
    }
    value.update(overrides)
    return value


class QualityPolicyTests(unittest.TestCase):
    def test_normalize_policy_defaults_all_rules_to_warning(self):
        policy = quality.normalize_policy(None)

        self.assertTrue(policy['enabled'])
        self.assertEqual(
            set(policy['rules'].values()),
            {quality.DISPOSITION_WARNING},
        )
        self.assertIn('Ren\'Py', policy['allowed_latin_tokens'])

    def test_normalize_policy_accepts_short_rule_keys_and_blockers(self):
        policy = quality.normalize_policy(
            {
                'enabled': True,
                'rules': {
                    'renpy_wait_inside_cjk': 'blocker',
                    quality.REASON_CJK_LATIN_SPACING: 'off',
                    'unknown_rule': 'blocker',
                },
                'allowed_latin_tokens': ['Nier'],
            }
        )

        self.assertEqual(
            policy['rules']['renpy_wait_inside_cjk'],
            quality.DISPOSITION_BLOCKER,
        )
        self.assertEqual(
            policy['rules']['cjk_latin_spacing'],
            quality.DISPOSITION_OFF,
        )
        self.assertIn('Nier', policy['allowed_latin_tokens'])

    def test_empty_allowed_latin_tokens_keeps_builtin_allowlist(self):
        policy = quality.normalize_policy({'allowed_latin_tokens': []})

        self.assertIn('HP', policy['allowed_latin_tokens'])

    def test_policy_digest_changes_with_disposition(self):
        base = quality.normalize_policy(None)
        changed = quality.normalize_policy(
            {'rules': {'renpy_wait_inside_cjk': 'blocker'}}
        )

        self.assertNotEqual(quality.policy_digest(base), quality.policy_digest(changed))

    def test_effective_policy_prefers_manifest_snapshot(self):
        manifest_policy = quality.normalize_policy(
            {'rules': {'known_garbled_phrase': 'off'}}
        )
        policy = quality.effective_policy({'quality_policy': manifest_policy})

        self.assertEqual(
            policy['rules']['known_garbled_phrase'],
            quality.DISPOSITION_OFF,
        )


class QualityRuleTests(unittest.TestCase):
    def test_wait_tag_inside_cjk_is_reported(self):
        findings = quality.check_subject(
            subject(translation='你{w=0.5}好'),
        )

        codes = {finding['reason_code'] for finding in findings}
        self.assertIn(quality.REASON_WAIT_TAG_INSIDE_CJK, codes)

    def test_unclosed_delimiters_are_reported(self):
        findings = quality.check_subject(
            subject(translation='你好 {w=0.5'),
        )

        self.assertIn(
            quality.REASON_UNCLOSED_DELIMITERS,
            {finding['reason_code'] for finding in findings},
        )

    def test_escaped_square_brackets_and_whitespace_tags_are_not_broken(self):
        for translation in (
            '你好[[世界]]',
            '你好 {image=bg room} 世界',
            '按 {{ 打开菜单 }}',
        ):
            with self.subTest(translation=translation):
                findings = quality.check_subject(subject(translation=translation))
                self.assertNotIn(
                    quality.REASON_UNCLOSED_DELIMITERS,
                    {finding['reason_code'] for finding in findings},
                )

    def test_english_suffix_adjacent_reports_ping_and_s(self):
        for translation in ('迷踪步ping', '残片s'):
            with self.subTest(translation=translation):
                findings = quality.check_subject(subject(translation=translation))
                self.assertIn(
                    quality.REASON_ENGLISH_SUFFIX_ADJACENT,
                    {finding['reason_code'] for finding in findings},
                )

    def test_suspicious_english_residue_excludes_allowlisted_tokens(self):
        allowed = quality.check_subject(
            subject(translation='当前HP为100'),
            policy=quality.normalize_policy(None),
        )
        self.assertNotIn(
            quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE,
            {finding['reason_code'] for finding in allowed},
        )

        flagged = quality.check_subject(
            subject(translation='迷踪步ping'),
            policy=quality.normalize_policy({'allowed_latin_tokens': []}),
        )
        self.assertIn(
            quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE,
            {finding['reason_code'] for finding in flagged},
        )

    def test_cjk_latin_spacing_reports_adjacent_token(self):
        findings = quality.check_subject(subject(translation='这是iPhone手机'))

        matching = [
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_CJK_LATIN_SPACING
        ]
        self.assertTrue(matching)
        self.assertIn('iPhone', matching[0]['evidence'])

    def test_markup_stripped_evidence_spans_still_point_at_original_text(self):
        translation = '中文{w}iPhone'
        findings = quality.check_subject(subject(translation=translation))

        residue = [
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE
        ]
        self.assertTrue(residue)
        evidence = json.loads(residue[0]['evidence'])
        start, end = evidence['span']
        self.assertEqual(translation[start:end], evidence['token'])

    def test_cjk_latin_spacing_ignores_renpy_tags(self):
        for translation in ('好的{w}', '他{b}突然{/b}说', '你好{color=#ff0000}'):
            with self.subTest(translation=translation):
                findings = quality.check_subject(subject(translation=translation))
                self.assertNotIn(
                    quality.REASON_CJK_LATIN_SPACING,
                    {finding['reason_code'] for finding in findings},
                )

    def test_english_suffix_adjacent_honors_allowlist(self):
        policy = quality.normalize_policy({'allowed_latin_tokens': ['cos']})
        findings = quality.check_subject(
            subject(translation='中文cos'),
            policy=policy,
        )

        self.assertNotIn(
            quality.REASON_ENGLISH_SUFFIX_ADJACENT,
            {finding['reason_code'] for finding in findings},
        )

    def test_halfwidth_punctuation_and_ascii_ellipsis(self):
        findings = quality.check_subject(subject(translation='你好,世界...'))

        codes = {finding['reason_code'] for finding in findings}
        self.assertIn(quality.REASON_HALFWIDTH_PUNCTUATION, codes)
        self.assertIn(quality.REASON_ASCII_ELLIPSIS, codes)

    def test_glossary_short_term_does_not_match_inside_other_words(self):
        findings = quality.check_subject(
            subject(
                source='The art room is open.',
                translation='心脏在跳动。',
            ),
            glossary_map={'art': '艺术'},
        )

        self.assertNotIn(
            quality.REASON_GLOSSARY_TERM_NOT_APPLIED,
            {finding['reason_code'] for finding in findings},
        )

        findings = quality.check_subject(
            subject(
                source='The art room is open.',
                translation='art 房间开着。',
            ),
            glossary_map={'art': '艺术'},
        )
        self.assertIn(
            quality.REASON_GLOSSARY_TERM_NOT_APPLIED,
            {finding['reason_code'] for finding in findings},
        )

    def test_glossary_relative_path_resolves_against_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'project'
            (base / 'nested').mkdir(parents=True)
            glossary_path = base / 'nested' / 'glossary.json'
            glossary_path.write_text(
                json.dumps(
                    {'normalize_map': {'Church Knight': '教会骑士'}},
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            loaded = quality.load_glossary_map(
                'nested/glossary.json',
                base_dir=str(base),
            )

        self.assertEqual(loaded['Church Knight'], '教会骑士')

    def test_glossary_rule_flags_visible_source_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            glossary_path = Path(tmp) / 'glossary.json'
            glossary_path.write_text(
                json.dumps(
                    {'normalize_map': {'Church Knight': '教会骑士'}},
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            findings = quality.check_subject(
                subject(
                    source='Church Knight says hello.',
                    translation='Church Knight说你好。',
                ),
                glossary_map={'Church Knight': '教会骑士'},
            )

        self.assertIn(
            quality.REASON_GLOSSARY_TERM_NOT_APPLIED,
            {finding['reason_code'] for finding in findings},
        )

    def test_speaker_label_rule_reports_visible_occupation_hint(self):
        findings = quality.check_subject(
            subject(
                source='Welcome.',
                translation='欢迎，Church Knight。',
                speaker_name='Church Knight',
            )
        )

        matching = [
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_SPEAKER_LABEL_UNTRANSLATED
        ]
        self.assertTrue(matching)
        self.assertIn('Church Knight', matching[0]['evidence'])

    def test_untranslated_speaker_label_reports_even_with_chinese_body(self):
        findings = quality.check_subject(
            subject(
                source='Welcome.',
                translation='欢迎。',
                speaker_name='Church Knight',
            )
        )

        matching = [
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_SPEAKER_LABEL_UNTRANSLATED
        ]
        self.assertTrue(matching)
        self.assertIn('Church Knight', matching[0]['evidence'])

    def test_translated_speaker_label_evidence_suppresses_rule(self):
        findings = quality.check_subject(
            subject(
                source='Welcome.',
                translation='欢迎。',
                speaker_name='Church Knight',
                speaker_name_translation='教会骑士',
            )
        )

        self.assertNotIn(
            quality.REASON_SPEAKER_LABEL_UNTRANSLATED,
            {finding['reason_code'] for finding in findings},
        )

    def test_single_word_speaker_label_rule_is_not_missed(self):
        findings = quality.check_subject(
            subject(
                source='Welcome.',
                translation='欢迎，Bouncer。',
                speaker_name='Bouncer',
            )
        )

        matching = [
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_SPEAKER_LABEL_UNTRANSLATED
        ]
        self.assertTrue(matching)
        self.assertIn('Bouncer', matching[0]['evidence'])

    def test_interjection_rule_reports_unchanged_short_interjection(self):
        findings = quality.check_subject(
            subject(source='Oh!', translation='Oh!')
        )

        self.assertIn(
            quality.REASON_INTERJECTION_UNTRANSLATED,
            {finding['reason_code'] for finding in findings},
        )

    def test_configured_garbled_phrase_is_reported(self):
        policy = quality.normalize_policy({'garbled_phrases': ['迷踪步ping']})
        findings = quality.check_subject(
            subject(translation='这是迷踪步ping吗'),
            policy=policy,
        )

        self.assertIn(
            quality.REASON_KNOWN_GARBLED_PHRASE,
            {finding['reason_code'] for finding in findings},
        )

    def test_disabled_rule_produces_no_finding(self):
        policy = quality.normalize_policy({'rules': {'renpy_wait_inside_cjk': 'off'}})
        findings = quality.check_subject(
            subject(translation='你{w=0.5}好'),
            policy=policy,
        )

        self.assertNotIn(
            quality.REASON_WAIT_TAG_INSIDE_CJK,
            {finding['reason_code'] for finding in findings},
        )

    def test_blocker_disposition_is_preserved(self):
        policy = quality.normalize_policy({'rules': {'unclosed_delimiters': 'blocker'}})
        findings = quality.check_subject(
            subject(translation='你好 {w=0.5'),
            policy=policy,
        )

        blocker = next(
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_UNCLOSED_DELIMITERS
        )
        self.assertEqual(blocker['disposition'], quality.DISPOSITION_BLOCKER)
        self.assertEqual(blocker['severity'], 'high')

    def test_multiple_hits_for_same_rule_on_same_line_are_both_reported(self):
        subjects = [
            subject(translation='你{w=0.5}好，他{w=0.2}们')
        ]

        findings = quality.check_quality(subjects)
        wait_findings = [
            finding
            for finding in findings
            if finding['reason_code'] == quality.REASON_WAIT_TAG_INSIDE_CJK
        ]

        self.assertEqual(len(wait_findings), 2)
        self.assertNotEqual(wait_findings[0]['finding_id'], wait_findings[1]['finding_id'])

    def test_finding_contract_fields_are_present(self):
        findings = quality.check_subject(subject(translation='你{w=0.5}好'))

        finding = findings[0]
        for field in (
            'finding_id',
            'reason_code',
            'severity',
            'disposition',
            'item_id',
            'file',
            'line',
            'source',
            'translation',
            'evidence',
            'suggestion',
            'rule_version',
            'schema_version',
        ):
            self.assertIn(field, finding)
        self.assertEqual(finding['file'], 'script.rpy')
        self.assertEqual(finding['line'], 5)


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_warning_counts_and_decision(self):
        findings = quality.check_subject(subject(translation='你{w=0.5}好'))

        gate = quality.summarize_quality_gate(findings)

        self.assertEqual(gate['decision'], quality.GATE_NEEDS_REVIEW)
        self.assertTrue(gate['has_warnings'])
        self.assertGreater(gate['warning_count'], 0)
        self.assertEqual(gate['blocker_count'], 0)

    def test_acknowledged_blocker_does_not_consume_warning_budget(self):
        findings = [
            {
                'finding_id': 'warning-1',
                'disposition': quality.DISPOSITION_WARNING,
            },
            {
                'finding_id': 'blocker-1',
                'disposition': quality.DISPOSITION_BLOCKER,
            },
        ]

        gate = quality.summarize_quality_gate(
            findings,
            acknowledged_ids=['warning-1', 'blocker-1'],
        )

        self.assertEqual(gate['acknowledged_count'], 1)
        self.assertEqual(gate['blocker_count'], 1)
        self.assertEqual(gate['decision'], quality.GATE_NEEDS_REVIEW)

    def test_quality_blocker_counts_as_blocker(self):
        policy = quality.normalize_policy({'rules': {'unclosed_delimiters': 'blocker'}})
        findings = quality.check_subject(
            subject(translation='你好 {w=0.5'),
            policy=policy,
        )

        gate = quality.summarize_quality_gate(findings)

        self.assertEqual(gate['blocker_count'], 1)
        self.assertEqual(gate['decision'], quality.GATE_NEEDS_REVIEW)

    def test_overall_status_matrix(self):
        self.assertEqual(
            quality.overall_check_status(
                {'can_apply': True},
                {'has_warnings': False},
            ),
            quality.GATE_READY,
        )
        self.assertEqual(
            quality.overall_check_status(
                {'can_apply': True},
                {'has_warnings': True},
            ),
            quality.GATE_READY_WITH_WARNINGS,
        )
        self.assertEqual(
            quality.overall_check_status(
                {'can_apply': False},
                {'has_warnings': True},
            ),
            quality.GATE_BLOCKED,
        )

    def test_apply_manifest_quality_acknowledgement_updates_gate(self):
        manifest = {
            'last_check_summary': {
                'check_status': 'ready_with_warnings',
                'writeback_gate': {
                    'decision': 'allow',
                    'can_apply': True,
                    'blocker_count': 0,
                    'quality_blocker_count': 0,
                },
            },
            'quality_acknowledged_finding_ids': [],
        }
        findings = [
            {'finding_id': 'w1', 'disposition': 'warning'},
            {'finding_id': 'w2', 'disposition': 'warning'},
            {'finding_id': 'b1', 'disposition': 'blocker'},
        ]

        applied = quality.apply_manifest_quality_acknowledgement(
            manifest,
            findings,
            finding_ids=['w1', 'missing'],
        )

        self.assertEqual(
            applied['manifest']['quality_acknowledged_finding_ids'],
            ['w1'],
        )
        self.assertEqual(applied['quality_gate']['acknowledged_count'], 1)
        self.assertEqual(applied['quality_gate']['decision'], quality.GATE_NEEDS_REVIEW)
        self.assertEqual(applied['selected_ids'], {'w1'})
        self.assertEqual(applied['unmatched'], ['missing'])
        summary = applied['manifest']['last_check_summary']
        self.assertEqual(summary['quality_gate']['acknowledged_count'], 1)
        self.assertEqual(summary['check_status'], 'ready_with_warnings')
        self.assertEqual(summary['writeback_gate']['decision'], 'allow')

    def test_apply_manifest_quality_unacknowledgement_never_unblocks_blocker(self):
        manifest = {
            'last_check_summary': {},
            'quality_acknowledged_finding_ids': ['w1', 'b1'],
        }
        findings = [
            {'finding_id': 'w1', 'disposition': 'warning'},
            {'finding_id': 'b1', 'disposition': 'blocker'},
        ]

        applied = quality.apply_manifest_quality_acknowledgement(
            manifest,
            findings,
            finding_ids=['w1'],
            unack=True,
        )

        self.assertEqual(
            applied['manifest']['quality_acknowledged_finding_ids'],
            ['b1'],
        )
        self.assertEqual(applied['quality_gate']['blocker_count'], 1)
        self.assertEqual(applied['quality_gate']['decision'], quality.GATE_NEEDS_REVIEW)


if __name__ == '__main__':
    unittest.main()
