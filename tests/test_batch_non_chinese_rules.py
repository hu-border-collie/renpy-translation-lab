import copy
import unittest

import batch_non_chinese_rules
import gemini_translate_batch as batch_mod


class BatchNonChineseRulesTests(unittest.TestCase):
    def test_defaults_match_legacy_project_paths(self):
        rules = batch_non_chinese_rules.normalize_non_chinese_rules(None)
        self.assertIn('screens_menu_about.rpy', rules['static_name_credit_rel_paths'])
        self.assertIn('screens_patronlistitem.rpy', rules['static_name_credit_unconditional_rel_paths'])
        self.assertIn('screens_charselect.rpy', rules['charselect_rel_paths'])

    def test_extra_static_paths_merge_into_static_name_credit_list(self):
        rules = batch_non_chinese_rules.normalize_non_chinese_rules(
            {
                'extra_static_name_credit_rel_paths': [
                    'custom_credits.rpy',
                    'screens_menu_about.rpy',
                ],
            }
        )
        self.assertIn('custom_credits.rpy', rules['static_name_credit_rel_paths'])
        self.assertEqual(
            rules['static_name_credit_rel_paths'].count('screens_menu_about.rpy'),
            1,
        )

    def test_manifest_rules_override_runtime_rules(self):
        manifest_rules = {
            'static_name_credit_rel_paths': ['custom_only.rpy'],
            'static_name_credit_unconditional_rel_paths': [],
            'charselect_rel_paths': [],
            'player_name_comparison_rel_paths': [],
            'define_rel_path_suffixes': [],
            'define_rel_path_prefixes': [],
        }
        effective = batch_non_chinese_rules.effective_non_chinese_rules(
            {'non_chinese_rules': manifest_rules},
            runtime_rules=batch_non_chinese_rules.DEFAULT_NON_CHINESE_RULES,
        )
        self.assertEqual(effective['static_name_credit_rel_paths'], ['custom_only.rpy'])

    def test_custom_manifest_rule_allows_configured_static_file(self):
        rules = {
            'static_name_credit_rel_paths': ['custom_credits.rpy'],
            'static_name_credit_unconditional_rel_paths': [],
            'charselect_rel_paths': [],
            'player_name_comparison_rel_paths': [],
            'define_rel_path_suffixes': [],
            'define_rel_path_prefixes': [],
        }
        manifest = {'non_chinese_rules': rules}
        chunk = {'file_rel_path': 'custom_credits.rpy'}
        self.assertTrue(
            batch_mod.is_manifest_static_non_chinese_item(
                manifest,
                chunk,
                'Avi, MJ, Sinta, Steven.',
                'Avi, MJ, Sinta, Steven。',
            )
        )
        self.assertFalse(
            batch_mod.is_manifest_static_non_chinese_item(
                manifest,
                chunk,
                'Main Writer: Andy Peng',
                'Main Writer: Andy Peng',
            )
        )

    def test_default_rules_still_allow_legacy_static_context_items(self):
        manifest = {
            'non_chinese_rules': copy.deepcopy(batch_non_chinese_rules.DEFAULT_NON_CHINESE_RULES),
        }
        self.assertTrue(
            batch_mod.allow_non_chinese_batch_translation(
                manifest,
                {'file_rel_path': 'screens_patronlistitem.rpy'},
                'Alpha, Beta, Gamma',
                'Alpha, Beta, Gamma',
            )
        )


if __name__ == '__main__':
    unittest.main()