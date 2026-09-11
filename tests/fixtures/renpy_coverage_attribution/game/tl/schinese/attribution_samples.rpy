# Original synthetic fixture for issue #426. MIT licensed with this repository.
# No third-party game content, dialogue, names, or paths are included.

translate schinese baseline:
    # e "Baseline dialogue."
    e "基线对白。"

translate schinese speaker_label:
    # "Terry" "Hello there."
    "Terry" "你好。"

translate schinese pending_translation:
    # e "This line still needs translation."
    e ""

translate schinese dynamic_template:
    # e "Dynamic greeting."
    e f"动态问候 {player_name}"

translate schinese multiline_dialogue:
    # e "Multi-line dialogue."
    e """多行对白第一行
第二行"""

translate schinese continued_dialogue:
    # e "Continued dialogue."
    e "续行对白第一段\
第二段"

translate schinese legacy_old_marker:
    old legacy "遗留标记"

translate schinese orphan_old_marker:
    old "只剩 old 行。"

translate schinese trailing_comment:
    # e "Normal marker."
    e "正常标记对白。"
    # TODO "internal-only-note"

translate schinese single_quoted_dialogue:
    # Intentionally unmarked translated dialogue for attribution.
    e '单引号对白。'

translate schinese exclusions:
    voice "voice/sample.ogg"
    play music "audio/theme.ogg"
    # e "Dialogue after exclusions."
    e "排除项后的正常对白。"

translate schinese lookalike_dialogue:
    # e "The asset path asset/theme.ogg is visible dialogue."
    e "资源路径 asset/theme.ogg 是可见对白。"
