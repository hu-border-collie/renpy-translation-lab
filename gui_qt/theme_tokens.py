"""Semantic design tokens for light and dark GUI themes.

Centralises every colour, radius, and spacing value that differs between
the light and dark stylesheets.  ``theme_helpers.load_theme_stylesheet``
reads the template QSS and substitutes these tokens via
``string.Template``.

Token naming convention
-----------------------
* ``bg_*``   – background colours
* ``fg_*``   – foreground / text colours
* ``border_*`` – border colours
* ``accent_*`` – accent / brand colours
* ``badge_*`` – status badge colours
* ``gradient_*`` – gradient stop colours
* ``scrollbar_*`` – scrollbar colours
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Light theme tokens
# ---------------------------------------------------------------------------
LIGHT_TOKENS: dict[str, str] = {
    # -- Global backgrounds ---------------------------------------------------
    "bg_window": "#f8fafc",
    "bg_surface": "#ffffff",
    "bg_surface_alt": "#f8fafc",       # Subtle alternation (tables, fields)
    "bg_elevated": "#ffffff",           # GroupBox / cards
    "bg_input": "#ffffff",
    "bg_input_focus": "#fdfdfd",
    "bg_disabled": "#f1f5f9",

    # -- Glassmorphism card backgrounds ----------------------------------------
    "bg_card": "rgba(255, 255, 255, 0.85)",
    "border_card": "rgba(15, 23, 42, 0.06)",
    "border_card_top": "rgba(255, 255, 255, 0.9)",

    # -- Foreground / text -----------------------------------------------------
    "fg_primary": "#0f172a",
    "fg_secondary": "#475569",
    "fg_muted": "#64748b",
    "fg_hint": "#4b5563",
    "fg_body": "#374151",
    "fg_mono": "#334155",
    "fg_disabled": "#64748b",
    "fg_disabled_strong": "#475569",
    "fg_on_accent": "#ffffff",

    # -- Borders ---------------------------------------------------------------
    "border_default": "#cbd5e1",
    "border_light": "#e2e8f0",
    "border_hover": "#94a3b8",

    # -- Accent (blue) ---------------------------------------------------------
    "accent_primary": "#2563eb",
    "accent_primary_hover": "#3b82f6",
    "accent_header": "#1e3a8a",

    # -- Tab styling -----------------------------------------------------------
    "tab_fg": "#64748b",
    "tab_fg_selected": "#2563eb",
    "tab_border_selected": "#2563eb",
    "tab_fg_hover": "#0f172a",
    "tab_border_hover": "#cbd5e1",

    # -- Splitter --------------------------------------------------------------
    "splitter_handle": "#cbd5e1",
    "splitter_handle_hover": "#94a3b8",

    # -- API key dialog --------------------------------------------------------
    "bg_dialog": "#ffffff",
    "fg_dialog": "#0f172a",
    "fg_dialog_label": "#374151",
    "bg_api_status": "#f8fafc",
    "border_api_status": "#e2e8f0",
    "fg_api_status": "#374151",
    "bg_api_input": "#ffffff",
    "border_api_input": "#cbd5e1",
    "fg_api_input": "#0f172a",
    "border_api_input_focus": "#3b82f6",
    "bg_api_input_focus": "#fdfdfd",
    "bg_api_list": "#ffffff",
    "border_api_list": "#e2e8f0",
    "fg_api_list": "#374151",
    "bg_api_list_selected": "#dbeafe",
    "fg_api_list_selected": "#1e40af",

    # -- Primary action buttons (Aurora Gradient) ------------------------------
    "gradient_primary_start": "#3b82f6",
    "gradient_primary_end": "#1d4ed8",
    "gradient_primary_hover_start": "#60a5fa",
    "gradient_primary_hover_end": "#2563eb",
    "gradient_primary_pressed_start": "#1e3a8a",
    "gradient_primary_pressed_end": "#1d4ed8",

    # -- Default buttons -------------------------------------------------------
    "bg_button": "#ffffff",
    "border_button": "#cbd5e1",
    "fg_button": "#334155",
    "bg_button_hover": "#f8fafc",
    "border_button_hover": "#94a3b8",
    "fg_button_hover": "#0f172a",
    "bg_button_pressed": "#f1f5f9",
    "bg_button_disabled": "#f1f5f9",
    "border_button_disabled": "#e2e8f0",
    "fg_button_disabled": "#475569",

    # -- Secondary buttons (mirror of defaults but explicit) -------------------
    "bg_secondary": "#ffffff",
    "border_secondary": "#cbd5e1",
    "fg_secondary_btn": "#475569",
    "bg_secondary_hover": "#f8fafc",
    "border_secondary_hover": "#94a3b8",
    "fg_secondary_hover": "#0f172a",
    "bg_secondary_pressed": "#f1f5f9",
    "bg_secondary_disabled": "#f1f5f9",
    "border_secondary_disabled": "#e2e8f0",
    "fg_secondary_disabled": "#475569",

    # -- Split select button ---------------------------------------------------
    "fg_split_select": "#475569",
    "bg_split_select_hover": "rgba(100, 116, 139, 0.10)",
    "border_split_select_hover": "rgba(100, 116, 139, 0.25)",
    "fg_split_select_hover": "#0f172a",
    "bg_split_select_pressed": "rgba(100, 116, 139, 0.18)",
    "fg_split_select_disabled": "#475569",

    # -- Danger button (kill) --------------------------------------------------
    "gradient_danger_start": "#dc2626",
    "gradient_danger_end": "#ef4444",
    "gradient_danger_hover_start": "#ef4444",
    "gradient_danger_hover_end": "#f87171",
    "gradient_danger_pressed_start": "#b91c1c",
    "gradient_danger_pressed_end": "#dc2626",
    "bg_danger_disabled": "#f1f5f9",
    "border_danger_disabled": "#e2e8f0",
    "fg_danger_disabled": "#cbd5e1",

    # -- Success / apply button ------------------------------------------------
    "gradient_success_start": "#10b981",
    "gradient_success_end": "#047857",
    "gradient_success_hover_start": "#34d399",
    "gradient_success_hover_end": "#059669",
    "gradient_success_pressed_start": "#064e3b",
    "gradient_success_pressed_end": "#065f46",
    "bg_success_disabled": "#f1f5f9",
    "border_success_disabled": "#cbd5e1",
    "fg_success_disabled": "#cbd5e1",

    # -- Log view (terminal) ---------------------------------------------------
    "bg_log": "#0f172a",
    "border_log": "#cbd5e1",
    "fg_log": "#f1f5f9",

    # -- Status badges ---------------------------------------------------------
    "badge_success_fg": "#065f46",
    "badge_success_bg": "rgba(16, 185, 129, 0.08)",
    "badge_success_border": "rgba(16, 185, 129, 0.4)",
    "badge_info_fg": "#1e40af",
    "badge_info_bg": "rgba(59, 130, 246, 0.08)",
    "badge_info_border": "rgba(59, 130, 246, 0.4)",
    "badge_warning_fg": "#92400e",
    "badge_warning_bg": "rgba(245, 158, 11, 0.08)",
    "badge_warning_border": "rgba(245, 158, 11, 0.4)",
    "badge_danger_fg": "#991b1b",
    "badge_danger_bg": "rgba(239, 68, 68, 0.08)",
    "badge_danger_border": "rgba(239, 68, 68, 0.4)",

    # -- Diagnostics -----------------------------------------------------------
    "bg_diag_facts": "#f8fafc",
    "border_diag_facts": "#cbd5e1",
    "fg_diag_facts": "#334155",
    "bg_diag_input": "#f8fafc",
    "border_diag_input": "#e2e8f0",
    "fg_diag_input": "#0f172a",

    # -- GroupBox title --------------------------------------------------------
    "bg_groupbox_title": "#ffffff",
    "fg_groupbox_title": "#2563eb",

    # -- ComboBox dropdown list ------------------------------------------------
    "bg_combo_dropdown": "#ffffff",
    "fg_combo_dropdown": "#0f172a",
    "bg_combo_dropdown_selected": "#dbeafe",
    "fg_combo_dropdown_selected": "#1e40af",
    "border_combo_dropdown": "#cbd5e1",

    # -- Progress bar ----------------------------------------------------------
    "border_progress": "#cbd5e1",
    "bg_progress": "#f8fafc",
    "fg_progress": "#0f172a",
    "gradient_progress_start": "#2563eb",
    "gradient_progress_end": "#60a5fa",

    # -- Table -----------------------------------------------------------------
    "border_table": "#cbd5e1",
    "bg_table": "#ffffff",
    "bg_table_alt": "#f8fafc",
    "fg_table": "#334155",
    "gridline_table": "#e2e8f0",
    "bg_table_hover": "rgba(148, 163, 184, 0.10)",
    "bg_table_selected": "rgba(59, 130, 246, 0.14)",
    "fg_table_selected": "#0f172a",
    "bg_table_header": "#f1f5f9",
    "fg_table_header": "#64748b",
    "border_table_header": "#cbd5e1",

    # -- Findings views --------------------------------------------------------
    "bg_findings": "#ffffff",
    "border_findings": "#cbd5e1",
    "fg_findings": "#334155",

    # -- Section labels --------------------------------------------------------
    "fg_section": "#64748b",
    "fg_workflow_section": "#2563eb",

    # -- Separators ------------------------------------------------------------
    "color_separator": "#e2e8f0",
    "color_action_separator": "#e2e8f0",

    # -- Scrollbar -------------------------------------------------------------
    "scrollbar_handle": "rgba(100, 116, 139, 0.2)",
    "scrollbar_handle_hover": "rgba(100, 116, 139, 0.4)",
}


# ---------------------------------------------------------------------------
# Dark theme tokens
# ---------------------------------------------------------------------------
DARK_TOKENS: dict[str, str] = {
    # -- Global backgrounds ---------------------------------------------------
    "bg_window": "#0b0f19",
    "bg_surface": "#111827",
    "bg_surface_alt": "#0b0f19",
    "bg_elevated": "#161e2e",
    "bg_input": "#0b0f19",
    "bg_input_focus": "#090d16",
    "bg_disabled": "#111827",

    # -- Glassmorphism card backgrounds ----------------------------------------
    "bg_card": "rgba(22, 30, 46, 0.7)",
    "border_card": "rgba(255, 255, 255, 0.05)",
    "border_card_top": "rgba(255, 255, 255, 0.12)",

    # -- Foreground / text -----------------------------------------------------
    "fg_primary": "#f3f4f6",
    "fg_secondary": "#d1d5db",
    "fg_muted": "#9ca3af",
    "fg_hint": "#9ca3af",
    "fg_body": "#d1d5db",
    "fg_mono": "#d1d5db",
    "fg_disabled": "#9ca3af",
    "fg_disabled_strong": "#d1d5db",
    "fg_on_accent": "#ffffff",

    # -- Borders ---------------------------------------------------------------
    "border_default": "#1f2937",
    "border_light": "#1f2937",
    "border_hover": "#4b5563",

    # -- Accent (indigo/blue) --------------------------------------------------
    "accent_primary": "#60a5fa",
    "accent_primary_hover": "#6366f1",
    "accent_header": "#38bdf8",

    # -- Tab styling -----------------------------------------------------------
    "tab_fg": "#9ca3af",
    "tab_fg_selected": "#60a5fa",
    "tab_border_selected": "#3b82f6",
    "tab_fg_hover": "#f3f4f6",
    "tab_border_hover": "#4b5563",

    # -- Splitter --------------------------------------------------------------
    "splitter_handle": "#1f2937",
    "splitter_handle_hover": "#4b5563",

    # -- API key dialog --------------------------------------------------------
    "bg_dialog": "#111827",
    "fg_dialog": "#f3f4f6",
    "fg_dialog_label": "#d1d5db",
    "bg_api_status": "#0b0f19",
    "border_api_status": "#1f2937",
    "fg_api_status": "#d1d5db",
    "bg_api_input": "#0b0f19",
    "border_api_input": "#374151",
    "fg_api_input": "#f3f4f6",
    "border_api_input_focus": "#6366f1",
    "bg_api_input_focus": "#090d16",
    "bg_api_list": "#0b0f19",
    "border_api_list": "#1f2937",
    "fg_api_list": "#cbd5e1",
    "bg_api_list_selected": "#312e81",
    "fg_api_list_selected": "#e0e7ff",

    # -- Primary action buttons ------------------------------------------------
    "gradient_primary_start": "#4f46e5",
    "gradient_primary_end": "#2563eb",
    "gradient_primary_hover_start": "#6366f1",
    "gradient_primary_hover_end": "#3b82f6",
    "gradient_primary_pressed_start": "#3730a3",
    "gradient_primary_pressed_end": "#1d4ed8",

    # -- Default buttons -------------------------------------------------------
    "bg_button": "#1f2937",
    "border_button": "#374151",
    "fg_button": "#d1d5db",
    "bg_button_hover": "#374151",
    "border_button_hover": "#4b5563",
    "fg_button_hover": "#ffffff",
    "bg_button_pressed": "#111827",
    "bg_button_disabled": "#111827",
    "border_button_disabled": "#1f2937",
    "fg_button_disabled": "#9ca3af",

    # -- Secondary buttons -----------------------------------------------------
    "bg_secondary": "#1f2937",
    "border_secondary": "#374151",
    "fg_secondary_btn": "#d1d5db",
    "bg_secondary_hover": "#374151",
    "border_secondary_hover": "#4b5563",
    "fg_secondary_hover": "#ffffff",
    "bg_secondary_pressed": "#111827",
    "bg_secondary_disabled": "#111827",
    "border_secondary_disabled": "#1f2937",
    "fg_secondary_disabled": "#9ca3af",

    # -- Split select button ---------------------------------------------------
    "fg_split_select": "#38bdf8",
    "bg_split_select_hover": "rgba(148, 163, 184, 0.14)",
    "border_split_select_hover": "rgba(148, 163, 184, 0.30)",
    "fg_split_select_hover": "#e2e8f0",
    "bg_split_select_pressed": "rgba(148, 163, 184, 0.22)",
    "fg_split_select_disabled": "#9ca3af",

    # -- Danger button (kill) --------------------------------------------------
    "gradient_danger_start": "#dc2626",
    "gradient_danger_end": "#ef4444",
    "gradient_danger_hover_start": "#ef4444",
    "gradient_danger_hover_end": "#f87171",
    "gradient_danger_pressed_start": "#b91c1c",
    "gradient_danger_pressed_end": "#dc2626",
    "bg_danger_disabled": "#111827",
    "border_danger_disabled": "#1f2937",
    "fg_danger_disabled": "#6b7280",

    # -- Success / apply button ------------------------------------------------
    "gradient_success_start": "#059669",
    "gradient_success_end": "#10b981",
    "gradient_success_hover_start": "#067652",
    "gradient_success_hover_end": "#059669",
    "gradient_success_pressed_start": "#044e37",
    "gradient_success_pressed_end": "#065f46",
    "bg_success_disabled": "#111827",
    "border_success_disabled": "#1f2937",
    "fg_success_disabled": "#6b7280",

    # -- Log view (terminal) ---------------------------------------------------
    "bg_log": "#030712",
    "border_log": "#1f2937",
    "fg_log": "#f3f4f6",

    # -- Status badges ---------------------------------------------------------
    "badge_success_fg": "#34d399",
    "badge_success_bg": "rgba(16, 185, 129, 0.12)",
    "badge_success_border": "rgba(16, 185, 129, 0.35)",
    "badge_info_fg": "#60a5fa",
    "badge_info_bg": "rgba(59, 130, 246, 0.12)",
    "badge_info_border": "rgba(59, 130, 246, 0.35)",
    "badge_warning_fg": "#fbbf24",
    "badge_warning_bg": "rgba(245, 158, 11, 0.12)",
    "badge_warning_border": "rgba(245, 158, 11, 0.35)",
    "badge_danger_fg": "#f87171",
    "badge_danger_bg": "rgba(239, 68, 68, 0.12)",
    "badge_danger_border": "rgba(239, 68, 68, 0.35)",

    # -- Diagnostics -----------------------------------------------------------
    "bg_diag_facts": "#0b0f19",
    "border_diag_facts": "#1f2937",
    "fg_diag_facts": "#d1d5db",
    "bg_diag_input": "#0b0f19",
    "border_diag_input": "#1f2937",
    "fg_diag_input": "#e5e7eb",

    # -- GroupBox title --------------------------------------------------------
    "bg_groupbox_title": "#161e2e",
    "fg_groupbox_title": "#818cf8",

    # -- ComboBox dropdown list ------------------------------------------------
    "bg_combo_dropdown": "#111827",
    "fg_combo_dropdown": "#f3f4f6",
    "bg_combo_dropdown_selected": "#312e81",
    "fg_combo_dropdown_selected": "#e0e7ff",
    "border_combo_dropdown": "#374151",

    # -- Progress bar ----------------------------------------------------------
    "border_progress": "#374151",
    "bg_progress": "#0b0f19",
    "fg_progress": "#ffffff",
    "gradient_progress_start": "#4f46e5",
    "gradient_progress_end": "#06b6d4",

    # -- Table -----------------------------------------------------------------
    "border_table": "#1f2937",
    "bg_table": "#0b0f19",
    "bg_table_alt": "#111c30",
    "fg_table": "#d1d5db",
    "gridline_table": "#1f2937",
    "bg_table_hover": "rgba(148, 163, 184, 0.12)",
    "bg_table_selected": "rgba(59, 130, 246, 0.28)",
    "fg_table_selected": "#f8fafc",
    "bg_table_header": "#111827",
    "fg_table_header": "#9ca3af",
    "border_table_header": "#1f2937",

    # -- Findings views --------------------------------------------------------
    "bg_findings": "#0b0f19",
    "border_findings": "#1f2937",
    "fg_findings": "#d1d5db",

    # -- Section labels --------------------------------------------------------
    "fg_section": "#9ca3af",
    "fg_workflow_section": "#818cf8",

    # -- Separators ------------------------------------------------------------
    "color_separator": "#1f2937",
    "color_action_separator": "#1f2937",

    # -- Scrollbar -------------------------------------------------------------
    "scrollbar_handle": "rgba(156, 163, 175, 0.25)",
    "scrollbar_handle_hover": "rgba(156, 163, 175, 0.45)",
}


def tokens_for_theme(effective_theme: str) -> dict[str, str]:
    """Return the token dict for the given effective theme name."""
    if effective_theme == "dark":
        return dict(DARK_TOKENS)
    return dict(LIGHT_TOKENS)
