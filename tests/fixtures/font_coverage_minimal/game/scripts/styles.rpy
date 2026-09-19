# Static font with a CJK subset that covers the sample text.
style default:
    font "fonts/test_cjk_subset.ttf"

# Static Latin-only font: the same text is expected to report missing CJK glyphs.
style latin_only:
    font "fonts/test_latin_subset.ttf"

# Static reference to a missing font file.
style missing_file:
    font "fonts/not_here.ttf"

# Static reference to a corrupt font file.
style corrupt_font:
    font "fonts/corrupt.ttf"

# Dynamic expression: must stay unknown, never evaluated.
define gui.dynamic_font = "fonts/" + "test_cjk_subset.ttf"
style dynamic_font:
    font gui.dynamic_font

# FontGroup / fallback chains cannot be reliably resolved without Ren'Py runtime.
style grouped_font:
    font FontGroup().add("fonts/test_cjk_subset.ttf", 0x0000, 0xffff)

# Ren'Py style inheritance syntax must still expose the font property.
style inherited_font is default:
    font "fonts/test_cjk_subset.ttf"
