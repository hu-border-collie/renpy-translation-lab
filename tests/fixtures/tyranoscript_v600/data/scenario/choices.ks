; choices.ks - registered custom tag, unknown macro, dynamic parameter, iscript
*menu|Menu

Plain [nw] text before a [glink text="Save Game" target=*save] choice.

[ptext layer=0 x=10 y=20 text='It\'s fine' ]

[mymacro value="Translate this"]

[unknown_macro caption="Do not auto-translate"]

[ptext layer=0 x=10 y=40 text=&sf.button_label]

[iscript]
// This JavaScript string is intentionally not a candidate.
var notice = "Not a candidate";
[endscript]

; trailing comment
