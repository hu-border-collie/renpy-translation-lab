# Fixture source for translation_plan golden tests (issue #346).
# Hand-written minimal Ren'Py dialogue: two labels (block boundary material),
# speakers, bracketed interpolation, Ren'Py tags, and glossary hit terms
# ("Sample Ensemble", "Director B", "setlist", "B-side").
label chapter01_start:
    scene bg dorm with fade
    g "Hey [CharacterA_name!t], did you finish the Sample Ensemble setlist?"
    p "Almost. {i}Almost{/i} is the operative word, CharacterA."
    g "Director B will flip if we're late to rehearsal again."
    p "Relax. I kept the B-side as the encore, like you asked."

label chapter01_hall:
    scene bg hallway with dissolve
    m "You two. Practice room, five minutes, or the setlist is cut."
    g "On it! Come on, [CharacterB_last!t], move!"
