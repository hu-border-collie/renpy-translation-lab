# Minimal branching Ren'Py script for Project Analysis route tests.

label start:
    "Welcome."
    jump hub

label hub:
    "Choose a path."
    menu:
        "Path A":
            jump path_a
        "Path B":
            jump path_b

label path_a:
    "You chose A."
    jump shared_end

label path_b:
    "You chose B."
    # Dynamic jump — must be unresolved, not invented as a fixed target.
    jump expression some_flag
    jump shared_end

label shared_end:
    "Shared ending."
    return
