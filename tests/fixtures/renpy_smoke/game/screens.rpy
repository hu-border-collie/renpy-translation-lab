screen say(who, what):
    window:
        id "window"

        vbox:
            if who:
                text who id "who"
            text what id "what"
