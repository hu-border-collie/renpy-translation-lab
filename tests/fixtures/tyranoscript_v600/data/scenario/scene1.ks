; scene1.ks - dialogue, character names, default glink/ptext
*start|Start

[chara_new name="akane" storage="akane.png" jname="Akane"]

#akane:smile
Hello, world!

#akane
A line with [ruby text="漢字" rt="kanji"] inline markup.

[ptext layer=0 x=40 y=680 text="Start Game" ]

[glink target=*start text="Continue"]

[lang_set name="ch" ]
