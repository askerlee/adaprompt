#!/usr/bin/fish
# " | " separated fields like "subject | prompt | folder | dreambooth-class-token".
# Temporary clear the previous cases, so as to only generate the following cases. 
# Undo clearing by commenting out the following line.
set cases
set -a cases "alita | Photography of a Reflexing water a cute sad {} in a wedding dress half submerged in the lake water just the eyes and head above water, glares and reflections like in a mirror, it's raining, depth of field, portrait, kodak portra 400, film grain and nice chromatic bokeh, 105mm f1.4 | alita-water | girl, instagram"
set -Ux cases $cases
