#!/bin/bash
# La figure d'annotation automatique : deux vues de la planche de contrôle de l'étape 7,
# recadrées sans leur bandeau français et légendées en anglais pour le chapitre conception.
set -e
P=../../experiments/10_detecteur/controle/planche_00.jpg
T=$(mktemp -d)
convert "$P" -crop 760x536+0+602   +repage "$T/a.png"
convert "$P" -crop 760x536+760+602 +repage "$T/b.png"
convert "$T/a.png" -background white -gravity south -splice 0x42 -gravity south \
        -fill '#16201C' -pointsize 20 -font DejaVu-Sans \
        -annotate +0+12 'a rack from an aisle : every visible label and carton is boxed' "$T/a.png"
convert "$T/b.png" -background white -gravity south -splice 0x42 -gravity south \
        -fill '#16201C' -pointsize 20 -font DejaVu-Sans \
        -annotate +0+12 'close range : the box encloses the visible part, not the object' "$T/b.png"
convert "$T/a.png" "$T/b.png" +append "$T/duo.png"
convert "$T/duo.png" -background white -gravity north -splice 0x54 -gravity northwest \
        -pointsize 22 -font DejaVu-Sans-Bold \
        -fill '#2F9E44' -annotate +28+16 'green : label' \
        -fill '#364FC7' -annotate +228+16 'blue : carton' \
        -fill '#98A4A0' -annotate +432+16 'grey : too hidden, ignored' \
        -pointsize 19 -font DejaVu-Sans -fill '#5F6B66' \
        -annotate +790+18 'visibility judged by physics rays — no box was placed by hand' \
        fig_annotation.png
rm -rf "$T"
identify fig_annotation.png
