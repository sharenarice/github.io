BrainsConnected site files
==========================

UPLOAD ALL OF THESE TO THE SAME FOLDER. The pages use relative links,
so index.html, quiz.html, resources.html, teach.html, bc-core.css and
the icon/image files must sit side by side.

PAGES
  index.html        Homepage. Updated with favicon + social tags, new nav
                    items, and a "Start here" card row above Latest Stories.
  quiz.html         Two quizzes: device-matching (8 questions) + myth quiz (7).
  resources.html    Tabbed: "I use neurotech" / "I build neurotech".
                    Deep link to the builder tab with resources.html#build
  teach.html        Four K-12 lesson plans, one per grade band.

SHARED STYLES
  bc-core.css       Palette, type, and components for the three new pages.
                    index.html still carries its own inline <style> block.
                    If you change a brand colour, change it in BOTH places,
                    or migrate index.html to this stylesheet later.

ICONS + SOCIAL IMAGE
  favicon.svg           Primary favicon (modern browsers)
  favicon-16/32/48/192  PNG fallbacks
  favicon.ico           Legacy fallback
  apple-touch-icon.png  iOS home screen (opaque tan background)
  og-image.png          1200x630 social preview thumbnail
  og-image.svg          Editable source for og-image.png

BEFORE GOING LIVE
  1. og:image should be an ABSOLUTE url for most social platforms.
     Change  content="og-image.png"
     to      content="https://yourdomain.com/og-image.png"
     in all four HTML files. Relative paths work in some scrapers
     and silently fail in others.
  2. The "Read the brain data privacy policy" link on index.html is
     still href="#". Point it somewhere real before launch.
  3. Fonts load from Google Fonts. Already wired up, nothing to do.

CONTENT TO VERIFY
  - NGSS codes on teach.html are our best read of closest fit.
    Check against your state framework before formal use.
  - Neural data law references (Colorado, California, Chile, UNESCO)
    are described in general terms. This area moves fast; re-check
    before making any stronger claim.
