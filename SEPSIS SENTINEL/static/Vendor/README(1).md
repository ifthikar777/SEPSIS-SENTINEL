# Vendored front-end assets

These files are third-party assets bundled locally so the dashboard renders
correctly with **no network access** — an offline or air-gapped ICU machine
would otherwise lose all icons and fall back to a system sans-serif.

Nothing here is authored by this project. Do not edit these files by hand;
re-fetch them with the commands below if they need updating.

## Contents

| Path | Source | Version | License |
|---|---|---|---|
| `outfit.css`, `fonts/*.woff2` | [Google Fonts — Outfit](https://fonts.google.com/specimen/Outfit) | v15 | SIL Open Font License 1.1 |
| `fontawesome.min.css`, `solid.min.css`, `webfonts/fa-solid-900.woff2` | [Font Awesome Free](https://fontawesome.com) | 6.4.0 | Icons CC BY 4.0 · Fonts SIL OFL 1.1 · Code MIT |

Outfit is a variable font, so the two `.woff2` files cover every weight from
300 to 700 (latin and latin-ext subsets respectively).

Only the **solid** Font Awesome family is bundled — it is the only one this
interface uses (`fa-solid`, 53 occurrences). The regular and brands families
are deliberately omitted to keep the payload small. If you introduce a
`fa-regular` or `fa-brands` icon, you must also vendor that family's CSS and
webfont, or the icon will not render.

Total vendored size: ~280 KB.

## Local modifications

Two edits were applied after download, both purely path-related:

1. `outfit.css` — absolute `https://fonts.gstatic.com/s/outfit/v15/…` URLs
   rewritten to relative `fonts/…` paths.
2. `solid.min.css` — `../webfonts/…` rewritten to `webfonts/…` to match this
   directory layout, and the `.ttf` fallback dropped (every browser that can
   run this dashboard supports woff2).

## Refreshing these assets

```bash
# Outfit
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
curl -A "$UA" "https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" -o outfit.css
# then download each URL in outfit.css into fonts/ and rewrite the paths to fonts/<name>.woff2
```

```bash
# Font Awesome (solid only)
BASE="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0"
curl "$BASE/css/fontawesome.min.css" -o fontawesome.min.css
curl "$BASE/css/solid.min.css"       -o solid.min.css
curl "$BASE/webfonts/fa-solid-900.woff2" -o webfonts/fa-solid-900.woff2
# then rewrite ../webfonts/ -> webfonts/ in solid.min.css
```

The `User-Agent` matters for the Outfit request: Google Fonts serves `.ttf`
to unrecognised clients and `.woff2` only to modern browsers.
