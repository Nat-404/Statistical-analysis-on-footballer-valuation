#!/bin/sh
# =============================================================================
# app/make_app.sh — Construit "Football Valuation.app" (application Mac à
# double-cliquer) qui ouvre la fenêtre de simulation (app/simulation_app.py).
#
#     sh app/make_app.sh            -> crée l'application dans le projet et dans le dossier du projet sur le Bureau
#
# L'application ne contient que le lanceur : elle utilise le code, les données
# et l'environnement Python (.venv) du projet. À refaire si le projet est déplacé.
# Journal des exécutions : data/cache/app.log
# =============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
APP="$ROOT/Football Valuation.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# 1. Lanceur : la fenêtre, avec le Python du projet
cat > "$APP/Contents/MacOS/launcher" <<EOF
#!/bin/sh
cd "$ROOT"
mkdir -p "$ROOT/data/cache"
# Sur puce Apple, forcer le mode arm64 : lancé par double-clic, le Python "universel"
# démarrerait sinon en mode Intel, incompatible avec numpy/pandas installés pour arm64.
if [ "\$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
  exec /usr/bin/arch -arm64 "$ROOT/.venv/bin/python" -m app.simulation_app >> "$ROOT/data/cache/app.log" 2>&1
fi
exec "$ROOT/.venv/bin/python" -m app.simulation_app >> "$ROOT/data/cache/app.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/launcher"

# 2. Description de l'application pour macOS
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Football Valuation</string>
  <key>CFBundleDisplayName</key><string>Football Valuation</string>
  <key>CFBundleIdentifier</key><string>io.github.nat-404.football-valuation</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSArchitecturePriority</key><array><string>arm64</string><string>x86_64</string></array>
</dict>
</plist>
EOF

# 3. Icône : trajectoires simulées, dans le style des fiches
ICONSET="$(mktemp -d)/icon.iconset"
mkdir -p "$ICONSET"
"$ROOT/.venv/bin/python" -m app.make_icon "$ICONSET/icon_512x512@2x.png"
for size in 16 32 128 256 512; do
  sips -z $size $size "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z $double $double "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/icon.icns"

# 4. Copie dans le dossier du projet sur le Bureau (à défaut, sur le Bureau)
DEST="$HOME/Desktop/Football valuation Projet (players)"
[ -d "$DEST" ] || DEST="$HOME/Desktop"
rm -rf "$DEST/Football Valuation.app"
cp -R "$APP" "$DEST/"
touch "$DEST/Football Valuation.app"
echo "✓ $APP"
echo "✓ $DEST/Football Valuation.app"
