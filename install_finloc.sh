#!/bin/bash

set -e

# ============================================================
# FINLOC AI - INSTALLER
# ============================================================

echo ""
echo "=============================================="
echo "          FINLOC AI INSTALLER"
echo "=============================================="
echo ""

# ------------------------------------------------------------
# FIND PROJECT DIRECTORY
# ------------------------------------------------------------

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Project directory:"
echo "  $PROJECT_DIR"
echo ""

# ------------------------------------------------------------
# CHECK PYTHON
# ------------------------------------------------------------

if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "ERROR: Python 3 is not installed."
    echo ""
    echo "Please install Python 3 and run this installer again."
    exit 1
fi

echo "Python:"
"$PYTHON" --version
echo ""

# ------------------------------------------------------------
# CHECK MAIN APPLICATION
# ------------------------------------------------------------

if [ ! -f "$PROJECT_DIR/src/app/main.py" ]; then
    echo "ERROR: FINLOC application was not found."
    echo ""
    echo "Expected:"
    echo "  $PROJECT_DIR/src/app/main.py"
    exit 1
fi

# ------------------------------------------------------------
# CREATE VIRTUAL ENVIRONMENT
# ------------------------------------------------------------

if [ ! -d "$PROJECT_DIR/.venv" ]; then

    echo "Creating Python virtual environment..."

    "$PYTHON" -m venv "$PROJECT_DIR/.venv"

else

    echo "Virtual environment already exists."

fi

echo ""

# ------------------------------------------------------------
# ACTIVATE VIRTUAL ENVIRONMENT
# ------------------------------------------------------------

source "$PROJECT_DIR/.venv/bin/activate"

echo "Virtual environment activated."
echo ""

# ------------------------------------------------------------
# UPGRADE PIP
# ------------------------------------------------------------

echo "Updating pip..."

python -m pip install --upgrade pip

echo ""

# ------------------------------------------------------------
# INSTALL REQUIREMENTS
# ------------------------------------------------------------

if [ -f "$PROJECT_DIR/requirements.txt" ]; then

    echo "Installing FINLOC dependencies..."
    echo ""

    python -m pip install -r "$PROJECT_DIR/requirements.txt"

else

    echo "WARNING: requirements.txt not found."
    echo "Skipping dependency installation."
    echo ""

fi

# ------------------------------------------------------------
# CREATE USER BIN DIRECTORY
# ------------------------------------------------------------

mkdir -p "$HOME/.local/bin"

# ------------------------------------------------------------
# CREATE FINLOC LAUNCHER
# ------------------------------------------------------------

LAUNCHER="$HOME/.local/bin/finloc"

cat > "$LAUNCHER" <<EOF
#!/bin/bash

PROJECT_DIR="$PROJECT_DIR"

cd "\$PROJECT_DIR" || exit 1

source "\$PROJECT_DIR/.venv/bin/activate"

export PYTHONPATH="\$PROJECT_DIR/src"

exec python "\$PROJECT_DIR/src/app/main.py"
EOF

chmod +x "$LAUNCHER"

# ------------------------------------------------------------
# ADD ~/.local/bin TO PATH
# ------------------------------------------------------------

PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'

add_path_to_file() {

    local FILE="$1"

    if [ -f "$FILE" ]; then

        if ! grep -Fxq "$PATH_LINE" "$FILE"; then

            echo "" >> "$FILE"
            echo "# FINLOC AI" >> "$FILE"
            echo "$PATH_LINE" >> "$FILE"

        fi

    fi

}

add_path_to_file "$HOME/.bashrc"
add_path_to_file "$HOME/.profile"

# ------------------------------------------------------------
# UPDATE CURRENT SESSION
# ------------------------------------------------------------

export PATH="$HOME/.local/bin:$PATH"

# ------------------------------------------------------------
# VERIFY INSTALLATION
# ------------------------------------------------------------

echo ""
echo "=============================================="
echo "        FINLOC INSTALLATION COMPLETE"
echo "=============================================="
echo ""

echo "Launcher:"
echo "  $LAUNCHER"
echo ""

if command -v finloc >/dev/null 2>&1; then

    echo "Command:"
    echo "  finloc"
    echo ""

else

    echo "NOTE:"
    echo "Restart your terminal or run:"
    echo ""
    echo "  source ~/.bashrc"
    echo ""

fi

echo "To start FINLOC AI:"
echo ""
echo "  finloc"
echo ""

echo "=============================================="
echo ""
