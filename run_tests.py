#!/usr/bin/env python3
"""Run pytest from any directory."""

import subprocess
import sys
from pathlib import Path

chatbot_dir = Path(__file__).parent
sys.path.insert(0, str(chatbot_dir))

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    cwd=chatbot_dir,
)
sys.exit(result.returncode)
