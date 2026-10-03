"""`python -m pyfaceau VIDEO [-o OUTPUT.csv]`: the same as the `pyfaceau` command."""

import sys

from .processor import main

sys.exit(main())
