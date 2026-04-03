from cement import Controller, ex

# --- DOMAIN-DRIVEN IMPORTS ---
from utils.hdf5._main import HDF5Manager, SCHEMA_VERSION, SCRIPT_VERSION

class HDF5Controller(Controller):
    class Meta:
        label = 'hdf5'
        stacked_on = 'base'         # <-- Stacked on the root base controller
        stacked_type = 'embedded'     # <-- Attaches sub-command to base controller without creating a new CLI layer (i.e., `python main.py cluster ...` instead of `python main.py cluster cluster ...`)
        description = 'HDF5 schema inspection and dataset repair tools'

    @ex(help='Print HDF5 utility help menu')
    def hdf5(self):
        # Enforces your rule: default ONLY prints help
        self.app.args.print_help()

    @ex(
        help='Print the hierarchical schema of an HDF5 file.',
        arguments=[
            (['target_file'], {'help': 'Path to the HDF5 file'}),
        ]
    )
    def schema(self):
        """CLI Routing for: python main.py hdf5 schema <file>"""
        print(f"HDF5 Diagnostic Tools (v{SCRIPT_VERSION} | Schema v{SCHEMA_VERSION})")
        HDF5Manager.print_schema(self.app.pargs.target_file)

    @ex(
        help='Inspect an HDF5 file for truncated datasets or structural corruption.',
        arguments=[
            (['target_file'], {'help': 'Path to the HDF5 file'}),
        ]
    )
    def inspect(self):
        """CLI Routing for: python main.py hdf5 inspect <file>"""
        print(f"HDF5 Diagnostic Tools (v{SCRIPT_VERSION} | Schema v{SCHEMA_VERSION})")
        HDF5Manager.inspect_file(self.app.pargs.target_file)

    @ex(
        help='Attempt to safely repair a corrupted HDF5 file by truncating damaged tail entries.',
        arguments=[
            (['target_file'], {'help': 'Path to the HDF5 file'}),
        ]
    )
    def fix(self):
        """CLI Routing for: python main.py hdf5 fix <file>"""
        print(f"HDF5 Diagnostic Tools (v{SCRIPT_VERSION} | Schema v{SCHEMA_VERSION})")
        HDF5Manager.repair_file(self.app.pargs.target_file)