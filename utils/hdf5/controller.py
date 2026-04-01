
from cement import Controller, ex
from core.hdf5_manager import analyze_and_fix_hdf5, print_hdf5_schema

class HDF5Controller(Controller):
    class Meta:
        label = 'hdf5'
        stacked_on = 'base'
        stacked_type = 'nested'
        description = 'HDF5 Diagnostic, Schema, and Repair Tools'

    @ex(
        help='Analyze and safely repair an HDF5 file by truncating corrupt entries.',
        arguments=[(['file_path'], {'help': 'Path to the HDF5 file to fix'})]
    )
    def fix(self):
        """CLI Routing for: python main.py hdf5 fix <file>"""
        file_path = self.app.pargs.file_path
        abort_check = lambda: getattr(self.app, 'shutdown_initiated', False)
        analyze_and_fix_hdf5(file_path, fix=True, abort_check=abort_check)

    @ex(
        help='Print a clean, formatted report of the hierarchical schema embedded in the file.',
        arguments=[(['file_path'], {'help': 'Path to the HDF5 file'})]
    )
    def schema(self):
        """CLI Routing for: python main.py hdf5 schema <file>"""
        print_hdf5_schema(self.app.pargs.file_path)

    @ex(
        help='Analyze the HDF5 file to find the last known good configuration without altering the file.',
        arguments=[(['file_path'], {'help': 'Path to the HDF5 file'})]
    )
    def inspect(self):
        """CLI Routing for: python main.py hdf5 inspect <file>"""
        file_path = self.app.pargs.file_path
        abort_check = lambda: getattr(self.app, 'shutdown_initiated', False)
        analyze_and_fix_hdf5(file_path, fix=False, abort_check=abort_check)