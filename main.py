import os
import signal
from cement import App, Controller, ex

# --- UPDATED DOMAIN-DRIVEN IMPORTS ---
from analysis.dmd.controller import DMDController
from analysis.cluster.controller import ClusterController
from meta_analysis.period.controller import PeriodController
from utils.hdf5.controller import HDF5Controller

def handle_signals(app, signum, frame):
    if getattr(app, 'shutdown_initiated', False):
        print("\n[Force Quit] Second signal received. Immediate termination forced.")
        os._exit(1)
    
    app.shutdown_initiated = True
    print("\n\n!!! Termination Signal Received !!!")
    print("Initiating safe shutdown (Press Ctrl+C again to force quit)...")


# ==========================================
# 1. THE ROOT ROUTER (Anchor for sub-commands)
# ==========================================
class BaseController(Controller):
    """The neutral root application controller."""
    class Meta:
        label = 'base'
        description = 'Dynamics Analysis CLI Application'

    @ex(help='Default action (prints help menu)')
    def default(self):
        # When a user types `python main.py` with no args, it prints the menu.
        self.app.args.print_help()

# ==========================================
# 2. THE APPLICATION CONTAINER
# ==========================================
class DMDProfilerApp(App):
    class Meta:
        label = 'dmd_profiler'
        version = '9.5.0'
        # Register the root BaseController AND all domain controllers here
        handlers = [
            BaseController,     # Defined right above
            DMDController,      # Imported from analysis.dmd
            HDF5Controller,     # Imported from utils.hdf5
            PeriodController,   # Imported from meta_analysis.period
            ClusterController   # Imported from analysis.cluster
        ]
        catch_signals = None    # Register all imported controllers here

    def setup(self):
        super(DMDProfilerApp, self).setup()
        self.shutdown_initiated = False
        signal.signal(signal.SIGINT, lambda s, f: handle_signals(self, s, f))
        signal.signal(signal.SIGTERM, lambda s, f: handle_signals(self, s, f))

# ==========================================
# 3. BOOTSTRAP
# ==========================================
def main():
    with DMDProfilerApp() as app:
        try:
            app.run()
        except Exception as e:
            print(f"\n[Fatal Error] {e}")
            app.exit_code = 1

if __name__ == '__main__':
    main()