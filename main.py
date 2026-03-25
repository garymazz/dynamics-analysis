
import os
import signal
from cement import App
from controllers.base import BaseController
from controllers.hdf5_tools import HDF5Controller
from controllers.analysis_tools import AnalysisController

def handle_signals(app, signum, frame):
    if getattr(app, 'shutdown_initiated', False):
        print("\n[Force Quit] Second signal received. Immediate termination forced.")
        os._exit(1)
    
    app.shutdown_initiated = True
    print("\n\n!!! Termination Signal Received !!!")
    print("Initiating safe shutdown (Press Ctrl+C again to force quit)...")

class DMDProfilerApp(App):
    class Meta:
        label = 'dmd_profiler'
        # Register ALL controllers here
        handlers = [
            BaseController,
            HDF5Controller,
            AnalysisController
        ]
        catch_signals = None

    def setup(self):
        super(DMDProfilerApp, self).setup()
        self.shutdown_initiated = False
        signal.signal(signal.SIGINT, lambda s, f: handle_signals(self, s, f))
        signal.signal(signal.SIGTERM, lambda s, f: handle_signals(self, s, f))

def main():
    with DMDProfilerApp() as app:
        try:
            app.run()
        except Exception as e:
            print(f"\n[Fatal Error] {e}")
            app.exit_code = 1

if __name__ == '__main__':
    main()