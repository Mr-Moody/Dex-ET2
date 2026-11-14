"""Arduino UNO Q edge node (QRB2210 Linux side): filtering, stall classification, RBF-DMP fits.

Pure NumPy + stdlib so this package runs unmodified on the UNO Q's Cortex-A53. Nothing here may
import ``mujoco`` or ``torch``. The laptop talks to it through ``UnoQClient``, which falls back to
an in-process loopback ``UnoQServer`` when no board answers.
"""
