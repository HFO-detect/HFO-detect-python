"""
Init for sim submodule.
"""

from .create_simulated import pinknoise, brownnoise, wavelet, hfo
from .create_simulated import wavelet_spike

__all__=['pinknoise', 'brownnoise','wavelet', 'hfo', 'wavelet_spike']