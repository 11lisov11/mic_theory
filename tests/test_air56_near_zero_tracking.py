import numpy as np

from fem.air56.current_tracking import CurrentMap
from fem.air56.near_zero_tracking import Detector, phase_bins, ramp_profile
from tools.run_air56_loaded_phase_tracking import ScheduledMap, scheduled_observations


def test_linear_reversal_integrates_through_exact_zero():
    rpm, theta = ramp_profile([0, 2, 3, 4, 5], 2., 2., .5, -.5)
    np.testing.assert_allclose(rpm, [.5,.5,0.,-.5,-.5])
    np.testing.assert_allclose(np.rad2deg(theta), [0,6,7.5,6,3])


def test_explicit_rest_class_distinguishes_both_signs_below_old_deadband():
    theta = np.arange(120)*np.pi/120
    cos, sin = np.cos(2*theta), np.sin(2*theta)
    admittance = .10*np.eye(2)+.06*np.array([[cos, sin],[sin,-cos]]).transpose(2,0,1)
    rotor = CurrentMap(np.pi, admittance)
    calibration = ScheduledMap((rotor,)*6, rotor)
    speeds = np.array([-.1,0.,.1])
    times = np.arange(108)/30
    waveforms = scheduled_observations(calibration, np.full(3,.31),speeds,times,100.,1.)
    detector = Detector.build(calibration,1/30,scheduled_observations,12)
    signs, estimated, _ = detector.estimate(phase_bins(waveforms), .0003)
    np.testing.assert_array_equal(signs,[-1,0,1])
    np.testing.assert_allclose(estimated,speeds,atol=.025)
