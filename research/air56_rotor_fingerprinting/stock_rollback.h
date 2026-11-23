/* Simulation reference for a prepared 0.30 A flux; input is transverse stator current.
   One call after each 1 ms sample. Return is the beta voltage for the NEXT interval.
   The first 24 ms only. No device I/O. */
#ifndef MIC_AIR56_STOCK_ROLLBACK_H
#define MIC_AIR56_STOCK_ROLLBACK_H
#include <math.h>

typedef struct {
    unsigned samples;
    unsigned pulse_remaining;
    int motion_sign;
    int pulse_sign;
    double sum_weighted_current;
    double sum_squared_weight;
    double displacement_deg;
} air56_release_reader;

/* Initialise with air56_release_reader r = {0} on each prepared release. */
static inline double air56_release_sample(air56_release_reader *r, double beta_a,
                                         double adc_rms_a, double action_floor_deg) {
    const double gain = -62.77561184070174;
    if (r->samples >= 24) return 0.;
    r->samples++;
    if (!r->pulse_sign) {
        double w = (double)r->samples * r->samples;
        r->sum_weighted_current += w * beta_a;
        r->sum_squared_weight += w * w;
        r->displacement_deg = gain * w * r->sum_weighted_current / r->sum_squared_weight;
        double statistical = 3.5 * fabs(gain) * adc_rms_a * w / sqrt(r->sum_squared_weight);
        int sign = r->displacement_deg < 0. ? -1 : 1;
        if (!r->motion_sign && fabs(r->displacement_deg) >= fmax(.12, statistical))
            r->motion_sign = sign;
        if (fabs(r->displacement_deg) >= fmax(action_floor_deg, statistical)) {
            r->pulse_sign = -sign;
            r->pulse_remaining = 2;
        }
    }
    if (r->samples == 24 || !r->pulse_remaining) return 0.;
    r->pulse_remaining--;
    return 24. * r->pulse_sign;
}
#endif
