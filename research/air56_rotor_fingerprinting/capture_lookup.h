#ifndef AIR56_CAPTURE_LOOKUP_H
#define AIR56_CAPTURE_LOOKUP_H

typedef struct {
    double angle_deg, speed_rad_s, load_nm, reduced_cost, polarity_margin, initial_deg;
    int profile;
} Air56Capture;

/* basis[sample][dimension], table[profile][angle][mean_or_load_slope][dimension+2].
   The last two channels of each table row are motion in degrees and speed in rad/s.
   A reading contains alpha/beta ampere pairs in chronological switching order. */
int air56_capture(const float *reading, const float *center, const float *basis, const float *table,
                  int samples, int dimensions, int angles, int profiles, double noise_a,
                  Air56Capture *out);
/* Signal basis is orthogonal to the removed unit-row vectors from paired calibrations. */
int air56_capture_filtered(const float *reading, const float *center, const float *basis,
                           const float *table, int samples, int dimensions, int angles,
                           int profiles, double noise_a, const float *removed_basis, int removed,
                           Air56Capture *out);
#endif
