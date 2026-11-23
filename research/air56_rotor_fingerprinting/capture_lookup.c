#include "capture_lookup.h"
#include <float.h>
#include <math.h>

enum { MAX_DIM = 96, SECTORS = 8, SUBDIVISIONS = 25, MAX_PROFILES = 2 };
typedef struct {
    double cost, load, angle, movement, speed;
} Fit;

static double wrap(double a) {
    a = fmod(a, 360.0);
    return a < 0 ? a + 360.0 : a;
}

static Fit evaluate(const double *z, const double *row, int dim, double noise) {
    const double *slope = row + dim + 2;
    double dot = 0.0, norm = (noise / .05) * (noise / .05);
    for (int j = 0; j < dim; ++j) {
        dot += (z[j] - row[j]) * slope[j];
        norm += slope[j] * slope[j];
    }
    double load = fmax(-.05, fmin(.05, dot / norm));
    Fit fit = {load * load * (noise / .05) * (noise / .05), load, 0, row[dim] + load * slope[dim],
               row[dim + 1] + load * slope[dim + 1]};
    for (int j = 0; j < dim; ++j) {
        double error = z[j] - row[j] - load * slope[j];
        fit.cost += error * error;
    }
    return fit;
}

int air56_capture_filtered(const float *reading, const float *center, const float *basis,
                           const float *table, int samples, int dim, int angles, int profiles,
                           double noise, const float *removed_basis, int removed,
                           Air56Capture *out) {
    if (!reading || !center || !basis || !table || !out || samples < 2 || dim < 1 ||
        dim > MAX_DIM || dim > samples || angles < SECTORS || profiles < 1 ||
        profiles > MAX_PROFILES || !(noise > 0) || removed < 0 || removed >= samples - 1 ||
        (removed && !removed_basis))
        return 0;
    double z[MAX_DIM] = {0}, total = 0, row[2 * (MAX_DIM + 2)];
    for (int i = 0; i < samples; ++i) {
        double r = (double)reading[i] - center[i];
        if (!isfinite(r))
            return 0;
        total += r * r;
        for (int j = 0; j < dim; ++j)
            z[j] += r * basis[i * dim + j];
    }
    for (int j = 0; j < dim; ++j)
        total -= z[j] * z[j];
    for (int k = 0; k < removed; ++k) {
        double projection = 0;
        for (int i = 0; i < samples; ++i)
            projection += ((double)reading[i] - center[i]) * removed_basis[k * samples + i];
        total -= projection * projection;
    }
    double orthogonal = fmax(0.0, total);
    int width = 2 * (dim + 2), winner = 0;
    Fit sectors[MAX_PROFILES * SECTORS];
    for (int p = 0; p < profiles; ++p) {
        const float *grid = table + p * angles * width;
        for (int s = 0; s < SECTORS; ++s) {
            Fit best = {.cost = DBL_MAX};
            int node = 0;
            for (int k = s * angles / SECTORS; k < (s + 1) * angles / SECTORS; ++k) {
                for (int j = 0; j < width; ++j)
                    row[j] = grid[k * width + j];
                Fit f = evaluate(z, row, dim, noise);
                if (f.cost < best.cost) {
                    best = f;
                    node = k;
                }
            }
            best.angle = node * 360.0 / angles;
            /* Periodic cubic Hermite interpolation with centred grid slopes. */
            for (int k = node - 1; k <= node; ++k) {
                const float *a = grid + ((k - 1 + angles) % angles) * width;
                const float *b = grid + ((k + angles) % angles) * width;
                const float *c = grid + ((k + 1) % angles) * width;
                const float *d = grid + ((k + 2) % angles) * width;
                for (int step = 0; step <= SUBDIVISIONS; ++step) {
                    double x = (double)step / SUBDIVISIONS, x2 = x * x, x3 = x2 * x;
                    double w0 = -.5 * x + x2 - .5 * x3, w1 = 1 - 2.5 * x2 + 1.5 * x3;
                    double w2 = .5 * x + 2 * x2 - 1.5 * x3, w3 = -.5 * x2 + .5 * x3;
                    for (int j = 0; j < width; ++j)
                        row[j] = w0 * a[j] + w1 * b[j] + w2 * c[j] + w3 * d[j];
                    Fit f = evaluate(z, row, dim, noise);
                    if (f.cost < best.cost) {
                        best = f;
                        best.angle = wrap((k + x) * 360.0 / angles);
                    }
                }
            }
            int index = p * SECTORS + s;
            sectors[index] = best;
            if (best.cost < sectors[winner].cost)
                winner = index;
        }
    }
    Fit best = sectors[winner];
    double other = DBL_MAX;
    for (int j = 0; j < profiles * SECTORS; ++j) {
        double delta = fabs(remainder(sectors[j].angle - best.angle, 360.0));
        if (delta >= 120 && sectors[j].cost < other)
            other = sectors[j].cost;
    }
    out->angle_deg = wrap(best.angle + best.movement);
    out->initial_deg = wrap(best.angle);
    out->load_nm = best.load;
    out->speed_rad_s = best.speed;
    out->reduced_cost = (best.cost + orthogonal) / ((samples - removed - 1) * noise * noise);
    out->polarity_margin = (other - best.cost) / (noise * noise);
    out->profile = winner / SECTORS;
    return 1;
}

int air56_capture(const float *reading, const float *center, const float *basis, const float *table,
                  int samples, int dim, int angles, int profiles, double noise, Air56Capture *out) {
    return air56_capture_filtered(reading, center, basis, table, samples, dim, angles, profiles,
                                  noise, 0, 0, out);
}
