/*
 * parabolic_inlet_profile.c
 * Parabolic streamwise velocity profile for a 2D rectangular channel inlet.
 * Fully developed laminar flow; U_max at centerline, zero on walls.
 *
 * Hook as: Boundary Conditions -> velocity_inlet -> Momentum -> X Velocity -> UDF inlet_x_velocity.
 */

#include "udf.h"

#define U_MAX 0.1  /* m/s */
#define Y_MIN 0.0
#define Y_MAX 0.01 /* m */

DEFINE_PROFILE(inlet_x_velocity, thread, position)
{
    real x[ND_ND];
    real y;
    face_t f;

    begin_f_loop(f, thread)
    {
        F_CENTROID(x, f, thread);
        y = x[1];
        F_PROFILE(f, thread, position) =
            U_MAX * 4.0 * (y - Y_MIN) * (Y_MAX - y) / ((Y_MAX - Y_MIN) * (Y_MAX - Y_MIN));
    }
    end_f_loop(f, thread)
}
